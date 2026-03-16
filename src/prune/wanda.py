import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import config_to_dict, load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune a model with Wanda.")
    parser.add_argument("-s", "--sparsity-ratio", type=float, default=0.6, help="Prune sparsity ratio, e.g. 0.5 or 0.7.")
    return parser.parse_args()


def _load_prompts(path: str, limit: int) -> List[str]:
    samples: List[str] = []
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Calibration dataset not found: {path}")

    with file_path.open("r", encoding="utf-8") as handle:
        if file_path.suffix == ".jsonl":
            for line in handle:
                row = json.loads(line)
                prompt = _extract_prompt(row)
                if prompt:
                    samples.append(prompt)
                if len(samples) >= limit:
                    break
        else:
            rows = json.load(handle)
            for row in rows:
                prompt = _extract_prompt(row)
                if prompt:
                    samples.append(prompt)
                if len(samples) >= limit:
                    break

    if not samples:
        raise ValueError(f"No usable prompts found in calibration dataset: {path}")
    return samples


def _extract_prompt(row: object) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("prompt", "question", "input", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _iter_target_linears(model: nn.Module, target_names: Iterable[str]) -> Dict[str, nn.Linear]:
    suffixes = tuple(target_names)
    modules: Dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(suffixes):
            modules[name] = module
    if not modules:
        raise ValueError("No target linear layers were found for Wanda pruning.")
    return modules


@dataclass
class ActivationStat:
    sum_sq: torch.Tensor
    count: int = 0

    def update(self, tensor: torch.Tensor) -> None:
        flat = tensor.detach().float().reshape(-1, tensor.shape[-1])
        self.sum_sq += flat.pow(2).sum(dim=0).cpu()
        self.count += flat.shape[0]

    def rms(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("Activation statistics are empty.")
        return torch.sqrt(self.sum_sq / self.count).clamp_min(1e-12)


def _collect_activation_stats(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: str,
    batch_size: int,
    max_length: int,
    target_names: Iterable[str],
) -> Dict[str, ActivationStat]:
    modules = _iter_target_linears(model, target_names)
    stats = {
        name: ActivationStat(sum_sq=torch.zeros(module.in_features, dtype=torch.float32))
        for name, module in modules.items()
    }
    hooks = []

    for name, module in modules.items():
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor, key: str = name) -> None:
            if not inputs:
                return
            stats[key].update(inputs[0])

        hooks.append(module.register_forward_hook(hook))

    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                model(**encoded, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    return stats


def _apply_wanda_pruning(
    model: AutoModelForCausalLM,
    stats: Dict[str, ActivationStat],
    target_names: Iterable[str],
    sparsity_ratio: float,
) -> Dict[str, float]:
    if not 0.0 < sparsity_ratio < 1.0:
        raise ValueError(f"sparsity_ratio must be in (0, 1), got {sparsity_ratio}")

    modules = _iter_target_linears(model, target_names)
    layer_sparsity: Dict[str, float] = {}

    for name, module in modules.items():
        weights = module.weight.data
        act = stats[name].rms().to(weights.device, dtype=weights.dtype)
        scores = weights.abs() * act.unsqueeze(0)
        prune_count = int(scores.shape[1] * sparsity_ratio)
        if prune_count <= 0:
            layer_sparsity[name] = 0.0
            continue

        prune_idx = torch.topk(scores, k=prune_count, dim=1, largest=False).indices
        mask = torch.zeros_like(weights, dtype=torch.bool)
        mask.scatter_(1, prune_idx, True)
        weights[mask] = 0
        layer_sparsity[name] = float(mask.float().mean().item())

    return layer_sparsity


def _count_zero_weights(model: nn.Module, target_names: Iterable[str]) -> tuple[int, int]:
    zeros = 0
    total = 0
    for _, module in _iter_target_linears(model, target_names).items():
        tensor = module.weight.data
        zeros += int((tensor == 0).sum().item())
        total += tensor.numel()
    return zeros, total


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    prune_cfg = cfg.prune
    if args.sparsity_ratio is not None:
        prune_cfg.sparsity_ratio = args.sparsity_ratio

    random.seed(prune_cfg.seed)
    torch.manual_seed(prune_cfg.seed)

    output_dir = Path(prune_cfg.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(prune_cfg.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        prune_cfg.model_path,
        torch_dtype=prune_cfg.torch_dtype,
        device_map=None,
    )
    model.to(prune_cfg.device)
    model.eval()

    prompts = _load_prompts(prune_cfg.calibration_data_path, prune_cfg.calibration_samples)
    activation_stats = _collect_activation_stats(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=prune_cfg.device,
        batch_size=prune_cfg.batch_size,
        max_length=prune_cfg.max_length,
        target_names=prune_cfg.target_linear_names,
    )
    layer_sparsity = _apply_wanda_pruning(
        model=model,
        stats=activation_stats,
        target_names=prune_cfg.target_linear_names,
        sparsity_ratio=prune_cfg.sparsity_ratio,
    )
    zero_count, total_count = _count_zero_weights(model, prune_cfg.target_linear_names)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "config": config_to_dict(cfg),
        "prune": {
            "method": "wanda",
            "layer_sparsity": layer_sparsity,
            "zero_params": zero_count,
            "total_params": total_count,
            "global_sparsity": (zero_count / total_count) if total_count else 0.0,
            "calibration_prompt_count": len(prompts),
            "output_dir": str(output_dir),
        },
    }
    with (output_dir / "prune_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wanda pruning complete. Saved pruned model to: {output_dir}")
    print(f"Global sparsity over target linear layers: {summary['prune']['global_sparsity']:.2%}")


if __name__ == "__main__":
    main()
