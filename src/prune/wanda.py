import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import config_to_dict, load_config


ALPACA_RATIO = 0.8
PKU_RATIO = 0.2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune a model with Wanda.")
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("-m", "--model", type=str, default=None, help="Base model directory to prune.")
    parser.add_argument("-a", "--adapter-path", type=str, default=None, help="Optional LoRA adapter directory to merge before pruning.")
    parser.add_argument("-s", "--sparsity-ratio", type=float, default=None, help="Prune sparsity ratio, e.g. 0.5 or 0.7.")
    return parser.parse_args()


def _load_rows(path: str) -> List[dict]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Calibration dataset not found: {path}")

    rows: List[dict] = []
    with file_path.open("r", encoding="utf-8") as handle:
        if file_path.suffix == ".jsonl":
            for line in handle:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
        else:
            payload = json.load(handle)
            if isinstance(payload, list):
                rows = [row for row in payload if isinstance(row, dict)]

    if not rows:
        raise ValueError(f"No usable rows found in calibration dataset: {path}")
    return rows


def _extract_prompt(row: object) -> str:
    if not isinstance(row, dict):
        return ""

    instruction = str(row.get("instruction", "")).strip()
    model_input = str(row.get("input", "")).strip()
    if instruction and model_input:
        return f"Instruction: {instruction}\nInput: {model_input}"
    if instruction:
        return instruction

    for key in ("prompt", "question", "input", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _sample_prompts(rows: List[dict], sample_size: int, seed: int) -> List[str]:
    prompts = [prompt for prompt in (_extract_prompt(row) for row in rows) if prompt]
    if not prompts:
        return []

    rng = random.Random(seed)
    rng.shuffle(prompts)
    if sample_size >= len(prompts):
        return prompts
    return prompts[:sample_size]


def _sample_tagged_prompts(rows: List[dict], sample_size: int, seed: int, source: str) -> List[Tuple[str, str]]:
    prompts = _sample_prompts(rows, sample_size, seed)
    return [(prompt, source) for prompt in prompts]


def _load_mixed_calibration_prompts(cfg, limit: int, seed: int) -> Tuple[List[str], Dict[str, object]]:
    if limit <= 0:
        raise ValueError(f"calibration_samples must be positive, got {limit}")

    alpaca_rows = _load_rows(cfg.alpaca_dataset_path)
    pku_rows = _load_rows(cfg.align.train_dataset_path)

    alpaca_target = int(round(limit * ALPACA_RATIO))
    alpaca_target = min(max(alpaca_target, 0), limit)
    pku_target = limit - alpaca_target

    alpaca_prompts = _sample_prompts(alpaca_rows, alpaca_target, seed)
    pku_prompts = _sample_prompts(pku_rows, pku_target, seed + 1)

    deficit = limit - (len(alpaca_prompts) + len(pku_prompts))
    if deficit > 0:
        extra_alpaca = _sample_tagged_prompts(alpaca_rows, limit, seed + 2, "alpaca")
        extra_pku = _sample_tagged_prompts(pku_rows, limit, seed + 3, "pku")
        used = set(alpaca_prompts) | set(pku_prompts)
        for prompt, source in extra_alpaca + extra_pku:
            if prompt in used:
                continue
            if len(alpaca_prompts) + len(pku_prompts) >= limit:
                break
            used.add(prompt)
            if source == "alpaca":
                alpaca_prompts.append(prompt)
            else:
                pku_prompts.append(prompt)

    prompts = alpaca_prompts + pku_prompts
    if len(prompts) < limit:
        raise ValueError(
            f"Unable to build {limit} calibration prompts from Alpaca and PKU-SafeRLHF; "
            f"got {len(prompts)} prompts."
        )

    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:limit], {
        "alpaca_dataset_path": cfg.alpaca_dataset_path,
        "pku_dataset_path": cfg.align.train_dataset_path,
        "alpaca_ratio": ALPACA_RATIO,
        "pku_ratio": PKU_RATIO,
        "alpaca_prompt_count": len(alpaca_prompts),
        "pku_prompt_count": len(pku_prompts),
        "total_prompt_count": min(len(prompts), limit),
    }


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


def _looks_like_adapter_dir(path: str | None) -> bool:
    if not path:
        return False
    return (Path(path) / 'adapter_config.json').is_file()


def _resolve_model_and_adapter_paths(model_path: str, adapter_path: str | None) -> tuple[str, str | None]:
    if adapter_path:
        return model_path, adapter_path

    if _looks_like_adapter_dir(model_path):
        adapter_cfg_path = Path(model_path) / 'adapter_config.json'
        with adapter_cfg_path.open('r', encoding='utf-8') as handle:
            adapter_cfg = json.load(handle)
        base_model_path = str(adapter_cfg.get('base_model_name_or_path', '')).strip()
        if not base_model_path:
            raise ValueError(f'LoRA adapter config is missing base_model_name_or_path: {adapter_cfg_path}')
        return base_model_path, model_path

    return model_path, None


def _load_model_for_pruning(model_path: str, adapter_path: str | None, torch_dtype: torch.dtype):
    tokenizer_source = adapter_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                'Pruning a LoRA-aligned model requires `peft`. Install it before running wanda.py.'
            ) from exc
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    return model, tokenizer


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    prune_cfg = cfg.prune
    if args.name is not None:
        prune_cfg.model_name = args.name
    if args.model is not None:
        prune_cfg.model_path = args.model
    if args.adapter_path is not None:
        prune_cfg.lora_adapter_path = args.adapter_path
    if args.sparsity_ratio is not None:
        prune_cfg.sparsity_ratio = args.sparsity_ratio

    resolved_model_path, resolved_adapter_path = _resolve_model_and_adapter_paths(
        prune_cfg.model_path,
        prune_cfg.lora_adapter_path,
    )

    random.seed(prune_cfg.seed)
    torch.manual_seed(prune_cfg.seed)

    output_dir = Path(prune_cfg.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model_for_pruning(
        model_path=resolved_model_path,
        adapter_path=resolved_adapter_path,
        torch_dtype=prune_cfg.torch_dtype,
    )
    model.to(prune_cfg.device)
    model.eval()

    prompts, calibration_mix = _load_mixed_calibration_prompts(
        cfg=cfg,
        limit=prune_cfg.calibration_samples,
        seed=prune_cfg.seed,
    )
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
        'config': config_to_dict(cfg),
        'prune': {
            'method': 'wanda',
            'base_model_path': resolved_model_path,
            'lora_adapter_path': resolved_adapter_path or '',
            'layer_sparsity': layer_sparsity,
            'zero_params': zero_count,
            'total_params': total_count,
            'global_sparsity': (zero_count / total_count) if total_count else 0.0,
            'calibration_prompt_count': len(prompts),
            'calibration_mix': calibration_mix,
            'output_dir': str(output_dir),
        },
    }
    with (output_dir / 'prune_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if resolved_adapter_path:
        print(f'Merged LoRA adapter before pruning: {resolved_adapter_path}')
    print(f'Wanda pruning complete. Saved pruned model to: {output_dir}')
    print(f"Global sparsity over target linear layers: {summary['prune']['global_sparsity']:.2%}")


if __name__ == '__main__':
    main()
