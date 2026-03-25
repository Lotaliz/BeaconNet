import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from config import load_config
from src.activation.common import load_model, load_tokenizer
from src.activation.hooks import SAELatentPatcher
from src.activation.train_sae import SparseAutoencoder
from src.eval.safety2 import _extract_prompt, _load_json_dataset, _load_xguard, _safe_safe_score, _sample_rows, _xguard_response_safety


XGUARD_RELATIVE_PATH = Path("models") / "XGuard-8B"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch SAE latents during generation and test safety impact.")
    parser.add_argument("-m", "--model-path", default=None)
    parser.add_argument("-a", "--adapter-path", default=None)
    parser.add_argument("--sae-checkpoint-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--patch-mode", action="append", default=None)
    parser.add_argument("--feature-top-k", type=int, default=None)
    parser.add_argument("--patch-strength", type=float, default=None)
    return parser.parse_args()


def _runtime_config(patch_cfg, model_path: str, adapter_path: str, sae_checkpoint_dir: str, output_dir: Path, sample_size: int, patch_modes: Tuple[str, ...], feature_top_k: int, patch_strength: float) -> Dict[str, Any]:
    return {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "sae_checkpoint_dir": sae_checkpoint_dir,
        "output_dir": str(output_dir),
        "dataset_paths": patch_cfg.dataset_paths,
        "sample_size": sample_size,
        "seed": patch_cfg.seed,
        "patch_modes": list(patch_modes),
        "feature_top_k": feature_top_k,
        "patch_strength": patch_strength,
        "max_length": patch_cfg.max_length,
        "generation_max_new_tokens": patch_cfg.generation_max_new_tokens,
        "xguard_max_new_tokens": patch_cfg.xguard_max_new_tokens,
        "device": patch_cfg.device,
        "torch_dtype": str(patch_cfg.torch_dtype),
    }


def _load_sae_bundles(checkpoint_root: str, device: str, feature_top_k: int) -> Dict[str, Dict[str, Any]]:
    root = Path(checkpoint_root)
    if not root.is_dir():
        raise FileNotFoundError(f"SAE checkpoint directory not found: {root}")

    bundles: Dict[str, Dict[str, Any]] = {}
    for layer_dir in sorted(root.iterdir()):
        if not layer_dir.is_dir():
            continue
        checkpoint_path = layer_dir / "sae.pt"
        if not checkpoint_path.is_file():
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        sae = SparseAutoencoder(input_dim=int(checkpoint["input_dim"]), hidden_dim=int(checkpoint["hidden_dim"])).to(device)
        sae.load_state_dict(checkpoint["state_dict"])
        sae.eval()

        positive = list(checkpoint.get("top_positive_features", []))[:feature_top_k]
        negative = list(checkpoint.get("top_negative_features", []))[:feature_top_k]
        layer_name = str(checkpoint["layer_name"])
        bundles[layer_name] = {
            "sae": sae,
            "mean": checkpoint["mean"].float().squeeze(0).to(device),
            "std": checkpoint["std"].float().squeeze(0).clamp_min(1e-6).to(device),
            "positive_indices": [int(item["feature_index"]) for item in positive],
            "positive_weights": [abs(float(item["correlation"])) for item in positive],
            "negative_indices": [int(item["feature_index"]) for item in negative],
            "negative_weights": [abs(float(item["correlation"])) for item in negative],
            "checkpoint_path": str(checkpoint_path),
        }

    if not bundles:
        raise ValueError(f"No SAE checkpoints found in: {root}")
    return bundles


def _generate_responses(model, tokenizer, prompts: List[str], max_length: int, max_new_tokens: int, desc: str) -> List[str]:
    responses: List[str] = []
    for prompt in tqdm(prompts, desc=desc):
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
        else:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[-1]
        response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
        responses.append(response)
    return responses


def _score_responses(guard_model, guard_tokenizer, prompts: List[str], responses: List[str], max_new_tokens: int, desc: str) -> List[Dict[str, Any]]:
    scores: List[Dict[str, Any]] = []
    for prompt, response in tqdm(list(zip(prompts, responses)), total=len(prompts), desc=desc):
        scores.append(
            _xguard_response_safety(
                guard_model=guard_model,
                guard_tokenizer=guard_tokenizer,
                prompt=prompt,
                response=response,
                max_new_tokens=max_new_tokens,
            )
        )
    return scores


def _build_sample_records(rows: List[Dict[str, Any]], prompts: List[str], baseline_responses: List[str], baseline_scores: List[Dict[str, Any]], patched_responses: List[str], patched_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row, prompt, baseline_response, baseline_score, patched_response, patched_score in zip(
        rows,
        prompts,
        baseline_responses,
        baseline_scores,
        patched_responses,
        patched_scores,
    ):
        baseline_safe = _safe_safe_score(baseline_score["risk_score"])
        patched_safe = _safe_safe_score(patched_score["risk_score"])
        records.append(
            {
                "prompt": prompt,
                "meta": row,
                "baseline_response": baseline_response,
                "baseline_xguard_risk_score": baseline_score["risk_score"],
                "baseline_safe_safe_score": baseline_safe,
                "patched_response": patched_response,
                "patched_xguard_risk_score": patched_score["risk_score"],
                "patched_safe_safe_score": patched_safe,
                "safe_safe_delta": patched_safe - baseline_safe,
            }
        )
    return records


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    patch_cfg = cfg.sae_patch

    model_path = args.model_path or patch_cfg.model_path
    adapter_path = args.adapter_path if args.adapter_path is not None else (patch_cfg.lora_adapter_path if patch_cfg.use_lora_adapter else "")
    sae_checkpoint_dir = args.sae_checkpoint_dir or patch_cfg.sae_checkpoint_dir
    output_dir = Path(args.output_dir or patch_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_size = args.sample_size or patch_cfg.sample_size
    patch_modes = tuple(args.patch_mode or patch_cfg.patch_modes)
    feature_top_k = args.feature_top_k or patch_cfg.feature_top_k
    patch_strength = args.patch_strength or patch_cfg.patch_strength

    torch.manual_seed(patch_cfg.seed)
    random.seed(patch_cfg.seed)

    bundles = _load_sae_bundles(sae_checkpoint_dir, device=patch_cfg.device, feature_top_k=feature_top_k)
    tokenizer = load_tokenizer(adapter_path or model_path)
    model = load_model(model_path=model_path, adapter_path=adapter_path, dtype=patch_cfg.torch_dtype, device=patch_cfg.device)
    xguard_model_path = str(Path(cfg.project_root) / XGUARD_RELATIVE_PATH)
    guard_model, guard_tokenizer = _load_xguard(xguard_model_path, patch_cfg.device)

    summary_rows: List[Dict[str, Any]] = []
    try:
        for dataset_name, dataset_path in patch_cfg.dataset_paths.items():
            rows = _load_json_dataset(dataset_path)
            sampled_rows = _sample_rows(rows, sample_size=sample_size, seed=patch_cfg.seed)
            prompts = [_extract_prompt(row) for row in sampled_rows]

            baseline_responses = _generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_length=patch_cfg.max_length,
                max_new_tokens=patch_cfg.generation_max_new_tokens,
                desc=f"{dataset_name}:baseline:generate",
            )
            baseline_scores = _score_responses(
                guard_model=guard_model,
                guard_tokenizer=guard_tokenizer,
                prompts=prompts,
                responses=baseline_responses,
                max_new_tokens=patch_cfg.xguard_max_new_tokens,
                desc=f"{dataset_name}:baseline:xguard",
            )
            baseline_mean = sum(_safe_safe_score(item["risk_score"]) for item in baseline_scores) / max(len(baseline_scores), 1)

            for patch_mode in patch_modes:
                mode_output_dir = output_dir / patch_mode
                mode_output_dir.mkdir(parents=True, exist_ok=True)
                patcher = SAELatentPatcher(
                    model=model,
                    module_bundles=bundles,
                    mode=patch_mode,
                    strength=patch_strength,
                )
                patcher.register()
                try:
                    patched_responses = _generate_responses(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        max_length=patch_cfg.max_length,
                        max_new_tokens=patch_cfg.generation_max_new_tokens,
                        desc=f"{dataset_name}:{patch_mode}:generate",
                    )
                finally:
                    patcher.remove()

                patched_scores = _score_responses(
                    guard_model=guard_model,
                    guard_tokenizer=guard_tokenizer,
                    prompts=prompts,
                    responses=patched_responses,
                    max_new_tokens=patch_cfg.xguard_max_new_tokens,
                    desc=f"{dataset_name}:{patch_mode}:xguard",
                )
                records = _build_sample_records(
                    rows=sampled_rows,
                    prompts=prompts,
                    baseline_responses=baseline_responses,
                    baseline_scores=baseline_scores,
                    patched_responses=patched_responses,
                    patched_scores=patched_scores,
                )
                patched_mean = sum(item["patched_safe_safe_score"] for item in records) / max(len(records), 1)
                payload = {
                    "config": _runtime_config(
                        patch_cfg=patch_cfg,
                        model_path=model_path,
                        adapter_path=adapter_path,
                        sae_checkpoint_dir=sae_checkpoint_dir,
                        output_dir=output_dir,
                        sample_size=sample_size,
                        patch_modes=patch_modes,
                        feature_top_k=feature_top_k,
                        patch_strength=patch_strength,
                    ),
                    "dataset_name": dataset_name,
                    "dataset_path": dataset_path,
                    "patch_mode": patch_mode,
                    "sample_count": len(records),
                    "baseline_mean_safe_safe_score": baseline_mean,
                    "patched_mean_safe_safe_score": patched_mean,
                    "mean_safe_safe_delta": patched_mean - baseline_mean,
                    "patched_modules": sorted(bundles),
                    "results": records,
                }
                output_path = mode_output_dir / f"{dataset_name}.json"
                with output_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                print(f"Saved SAE patch results to: {output_path}")
                summary_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "patch_mode": patch_mode,
                        "sample_count": len(records),
                        "baseline_mean_safe_safe_score": baseline_mean,
                        "patched_mean_safe_safe_score": patched_mean,
                        "mean_safe_safe_delta": patched_mean - baseline_mean,
                        "output_path": str(output_path),
                    }
                )
    finally:
        del model
        del guard_model
        if patch_cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_payload = {
        "config": _runtime_config(
            patch_cfg=patch_cfg,
            model_path=model_path,
            adapter_path=adapter_path,
            sae_checkpoint_dir=sae_checkpoint_dir,
            output_dir=output_dir,
            sample_size=sample_size,
            patch_modes=patch_modes,
            feature_top_k=feature_top_k,
            patch_strength=patch_strength,
        ),
        "summaries": summary_rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
    print(f"Saved SAE patch summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
