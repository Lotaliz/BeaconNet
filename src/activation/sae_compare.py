import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm

from config import load_config
from src.activation.common import build_prompt_response_inputs, load_model, load_tokenizer, move_to_device, sanitize_filename
from src.activation.hooks import SampleActivationCollector
from src.activation.train_sae import SparseAutoencoder


@dataclass
class SafetyExample:
    sample_id: str
    dataset_name: str
    prompt: str
    response: str
    safety_score: float
    model_path: str
    source_file: str


@dataclass
class FeatureSpec:
    feature_index: int
    correlation: float
    role: str
    weight: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two models under the same SAE and relate latent shifts to safety2 scores.")
    parser.add_argument("--baseline-model-path", default=None)
    parser.add_argument("--baseline-adapter-path", default=None)
    parser.add_argument("--baseline-safety-dir", default=None)
    parser.add_argument("--compressed-model-path", default=None)
    parser.add_argument("--compressed-adapter-path", default=None)
    parser.add_argument("--compressed-safety-dir", default=None)
    parser.add_argument("--sae-checkpoint-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--aggregation-method", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _load_safety_examples(source_dir: str, max_samples: int) -> Dict[Tuple[str, str], SafetyExample]:
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Safety2 directory not found: {root}")

    examples: Dict[Tuple[str, str], SafetyExample] = {}
    for path in sorted(root.glob("*.json")):
        payload = _load_json(path)
        dataset_name = str(payload.get("dataset_name") or path.stem)
        model_path = str(payload.get("model_path", ""))
        results = payload.get("results", [])
        if not isinstance(results, list):
            continue
        if max_samples > 0:
            results = results[:max_samples]
        for index, row in enumerate(results):
            if not isinstance(row, dict):
                continue
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("model_output", "")).strip()
            if not prompt or not response:
                continue
            risk_score = row.get("xguard_risk_score", {})
            safe_safe = float(risk_score.get("Safe-Safe", 0.0)) if isinstance(risk_score, dict) else 0.0
            key = (dataset_name, prompt)
            examples[key] = SafetyExample(
                sample_id=f"{dataset_name}-{index:05d}",
                dataset_name=dataset_name,
                prompt=prompt,
                response=response,
                safety_score=safe_safe,
                model_path=model_path,
                source_file=str(path),
            )
    if not examples:
        raise ValueError(f"No usable safety2 examples found in {root}")
    return examples


def _align_examples(
    baseline: Dict[Tuple[str, str], SafetyExample],
    compressed: Dict[Tuple[str, str], SafetyExample],
) -> List[Tuple[SafetyExample, SafetyExample]]:
    keys = sorted(set(baseline) & set(compressed), key=lambda item: (item[0], item[1]))
    if not keys:
        raise ValueError("No overlapping prompts found between baseline and compressed safety2 directories.")
    return [(baseline[key], compressed[key]) for key in keys]


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    if x.numel() < 2 or y.numel() < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.pow(2).sum() * y.pow(2).sum()).clamp_min(1e-8))
    return float((x * y).sum() / denom)


def _rank_tensor(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, dim=0)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, values.numel() + 1, dtype=torch.float32)
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return _pearson(_rank_tensor(x.float()), _rank_tensor(y.float()))


def _load_sae_bundles(checkpoint_root: str) -> Dict[str, Dict[str, Any]]:
    root = Path(checkpoint_root)
    if not root.is_dir():
        raise FileNotFoundError(f"SAE checkpoint directory not found: {root}")
    bundles: Dict[str, Dict[str, Any]] = {}
    for layer_dir in sorted(root.iterdir()):
        if not layer_dir.is_dir():
            continue
        checkpoint_path = layer_dir / "sae.pt"
        metrics_path = layer_dir / "metrics.json"
        if not checkpoint_path.is_file():
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        metrics = _load_json(metrics_path) if metrics_path.is_file() else {}
        layer_name = str(checkpoint["layer_name"])
        bundles[layer_name] = {
            "checkpoint": checkpoint,
            "metrics": metrics,
            "checkpoint_path": str(checkpoint_path),
        }
    if not bundles:
        raise ValueError(f"No SAE checkpoints found under {root}")
    return bundles


def _build_feature_specs(bundle: Dict[str, Any]) -> List[FeatureSpec]:
    checkpoint = bundle["checkpoint"]
    positive = checkpoint.get("top_positive_features") or bundle["metrics"].get("top_positive_features", [])
    negative = checkpoint.get("top_negative_features") or bundle["metrics"].get("top_negative_features", [])
    features: Dict[int, FeatureSpec] = {}
    for row in positive:
        index = int(row["feature_index"])
        corr = float(row["correlation"])
        features[index] = FeatureSpec(feature_index=index, correlation=corr, role="positive", weight=abs(corr))
    for row in negative:
        index = int(row["feature_index"])
        corr = float(row["correlation"])
        features[index] = FeatureSpec(feature_index=index, correlation=corr, role="negative", weight=abs(corr))
    return [features[index] for index in sorted(features)]


def _instantiate_sae(checkpoint: Dict[str, Any], device: str) -> SparseAutoencoder:
    model = SparseAutoencoder(input_dim=int(checkpoint["input_dim"]), hidden_dim=int(checkpoint["hidden_dim"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _collect_model_activations(
    model_path: str,
    adapter_path: str,
    device: str,
    dtype: torch.dtype,
    max_length: int,
    token_aggregation: str,
    module_names: Iterable[str],
    samples: List[SafetyExample],
    desc: str,
) -> Dict[str, torch.Tensor]:
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path=model_path, adapter_path=adapter_path, dtype=dtype, device=device)
    collector = SampleActivationCollector(model=model, module_names=module_names, token_aggregation=token_aggregation)
    collector.register()
    per_module: Dict[str, List[torch.Tensor]] = {name: [] for name in module_names}
    try:
        for sample in tqdm(samples, desc=desc):
            encoded, token_span = build_prompt_response_inputs(
                tokenizer=tokenizer,
                prompt=sample.prompt,
                response=sample.response,
                max_length=max_length,
            )
            batch = move_to_device(encoded, device)
            collector.clear()
            collector.set_token_span(*token_span)
            with torch.no_grad():
                _ = model(**batch, use_cache=False)
            vectors = collector.sample_vectors()
            for module_name in module_names:
                per_module[module_name].append(vectors[module_name].float())
    finally:
        collector.remove()
        del model
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {name: torch.stack(vectors, dim=0) for name, vectors in per_module.items()}


def _aggregate_scores(
    latent_values: torch.Tensor,
    feature_specs: List[FeatureSpec],
    method: str,
    top_k: int,
    z_values: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if not feature_specs:
        zeros = torch.zeros(latent_values.shape[0], dtype=torch.float32)
        return {
            "positive_safety_score": zeros,
            "negative_safety_score": zeros,
            "safety_probe_score": zeros,
        }

    weights = torch.tensor([feature.weight for feature in feature_specs], dtype=torch.float32)
    roles = [feature.role for feature in feature_specs]
    values = latent_values.float()
    if method == "standardized_weighted_sum":
        if z_values is None:
            raise ValueError("standardized_weighted_sum requires z_values")
        source = z_values.float()
    else:
        source = values

    contributions = source * weights.unsqueeze(0)
    pos_indices = [idx for idx, role in enumerate(roles) if role == "positive"]
    neg_indices = [idx for idx, role in enumerate(roles) if role == "negative"]

    def _reduce(indices: List[int]) -> torch.Tensor:
        if not indices:
            return torch.zeros(source.shape[0], dtype=torch.float32)
        subset = contributions[:, indices]
        if method == "topk_mean":
            k = min(top_k, subset.shape[1])
            return subset.topk(k=k, dim=1).values.mean(dim=1)
        return subset.sum(dim=1)

    positive = _reduce(pos_indices)
    negative = _reduce(neg_indices)
    return {
        "positive_safety_score": positive,
        "negative_safety_score": negative,
        "safety_probe_score": positive - negative,
    }



def _runtime_compare_config(
    compare_cfg,
    baseline_model_path: str,
    baseline_adapter_path: str,
    baseline_safety_dir: str,
    compressed_model_path: str,
    compressed_adapter_path: str,
    compressed_safety_dir: str,
    sae_checkpoint_dir: str,
    output_dir: Path,
    aggregation_methods: Tuple[str, ...],
    max_samples: int,
) -> Dict[str, Any]:
    return {
        "baseline_model_path": baseline_model_path,
        "baseline_adapter_path": baseline_adapter_path,
        "baseline_safety_dir": baseline_safety_dir,
        "compressed_model_path": compressed_model_path,
        "compressed_adapter_path": compressed_adapter_path,
        "compressed_safety_dir": compressed_safety_dir,
        "sae_checkpoint_dir": sae_checkpoint_dir,
        "output_dir": str(output_dir),
        "aggregation_methods": list(aggregation_methods),
        "top_k": compare_cfg.top_k,
        "max_samples": max_samples,
        "max_length": compare_cfg.max_length,
        "token_aggregation": compare_cfg.token_aggregation,
        "device": compare_cfg.device,
        "torch_dtype": str(compare_cfg.torch_dtype),
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    compare_cfg = cfg.sae_compare

    baseline_model_path = args.baseline_model_path or compare_cfg.baseline_model_path
    baseline_adapter_path = args.baseline_adapter_path if args.baseline_adapter_path is not None else (
        compare_cfg.baseline_lora_adapter_path if compare_cfg.baseline_use_lora_adapter else ""
    )
    baseline_safety_dir = args.baseline_safety_dir or compare_cfg.baseline_safety_dir
    compressed_model_path = args.compressed_model_path or compare_cfg.compressed_model_path
    compressed_adapter_path = args.compressed_adapter_path if args.compressed_adapter_path is not None else (
        compare_cfg.compressed_lora_adapter_path if compare_cfg.compressed_use_lora_adapter else ""
    )
    compressed_safety_dir = args.compressed_safety_dir or compare_cfg.compressed_safety_dir
    sae_checkpoint_dir = args.sae_checkpoint_dir or compare_cfg.sae_checkpoint_dir
    output_dir = Path(args.output_dir or compare_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregation_methods = tuple(args.aggregation_method or compare_cfg.aggregation_methods)
    max_samples = compare_cfg.max_samples if args.max_samples is None else args.max_samples

    baseline_examples_map = _load_safety_examples(baseline_safety_dir, max_samples=max_samples)
    compressed_examples_map = _load_safety_examples(compressed_safety_dir, max_samples=max_samples)
    paired_examples = _align_examples(baseline_examples_map, compressed_examples_map)
    baseline_examples = [item[0] for item in paired_examples]
    compressed_examples = [item[1] for item in paired_examples]

    sae_bundles = _load_sae_bundles(sae_checkpoint_dir)
    module_names = tuple(sorted(sae_bundles))

    baseline_activations = _collect_model_activations(
        model_path=baseline_model_path,
        adapter_path=baseline_adapter_path,
        device=compare_cfg.device,
        dtype=compare_cfg.torch_dtype,
        max_length=compare_cfg.max_length,
        token_aggregation=compare_cfg.token_aggregation,
        module_names=module_names,
        samples=baseline_examples,
        desc="sae-compare:baseline",
    )
    compressed_activations = _collect_model_activations(
        model_path=compressed_model_path,
        adapter_path=compressed_adapter_path,
        device=compare_cfg.device,
        dtype=compare_cfg.torch_dtype,
        max_length=compare_cfg.max_length,
        token_aggregation=compare_cfg.token_aggregation,
        module_names=module_names,
        samples=compressed_examples,
        desc="sae-compare:compressed",
    )

    baseline_scores = torch.tensor([item.safety_score for item in baseline_examples], dtype=torch.float32)
    compressed_scores = torch.tensor([item.safety_score for item in compressed_examples], dtype=torch.float32)
    score_delta = compressed_scores - baseline_scores

    sample_records: List[Dict[str, Any]] = []
    module_results: Dict[str, Any] = {}
    feature_results: Dict[str, Any] = {}

    for sample_idx, (baseline_item, compressed_item) in enumerate(paired_examples):
        sample_records.append(
            {
                "sample_id": baseline_item.sample_id,
                "dataset_name": baseline_item.dataset_name,
                "prompt": baseline_item.prompt,
                "baseline_response": baseline_item.response,
                "compressed_response": compressed_item.response,
                "baseline_safety_score": baseline_item.safety_score,
                "compressed_safety_score": compressed_item.safety_score,
                "safety_score_delta": float(compressed_item.safety_score - baseline_item.safety_score),
                "modules": {},
            }
        )

    overall_probe_deltas: Dict[str, List[float]] = {method: [] for method in aggregation_methods}

    for module_name, bundle in sae_bundles.items():
        checkpoint = bundle["checkpoint"]
        sae_model = _instantiate_sae(checkpoint, compare_cfg.device)
        mean = checkpoint["mean"].float()
        std = checkpoint["std"].float().clamp_min(1e-6)
        feature_specs = _build_feature_specs(bundle)
        feature_indices = [feature.feature_index for feature in feature_specs]

        with torch.no_grad():
            baseline_latents = sae_model.encode(((baseline_activations[module_name] - mean) / std).to(compare_cfg.device)).cpu()
            compressed_latents = sae_model.encode(((compressed_activations[module_name] - mean) / std).to(compare_cfg.device)).cpu()

        baseline_selected = baseline_latents[:, feature_indices] if feature_indices else torch.zeros((len(sample_records), 0))
        compressed_selected = compressed_latents[:, feature_indices] if feature_indices else torch.zeros((len(sample_records), 0))
        combined = torch.cat([baseline_selected, compressed_selected], dim=0) if feature_indices else torch.zeros((0, 0))
        if feature_indices:
            feature_mean = combined.mean(dim=0, keepdim=True)
            feature_std = combined.std(dim=0, keepdim=True).clamp_min(1e-6)
            baseline_z = (baseline_selected - feature_mean) / feature_std
            compressed_z = (compressed_selected - feature_mean) / feature_std
        else:
            baseline_z = baseline_selected
            compressed_z = compressed_selected

        method_scores: Dict[str, Dict[str, torch.Tensor]] = {}
        for method in aggregation_methods:
            method_scores[method] = {
                "baseline": _aggregate_scores(baseline_selected, feature_specs, method, compare_cfg.top_k, z_values=baseline_z),
                "compressed": _aggregate_scores(compressed_selected, feature_specs, method, compare_cfg.top_k, z_values=compressed_z),
            }

        per_feature_summaries = []
        weights = torch.tensor([feature.weight for feature in feature_specs], dtype=torch.float32) if feature_specs else torch.zeros(0)
        baseline_contrib = baseline_selected * weights.unsqueeze(0) if feature_specs else torch.zeros_like(baseline_selected)
        compressed_contrib = compressed_selected * weights.unsqueeze(0) if feature_specs else torch.zeros_like(compressed_selected)
        contrib_delta = compressed_contrib - baseline_contrib

        for local_idx, feature in enumerate(feature_specs):
            baseline_feature = baseline_selected[:, local_idx]
            compressed_feature = compressed_selected[:, local_idx]
            per_feature_summaries.append(
                {
                    "feature_index": feature.feature_index,
                    "role": feature.role,
                    "correlation_weight": feature.correlation,
                    "baseline": {
                        "pearson": _pearson(baseline_feature, baseline_scores),
                        "spearman": _spearman(baseline_feature, baseline_scores),
                        "mean": float(baseline_feature.mean()),
                    },
                    "compressed": {
                        "pearson": _pearson(compressed_feature, compressed_scores),
                        "spearman": _spearman(compressed_feature, compressed_scores),
                        "mean": float(compressed_feature.mean()),
                    },
                    "delta": {
                        "mean": float((compressed_feature - baseline_feature).mean()),
                        "mean_abs": float((compressed_feature - baseline_feature).abs().mean()),
                        "probe_contribution_delta_mean": float(contrib_delta[:, local_idx].mean()),
                    },
                }
            )

        for sample_idx, sample in enumerate(sample_records):
            feature_entries = []
            for local_idx, feature in enumerate(feature_specs):
                base_value = float(baseline_selected[sample_idx, local_idx])
                comp_value = float(compressed_selected[sample_idx, local_idx])
                delta_value = comp_value - base_value
                contribution_delta = float(contrib_delta[sample_idx, local_idx])
                feature_entries.append(
                    {
                        "feature_index": feature.feature_index,
                        "role": feature.role,
                        "correlation_weight": feature.correlation,
                        "baseline_latent": base_value,
                        "compressed_latent": comp_value,
                        "delta": delta_value,
                        "abs_delta": abs(delta_value),
                        "relative_change_rate": float(delta_value / (abs(base_value) + 1e-8)),
                        "sign_flip": bool(base_value * comp_value < 0),
                        "baseline_contribution": float(baseline_contrib[sample_idx, local_idx]) if feature_specs else 0.0,
                        "compressed_contribution": float(compressed_contrib[sample_idx, local_idx]) if feature_specs else 0.0,
                        "probe_contribution_delta": contribution_delta,
                    }
                )

            positive_sorted = sorted(
                [item for item in feature_entries if item["role"] == "positive"],
                key=lambda item: item["abs_delta"],
                reverse=True,
            )
            negative_sorted = sorted(
                [item for item in feature_entries if item["role"] == "negative"],
                key=lambda item: item["abs_delta"],
                reverse=True,
            )
            probe_sorted = sorted(
                feature_entries,
                key=lambda item: abs(item["probe_contribution_delta"]),
                reverse=True,
            )

            aggregations = {}
            for method in aggregation_methods:
                baseline_method = method_scores[method]["baseline"]
                compressed_method = method_scores[method]["compressed"]
                positive_delta = float(compressed_method["positive_safety_score"][sample_idx] - baseline_method["positive_safety_score"][sample_idx])
                negative_delta = float(compressed_method["negative_safety_score"][sample_idx] - baseline_method["negative_safety_score"][sample_idx])
                probe_delta = float(compressed_method["safety_probe_score"][sample_idx] - baseline_method["safety_probe_score"][sample_idx])
                aggregations[method] = {
                    "baseline_positive_safety_score": float(baseline_method["positive_safety_score"][sample_idx]),
                    "compressed_positive_safety_score": float(compressed_method["positive_safety_score"][sample_idx]),
                    "positive_safety_score_delta": positive_delta,
                    "baseline_negative_safety_score": float(baseline_method["negative_safety_score"][sample_idx]),
                    "compressed_negative_safety_score": float(compressed_method["negative_safety_score"][sample_idx]),
                    "negative_safety_score_delta": negative_delta,
                    "baseline_safety_probe_score": float(baseline_method["safety_probe_score"][sample_idx]),
                    "compressed_safety_probe_score": float(compressed_method["safety_probe_score"][sample_idx]),
                    "safety_probe_score_delta": probe_delta,
                }

            sample["modules"][module_name] = {
                "aggregations": aggregations,
                "feature_entries": feature_entries,
                "largest_positive_feature_changes": positive_sorted[:5],
                "largest_negative_feature_changes": negative_sorted[:5],
                "largest_probe_contribution_changes": probe_sorted[:5],
            }

        aggregation_summary = {}
        for method in aggregation_methods:
            baseline_method = method_scores[method]["baseline"]
            compressed_method = method_scores[method]["compressed"]
            positive_delta = compressed_method["positive_safety_score"] - baseline_method["positive_safety_score"]
            negative_delta = compressed_method["negative_safety_score"] - baseline_method["negative_safety_score"]
            probe_delta = compressed_method["safety_probe_score"] - baseline_method["safety_probe_score"]
            overall_probe_deltas[method].extend(probe_delta.tolist())
            aggregation_summary[method] = {
                "positive_safety_score": {
                    "baseline_mean": float(baseline_method["positive_safety_score"].mean()),
                    "compressed_mean": float(compressed_method["positive_safety_score"].mean()),
                    "delta_mean": float(positive_delta.mean()),
                },
                "negative_safety_score": {
                    "baseline_mean": float(baseline_method["negative_safety_score"].mean()),
                    "compressed_mean": float(compressed_method["negative_safety_score"].mean()),
                    "delta_mean": float(negative_delta.mean()),
                },
                "safety_probe_score": {
                    "baseline_mean": float(baseline_method["safety_probe_score"].mean()),
                    "compressed_mean": float(compressed_method["safety_probe_score"].mean()),
                    "delta_mean": float(probe_delta.mean()),
                    "delta_abs_mean": float(probe_delta.abs().mean()),
                },
                "correlation_with_safety_score": {
                    "baseline_probe_pearson": _pearson(baseline_method["safety_probe_score"], baseline_scores),
                    "baseline_probe_spearman": _spearman(baseline_method["safety_probe_score"], baseline_scores),
                    "compressed_probe_pearson": _pearson(compressed_method["safety_probe_score"], compressed_scores),
                    "compressed_probe_spearman": _spearman(compressed_method["safety_probe_score"], compressed_scores),
                    "baseline_positive_pearson": _pearson(baseline_method["positive_safety_score"], baseline_scores),
                    "compressed_positive_pearson": _pearson(compressed_method["positive_safety_score"], compressed_scores),
                    "baseline_negative_pearson": _pearson(baseline_method["negative_safety_score"], baseline_scores),
                    "compressed_negative_pearson": _pearson(compressed_method["negative_safety_score"], compressed_scores),
                },
                "delta_correlation": {
                    "safety_score_delta_vs_probe_delta_pearson": _pearson(score_delta, probe_delta),
                    "safety_score_delta_vs_probe_delta_spearman": _spearman(score_delta, probe_delta),
                },
            }

        module_results[module_name] = {
            "feature_count": len(feature_specs),
            "aggregation_summary": aggregation_summary,
            "mean_baseline_safety_score": float(baseline_scores.mean()),
            "mean_compressed_safety_score": float(compressed_scores.mean()),
            "mean_safety_score_delta": float(score_delta.mean()),
            "positive_weakened_mean": float(aggregation_summary[aggregation_methods[0]]["positive_safety_score"]["delta_mean"]),
            "negative_amplified_mean": float(aggregation_summary[aggregation_methods[0]]["negative_safety_score"]["delta_mean"]),
            "probe_deviation_mean": float(aggregation_summary[aggregation_methods[0]]["safety_probe_score"]["delta_abs_mean"]),
        }
        feature_results[module_name] = per_feature_summaries

    default_method = aggregation_methods[0]
    ranked_modules = sorted(
        module_results.items(),
        key=lambda item: abs(item[1]["aggregation_summary"][default_method]["safety_probe_score"]["delta_mean"]),
        reverse=True,
    )
    summary = {
        "config": _runtime_compare_config(
            compare_cfg=compare_cfg,
            baseline_model_path=baseline_model_path,
            baseline_adapter_path=baseline_adapter_path,
            baseline_safety_dir=baseline_safety_dir,
            compressed_model_path=compressed_model_path,
            compressed_adapter_path=compressed_adapter_path,
            compressed_safety_dir=compressed_safety_dir,
            sae_checkpoint_dir=sae_checkpoint_dir,
            output_dir=output_dir,
            aggregation_methods=aggregation_methods,
            max_samples=max_samples,
        ),
        "baseline_model_path": baseline_model_path,
        "baseline_adapter_path": baseline_adapter_path,
        "compressed_model_path": compressed_model_path,
        "compressed_adapter_path": compressed_adapter_path,
        "baseline_safety_dir": baseline_safety_dir,
        "compressed_safety_dir": compressed_safety_dir,
        "sae_checkpoint_dir": sae_checkpoint_dir,
        "sample_count": len(sample_records),
        "aggregation_methods": aggregation_methods,
        "most_shifted_modules": [
            {
                "module_name": name,
                "probe_delta_mean": item["aggregation_summary"][default_method]["safety_probe_score"]["delta_mean"],
                "probe_delta_abs_mean": item["aggregation_summary"][default_method]["safety_probe_score"]["delta_abs_mean"],
            }
            for name, item in ranked_modules[:10]
        ],
        "overall_mean_safety_score_delta": float(score_delta.mean()),
        "module_level_interpretation": {
            name: {
                "positive_weakened": bool(item["aggregation_summary"][default_method]["positive_safety_score"]["delta_mean"] < 0),
                "negative_amplified": bool(item["aggregation_summary"][default_method]["negative_safety_score"]["delta_mean"] > 0),
                "probe_explains_safety_drop": float(item["aggregation_summary"][default_method]["delta_correlation"]["safety_score_delta_vs_probe_delta_pearson"]),
            }
            for name, item in module_results.items()
        },
    }

    with (output_dir / "sample_level.json").open("w", encoding="utf-8") as handle:
        json.dump(sample_records, handle, ensure_ascii=False, indent=2)
    with (output_dir / "feature_level.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_results, handle, ensure_ascii=False, indent=2)
    with (output_dir / "module_level.json").open("w", encoding="utf-8") as handle:
        json.dump(module_results, handle, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Saved SAE compare sample-level results to: {output_dir / 'sample_level.json'}")
    print(f"Saved SAE compare feature-level results to: {output_dir / 'feature_level.json'}")
    print(f"Saved SAE compare module-level results to: {output_dir / 'module_level.json'}")
    print(f"Saved SAE compare summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
