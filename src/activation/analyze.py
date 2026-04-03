import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_PATCH_ROOT = Path("data/activation/sae_patch_mixed_prompt_last_s200_u200")
DEFAULT_OUTPUT_NAME = "analysis_summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SAE patch reports and summarize the most significant safety shifts.")
    parser.add_argument(
        "--patch-root",
        default=str(DEFAULT_PATCH_ROOT),
        help="Root directory containing per-model SAE patch outputs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path. Defaults to <patch-root>/analysis_summary.json.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _model_key(model_dir: Path) -> str:
    return model_dir.name


def _collect_model_rows(model_dir: Path) -> List[Dict[str, Any]]:
    summary_path = model_dir / "summary.json"
    if summary_path.is_file():
        payload = _load_json(summary_path)
        summaries = payload.get("summaries", []) if isinstance(payload, dict) else []
        if isinstance(summaries, list):
            return [row for row in summaries if isinstance(row, dict)]

    rows: List[Dict[str, Any]] = []
    for path in sorted(model_dir.glob("*/*/*.json")):
        if path.name == "summary.json":
            continue
        payload = _load_json(path)
        if isinstance(payload, dict):
            rows.append(
                {
                    "dataset_name": payload.get("dataset_name", ""),
                    "layer_name": payload.get("layer_name", path.parent.parent.name),
                    "patch_mode": payload.get("patch_mode", path.parent.name),
                    "sample_count": payload.get("sample_count", 0),
                    "baseline_mean_safe_safe_score": payload.get("baseline_mean_safe_safe_score", 0.0),
                    "patched_mean_safe_safe_score": payload.get("patched_mean_safe_safe_score", 0.0),
                    "mean_safe_safe_delta": payload.get("mean_safe_safe_delta", 0.0),
                    "safe_count": payload.get("safe_count", 0),
                    "unsafe_count": payload.get("unsafe_count", 0),
                    "output_path": str(path),
                }
            )
    return rows


def _summarize_layer(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("layer_name", "")), str(row.get("patch_mode", "")))].append(row)

    summary: Dict[str, Dict[str, Any]] = {}
    for (layer_name, patch_mode), items in grouped.items():
        deltas = [_safe_float(item.get("mean_safe_safe_delta")) for item in items]
        baseline_scores = [_safe_float(item.get("baseline_mean_safe_safe_score")) for item in items]
        patched_scores = [_safe_float(item.get("patched_mean_safe_safe_score")) for item in items]
        summary_key = f"{layer_name}::{patch_mode}"
        summary[summary_key] = {
            "layer_name": layer_name,
            "patch_mode": patch_mode,
            "report_count": len(items),
            "datasets": sorted({str(item.get("dataset_name", "")) for item in items}),
            "mean_safe_safe_delta": sum(deltas) / max(len(deltas), 1),
            "mean_abs_safe_safe_delta": sum(abs(delta) for delta in deltas) / max(len(deltas), 1),
            "max_abs_safe_safe_delta": max((abs(delta) for delta in deltas), default=0.0),
            "baseline_mean_safe_safe_score": sum(baseline_scores) / max(len(baseline_scores), 1),
            "patched_mean_safe_safe_score": sum(patched_scores) / max(len(patched_scores), 1),
            "sample_count_total": sum(_safe_int(item.get("sample_count")) for item in items),
            "safe_count": max((_safe_int(item.get("safe_count")) for item in items), default=0),
            "unsafe_count": max((_safe_int(item.get("unsafe_count")) for item in items), default=0),
            "reports": items,
        }
    return summary


def _most_significant_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    best = max(rows, key=lambda item: abs(_safe_float(item.get("mean_safe_safe_delta"))))
    return {
        "dataset_name": best.get("dataset_name", ""),
        "layer_name": best.get("layer_name", ""),
        "patch_mode": best.get("patch_mode", ""),
        "mean_safe_safe_delta": _safe_float(best.get("mean_safe_safe_delta")),
        "baseline_mean_safe_safe_score": _safe_float(best.get("baseline_mean_safe_safe_score")),
        "patched_mean_safe_safe_score": _safe_float(best.get("patched_mean_safe_safe_score")),
        "sample_count": _safe_int(best.get("sample_count")),
        "output_path": str(best.get("output_path", "")),
    }


def _most_significant_layer(layer_summary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not layer_summary:
        return {}
    best = max(layer_summary.values(), key=lambda item: abs(_safe_float(item.get("mean_safe_safe_delta"))))
    return {
        "layer_name": best.get("layer_name", ""),
        "patch_mode": best.get("patch_mode", ""),
        "mean_safe_safe_delta": _safe_float(best.get("mean_safe_safe_delta")),
        "mean_abs_safe_safe_delta": _safe_float(best.get("mean_abs_safe_safe_delta")),
        "max_abs_safe_safe_delta": _safe_float(best.get("max_abs_safe_safe_delta")),
        "report_count": _safe_int(best.get("report_count")),
    }


def _sorted_layer_ranking(layer_summary: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (
            {
                "layer_name": item.get("layer_name", ""),
                "patch_mode": item.get("patch_mode", ""),
                "mean_safe_safe_delta": _safe_float(item.get("mean_safe_safe_delta")),
                "mean_abs_safe_safe_delta": _safe_float(item.get("mean_abs_safe_safe_delta")),
                "max_abs_safe_safe_delta": _safe_float(item.get("max_abs_safe_safe_delta")),
                "report_count": _safe_int(item.get("report_count")),
                "datasets": item.get("datasets", []),
            }
            for item in layer_summary.values()
        ),
        key=lambda item: item["mean_abs_safe_safe_delta"],
        reverse=True,
    )


def main() -> None:
    args = _parse_args()
    patch_root = Path(args.patch_root).expanduser().resolve()
    if not patch_root.is_dir():
        raise FileNotFoundError(f"Patch root not found: {patch_root}")

    output_path = Path(args.output).expanduser().resolve() if args.output else patch_root / DEFAULT_OUTPUT_NAME

    per_model: Dict[str, Dict[str, Any]] = {}
    for model_dir in sorted(path for path in patch_root.iterdir() if path.is_dir()):
        rows = _collect_model_rows(model_dir)
        if not rows:
            continue
        layer_summary = _summarize_layer(rows)
        per_model[_model_key(model_dir)] = {
            "model_dir": str(model_dir),
            "report_count": len(rows),
            "datasets": sorted({str(row.get("dataset_name", "")) for row in rows}),
            "patch_modes": sorted({str(row.get("patch_mode", "")) for row in rows}),
            "layers": sorted({str(row.get("layer_name", "")) for row in rows}),
            "most_significant_report": _most_significant_report(rows),
            "most_significant_layer": _most_significant_layer(layer_summary),
            "layer_ranking": _sorted_layer_ranking(layer_summary),
        }

    payload = {
        "patch_root": str(patch_root),
        "model_count": len(per_model),
        "models": per_model,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved patch analysis to: {output_path}")


if __name__ == "__main__":
    main()
