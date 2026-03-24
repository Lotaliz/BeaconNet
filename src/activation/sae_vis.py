import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import load_config


def sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SAE probe shift and safety score relations across compare runs.")
    parser.add_argument("--compare-dir", action="append", default=None, help="Repeatable compare result directory.")
    parser.add_argument("--output-dir", default=None, help="Directory to save plots and summaries.")
    parser.add_argument("--method", default=None, help="Aggregation method to visualize.")
    parser.add_argument("--top-modules", type=int, default=None, help="How many modules to emphasize.")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _method_names(summary: Dict[str, Any]) -> List[str]:
    methods = summary.get("aggregation_methods", [])
    return [str(item) for item in methods] if isinstance(methods, list) else []


def _pick_method(summary: Dict[str, Any], requested: str | None, fallback: str) -> str:
    methods = _method_names(summary)
    if requested:
        if requested not in methods:
            raise ValueError(f"Aggregation method {requested!r} not found in summary. Available: {methods}")
        return requested
    if fallback in methods:
        return fallback
    if methods:
        return methods[0]
    raise ValueError("No aggregation methods found in summary.json")


def _run_label(compare_dir: Path, summary: Dict[str, Any]) -> str:
    config = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
    compressed_path = str(summary.get("compressed_model_path") or config.get("compressed_model_path") or compare_dir.name)
    return Path(compressed_path).name or compare_dir.name


def _discover_compare_dirs(cfg) -> List[Path]:
    root = Path(cfg.batch.sae_compare_output_root)
    pattern = f"{cfg.batch.sae_compare_output_prefix}*"
    discovered = [
        path
        for path in sorted(root.glob(pattern))
        if path.is_dir() and (path / "summary.json").is_file() and (path / "module_level.json").is_file()
    ]
    if discovered:
        return discovered
    return [
        Path(item)
        for item in cfg.sae_compare.viz_compare_dirs
        if (Path(item) / "summary.json").is_file() and (Path(item) / "module_level.json").is_file()
    ]


def _collect_run_metrics(compare_dir: Path, method: str) -> Dict[str, Any]:
    summary = _load_json(compare_dir / "summary.json")
    module_level = _load_json(compare_dir / "module_level.json")
    sample_level_path = compare_dir / "sample_level.json"
    if not isinstance(module_level, dict):
        raise TypeError(f"Expected dict in {compare_dir / 'module_level.json'}")

    compressed_safety_scores = []
    baseline_safety_scores = []
    if sample_level_path.is_file():
        sample_level = _load_json(sample_level_path)
        if isinstance(sample_level, list):
            compressed_safety_scores = [float(item.get("compressed_safety_score", 0.0)) for item in sample_level if isinstance(item, dict)]
            baseline_safety_scores = [float(item.get("baseline_safety_score", 0.0)) for item in sample_level if isinstance(item, dict)]

    module_rows = []
    for module_name, module_payload in module_level.items():
        aggregation_summary = module_payload.get("aggregation_summary", {})
        if method not in aggregation_summary:
            continue
        method_payload = aggregation_summary[method]
        probe_section = method_payload.get("safety_probe_score", {})
        delta_corr = method_payload.get("delta_correlation", {})
        module_rows.append(
            {
                "module_name": module_name,
                "probe_delta_mean": float(probe_section.get("delta_mean", 0.0)),
                "probe_delta_abs_mean": float(probe_section.get("delta_abs_mean", 0.0)),
                "positive_delta_mean": float(method_payload.get("positive_safety_score", {}).get("delta_mean", 0.0)),
                "negative_delta_mean": float(method_payload.get("negative_safety_score", {}).get("delta_mean", 0.0)),
                "score_probe_delta_pearson": float(delta_corr.get("safety_score_delta_vs_probe_delta_pearson", 0.0)),
                "score_probe_delta_spearman": float(delta_corr.get("safety_score_delta_vs_probe_delta_spearman", 0.0)),
            }
        )

    mean_probe_shift_abs = float(np.mean([row["probe_delta_abs_mean"] for row in module_rows])) if module_rows else 0.0
    mean_probe_shift_signed = float(np.mean([row["probe_delta_mean"] for row in module_rows])) if module_rows else 0.0
    mean_corr = float(np.mean([row["score_probe_delta_pearson"] for row in module_rows])) if module_rows else 0.0
    return {
        "compare_dir": str(compare_dir),
        "label": _run_label(compare_dir, summary),
        "summary": summary,
        "module_rows": module_rows,
        "mean_probe_shift_abs": mean_probe_shift_abs,
        "mean_probe_shift_signed": mean_probe_shift_signed,
        "mean_baseline_safety_score": float(np.mean(baseline_safety_scores)) if baseline_safety_scores else 0.0,
        "mean_compressed_safety_score": float(np.mean(compressed_safety_scores)) if compressed_safety_scores else 0.0,
        "overall_mean_safety_score_delta": float(summary.get("overall_mean_safety_score_delta", 0.0)),
        "mean_probe_score_delta_corr": mean_corr,
    }


def _scatter_runs(
    run_metrics: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    if not run_metrics:
        return
    x = np.array([row[x_key] for row in run_metrics], dtype=float)
    y = np.array([row[y_key] for row in run_metrics], dtype=float)
    labels = [row["label"] for row in run_metrics]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=80, color="#1768ac")
    for idx, label in enumerate(labels):
        ax.annotate(label, (x[idx], y[idx]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    if len(run_metrics) >= 2 and np.std(x) > 1e-8 and np.std(y) > 1e-8:
        coef = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = coef[0] * x_line + coef[1]
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.plot(x_line, y_line, color="#b22222", linewidth=1.5, label=f"fit, r={corr:.3f}")
        ax.legend(frameon=False)
    ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _heatmap_modules(run_metrics: List[Dict[str, Any]], top_modules: int, output_path: Path) -> List[str]:
    if not run_metrics:
        return []
    aggregate: Dict[str, float] = {}
    for run in run_metrics:
        for row in run["module_rows"]:
            aggregate[row["module_name"]] = aggregate.get(row["module_name"], 0.0) + abs(row["probe_delta_abs_mean"])
    selected = [name for name, _ in sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:top_modules]]
    if not selected:
        return []

    matrix = np.zeros((len(selected), len(run_metrics)), dtype=float)
    for col, run in enumerate(run_metrics):
        row_map = {row["module_name"]: row for row in run["module_rows"]}
        for row_idx, module_name in enumerate(selected):
            if module_name in row_map:
                matrix[row_idx, col] = row_map[module_name]["probe_delta_mean"]

    fig_w = max(8, len(run_metrics) * 1.4)
    fig_h = max(4, len(selected) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(run_metrics)))
    ax.set_xticklabels([row["label"] for row in run_metrics], rotation=35, ha="right")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected)
    ax.set_title("Module Probe Shift Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Probe delta mean")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return selected


def _sample_scatter(compare_dir: Path, method: str, top_modules: int, output_dir: Path) -> List[Dict[str, Any]]:
    sample_level = _load_json(compare_dir / "sample_level.json")
    module_level = _load_json(compare_dir / "module_level.json")
    if not isinstance(sample_level, list):
        raise TypeError(f"Expected list in {compare_dir / 'sample_level.json'}")
    if not isinstance(module_level, dict):
        raise TypeError(f"Expected dict in {compare_dir / 'module_level.json'}")

    ranked_modules = []
    for module_name, payload in module_level.items():
        aggregation_summary = payload.get("aggregation_summary", {})
        if method not in aggregation_summary:
            continue
        ranked_modules.append(
            (
                module_name,
                abs(float(aggregation_summary[method].get("safety_probe_score", {}).get("delta_abs_mean", 0.0))),
            )
        )
    ranked_modules.sort(key=lambda item: item[1], reverse=True)
    selected_modules = [name for name, _ in ranked_modules[:top_modules]]

    outputs = []
    for module_name in selected_modules:
        x_values = []
        y_values = []
        for row in sample_level:
            modules = row.get("modules", {})
            if module_name not in modules:
                continue
            aggregations = modules[module_name].get("aggregations", {})
            if method not in aggregations:
                continue
            x_values.append(float(aggregations[method].get("safety_probe_score_delta", 0.0)))
            y_values.append(float(row.get("safety_score_delta", 0.0)))
        if len(x_values) < 2:
            continue

        x = np.array(x_values, dtype=float)
        y = np.array(y_values, dtype=float)
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-8 and np.std(y) > 1e-8 else 0.0
        filename = f"sample_scatter_{sanitize_filename(module_name)}.png"
        output_path = output_dir / filename

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, s=18, alpha=0.7, color="#1768ac")
        if np.std(x) > 1e-8 and np.std(y) > 1e-8:
            coef = np.polyfit(x, y, deg=1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, color="#b22222", linewidth=1.5)
        ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")
        ax.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Safety probe score delta")
        ax.set_ylabel("Safety score delta")
        ax.set_title(f"{module_name}\nr={corr:.3f}")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

        outputs.append(
            {
                "module_name": module_name,
                "correlation": corr,
                "output_path": str(output_path),
            }
        )
    return outputs


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    compare_cfg = cfg.sae_compare

    compare_dirs = [Path(item) for item in args.compare_dir] if args.compare_dir else _discover_compare_dirs(cfg)
    output_dir = Path(args.output_dir or compare_cfg.viz_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not compare_dirs:
        raise ValueError("No compare directories provided.")

    run_metrics: List[Dict[str, Any]] = []
    method = args.method or compare_cfg.viz_default_method
    for compare_dir in compare_dirs:
        if not compare_dir.is_dir():
            raise FileNotFoundError(f"Compare directory not found: {compare_dir}")
        summary = _load_json(compare_dir / "summary.json")
        chosen_method = _pick_method(summary, method, compare_cfg.viz_default_method)
        run_metrics.append(_collect_run_metrics(compare_dir, chosen_method))
        method = chosen_method

    top_modules = args.top_modules or compare_cfg.viz_top_modules
    scatter_delta_path = output_dir / "probe_shift_vs_safety_delta.png"
    scatter_level_path = output_dir / "probe_shift_vs_safety_level.png"
    heatmap_path = output_dir / "module_probe_shift_heatmap.png"

    _scatter_runs(
        run_metrics,
        x_key="mean_probe_shift_abs",
        y_key="overall_mean_safety_score_delta",
        title="Probe Shift vs Safety Score Delta",
        x_label="Mean absolute probe shift",
        y_label="Mean safety score delta",
        output_path=scatter_delta_path,
    )
    _scatter_runs(
        run_metrics,
        x_key="mean_probe_shift_abs",
        y_key="mean_compressed_safety_score",
        title="Probe Shift vs Compressed Safety Level",
        x_label="Mean absolute probe shift",
        y_label="Mean compressed safety score",
        output_path=scatter_level_path,
    )
    selected_modules = _heatmap_modules(run_metrics, top_modules=top_modules, output_path=heatmap_path)

    sample_scatter_outputs: List[Dict[str, Any]] = []
    if len(compare_dirs) == 1:
        sample_scatter_outputs = _sample_scatter(compare_dirs[0], method=method, top_modules=top_modules, output_dir=output_dir)

    payload = {
        "method": method,
        "compare_dirs": [str(path) for path in compare_dirs],
        "selected_modules": selected_modules,
        "run_metrics": [
            {
                "label": row["label"],
                "compare_dir": row["compare_dir"],
                "mean_probe_shift_abs": row["mean_probe_shift_abs"],
                "mean_probe_shift_signed": row["mean_probe_shift_signed"],
                "mean_baseline_safety_score": row["mean_baseline_safety_score"],
                "mean_compressed_safety_score": row["mean_compressed_safety_score"],
                "overall_mean_safety_score_delta": row["overall_mean_safety_score_delta"],
                "mean_probe_score_delta_corr": row["mean_probe_score_delta_corr"],
            }
            for row in run_metrics
        ],
        "outputs": {
            "probe_shift_vs_safety_delta": str(scatter_delta_path),
            "probe_shift_vs_safety_level": str(scatter_level_path),
            "module_probe_shift_heatmap": str(heatmap_path),
            "sample_scatter_plots": sample_scatter_outputs,
        },
    }
    with (output_dir / "viz_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved run-level delta scatter to: {scatter_delta_path}")
    print(f"Saved run-level safety-level scatter to: {scatter_level_path}")
    print(f"Saved module heatmap to: {heatmap_path}")
    if sample_scatter_outputs:
        print(f"Saved sample-level scatter plots to: {output_dir}")
    print(f"Saved visualization summary to: {output_dir / 'viz_summary.json'}")


if __name__ == "__main__":
    main()
