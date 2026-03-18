import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from config import load_config
from src.activation.plots import (
    build_adjacent_similarity,
    build_aligned_diff_matrix,
    build_similarity_diff,
    build_topk_matrix,
    ensure_dir,
    plot_heatmap,
    plot_similarity_strip,
    write_html_overview,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate activation-path visualizations from two activation reports.")
    parser.add_argument("--base", default=None, help="Base activation JSON path.")
    parser.add_argument("--target", default=None, help="Target activation JSON path.")
    parser.add_argument("--output-dir", default=None, help="Directory for PNG outputs.")
    parser.add_argument("--output", default=None, help="Output HTML path.")
    return parser.parse_args()


def _load_report(path: str) -> Dict[str, Any]:
    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Activation report not found: {path}")
    with report_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Activation report must be a JSON object: {path}")
    return data


def _require_full_vectors(report: Dict[str, Any], path: str) -> None:
    if report.get("has_full_activation_vector") is True:
        return
    categories = report.get("categories", {})
    if isinstance(categories, dict):
        for item in categories.values():
            if not isinstance(item, dict):
                continue
            layers = item.get("layers", {})
            if not isinstance(layers, dict):
                continue
            for layer in layers.values():
                if isinstance(layer, dict) and isinstance(layer.get("avg_activation_vector"), list):
                    return
    raise ValueError(
        f"Activation report {path} does not include full average activation vectors. "
        "Regenerate it with save_full_vector=True before plotting similarity and diff maps."
    )


def _categories(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    categories = report.get("categories", {})
    if not isinstance(categories, dict):
        return {}
    return {name: item for name, item in categories.items() if isinstance(item, dict)}


def _sanitize(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    act_cfg = cfg.activation

    base_path = args.base or act_cfg.compare_base_file
    target_path = args.target or act_cfg.compare_target_file
    output_dir = ensure_dir(args.output_dir or act_cfg.path_compare_dir)
    output_html = Path(args.output or act_cfg.path_compare_index_path)

    base_report = _load_report(base_path)
    target_report = _load_report(target_path)
    _require_full_vectors(base_report, base_path)
    _require_full_vectors(target_report, target_path)

    base_categories = _categories(base_report)
    target_categories = _categories(target_report)
    shared_categories = [name for name in base_categories if name in target_categories]
    if not shared_categories:
        raise ValueError("No shared prompt categories found between the two activation reports.")

    sections: List[Dict[str, object]] = []
    top_k = int(act_cfg.top_k)

    for category_name in shared_categories:
        base_layers = base_categories[category_name].get("layers", {})
        target_layers = target_categories[category_name].get("layers", {})
        if not isinstance(base_layers, dict) or not isinstance(target_layers, dict):
            continue

        slug = _sanitize(category_name)
        items: List[Dict[str, str]] = []

        base_layer_names, base_values, base_indices = build_topk_matrix(base_layers, top_k)
        base_heatmap = output_dir / f"{slug}_base_topk_heatmap.png"
        plot_heatmap(
            output_path=base_heatmap,
            values=base_values,
            row_labels=base_layer_names,
            col_labels=[f"top-{idx + 1}" for idx in range(top_k)],
            title=f"{category_name}: base top-k mean activations",
            subtitle="Cell text is the neuron index for that top-k slot.",
            annotation=base_indices,
            cmap="coolwarm",
            center=0.0,
        )
        items.append(
            {
                "title": "Base Top-k Heatmap",
                "caption": "Rows are layers in forward order. Columns are top-k ranks. Colors show signed mean activation values.",
                "src": base_heatmap.name,
            }
        )

        target_layer_names, target_values, target_indices = build_topk_matrix(target_layers, top_k)
        target_heatmap = output_dir / f"{slug}_target_topk_heatmap.png"
        plot_heatmap(
            output_path=target_heatmap,
            values=target_values,
            row_labels=target_layer_names,
            col_labels=[f"top-{idx + 1}" for idx in range(top_k)],
            title=f"{category_name}: target top-k mean activations",
            subtitle="Cell text is the neuron index for that top-k slot.",
            annotation=target_indices,
            cmap="coolwarm",
            center=0.0,
        )
        items.append(
            {
                "title": "Target Top-k Heatmap",
                "caption": "Target model's own top-k mean activations. This exposes shifts in the dominant neurons after adaptation or pruning.",
                "src": target_heatmap.name,
            }
        )

        shared_layer_names, aligned_indices, aligned_base, aligned_target, aligned_diff = build_aligned_diff_matrix(
            base_layers,
            target_layers,
            top_k,
        )
        diff_heatmap = output_dir / f"{slug}_aligned_diff_heatmap.png"
        plot_heatmap(
            output_path=diff_heatmap,
            values=aligned_diff,
            row_labels=shared_layer_names,
            col_labels=[f"base-top-{idx + 1}" for idx in range(top_k)],
            title=f"{category_name}: aligned activation difference",
            subtitle="Target minus base, aligned on base model top-k neuron ids. Cell text is the base neuron index.",
            annotation=aligned_indices,
            cmap="PiYG",
            center=0.0,
        )
        items.append(
            {
                "title": "Aligned Difference Heatmap",
                "caption": "For each layer, target values are read at the same neuron ids selected by the base model top-k ranking.",
                "src": diff_heatmap.name,
            }
        )

        base_transition_labels, base_similarity_values = build_adjacent_similarity(base_layers, top_k)
        base_similarity = output_dir / f"{slug}_base_similarity.png"
        plot_similarity_strip(
            output_path=base_similarity,
            values=base_similarity_values,
            labels=base_transition_labels,
            title=f"{category_name}: base adjacent-layer similarity",
            subtitle="Cosine similarity between consecutive layers' top-k absolute-activation profiles.",
            cmap="viridis",
            center=None,
        )
        items.append(
            {
                "title": "Base Adjacent-layer Similarity",
                "caption": "This strip compares adjacent layers using fixed-length top-k activation profiles, so it remains valid even when module widths differ.",
                "src": base_similarity.name,
            }
        )

        target_transition_labels, target_similarity_values = build_adjacent_similarity(target_layers, top_k)
        target_similarity = output_dir / f"{slug}_target_similarity.png"
        plot_similarity_strip(
            output_path=target_similarity,
            values=target_similarity_values,
            labels=target_transition_labels,
            title=f"{category_name}: target adjacent-layer similarity",
            subtitle="Cosine similarity between consecutive layers' top-k absolute-activation profiles.",
            cmap="viridis",
            center=None,
        )
        items.append(
            {
                "title": "Target Adjacent-layer Similarity",
                "caption": "Use this together with the base strip to see whether the forward activation path becomes smoother, noisier, or rerouted under the same fixed-length profile metric.",
                "src": target_similarity.name,
            }
        )

        shared_transition_labels, shared_base_similarity, shared_target_similarity, similarity_diff = build_similarity_diff(
            base_layers,
            target_layers,
            top_k,
        )
        similarity_diff_path = output_dir / f"{slug}_similarity_diff.png"
        plot_similarity_strip(
            output_path=similarity_diff_path,
            values=similarity_diff,
            labels=shared_transition_labels,
            title=f"{category_name}: adjacent-layer similarity difference",
            subtitle="Target minus base cosine similarity on the same transition's top-k activation profiles.",
            cmap="coolwarm",
            center=0.0,
        )
        items.append(
            {
                "title": "Similarity Difference",
                "caption": "Positive cells indicate the target model preserves or sharpens the transition more strongly than the base model.",
                "src": similarity_diff_path.name,
            }
        )

        sections.append({"title": category_name, "items": items})
        print(f"[activation.index] rendered category={category_name!r} to {output_dir}")

    write_html_overview(output_html, "Activation Path Compare", sections)
    print(f"Saved activation path overview to: {output_html}")


if __name__ == "__main__":
    main()
