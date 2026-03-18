from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def layer_sort_key(layer_name: str) -> tuple[int, int, str]:
    import re

    match = re.search(r"model\.layers\.(\d+)\.(.+?)(?:\.base_layer)?$", layer_name)
    if not match:
        return (10**9, 10**9, layer_name)
    layer_id = int(match.group(1))
    suffix = match.group(2)
    order = {
        "self_attn.q_proj": 0,
        "self_attn.k_proj": 1,
        "self_attn.v_proj": 2,
        "self_attn.o_proj": 3,
        "mlp.gate_proj": 4,
        "mlp.up_proj": 5,
        "mlp.down_proj": 6,
    }
    return (layer_id, order.get(suffix, 99), suffix)


def short_layer_name(layer_name: str) -> str:
    import re

    match = re.search(r"model\.layers\.(\d+)\.(.+?)(?:\.base_layer)?$", layer_name)
    if not match:
        return layer_name
    return f"L{match.group(1)} {match.group(2)}"


def cosine_similarity(left: List[float], right: List[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"shape mismatch for cosine similarity: {left_arr.shape} vs {right_arr.shape}")
    denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left_arr, right_arr) / denom)


def topk_profile(layer: Dict[str, Dict[str, object]] | Dict[str, object], top_k: int) -> List[float]:
    values = layer.get("topk_activation_values", []) or []
    profile = [abs(float(value)) for value in values[:top_k]]
    if len(profile) < top_k:
        profile.extend([0.0] * (top_k - len(profile)))
    return profile


def build_topk_matrix(category_layers: Dict[str, Dict[str, object]], top_k: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
    layer_names = sorted(category_layers, key=layer_sort_key)
    values = np.full((len(layer_names), top_k), np.nan, dtype=np.float32)
    indices = np.full((len(layer_names), top_k), -1, dtype=np.int32)
    for row_idx, layer_name in enumerate(layer_names):
        layer = category_layers[layer_name]
        layer_values = layer.get("topk_activation_values", []) or []
        layer_indices = layer.get("topk_neuron_indices", []) or []
        limit = min(top_k, len(layer_values), len(layer_indices))
        for col_idx in range(limit):
            values[row_idx, col_idx] = float(layer_values[col_idx])
            indices[row_idx, col_idx] = int(layer_indices[col_idx])
    return layer_names, values, indices


def build_aligned_diff_matrix(
    base_layers: Dict[str, Dict[str, object]],
    target_layers: Dict[str, Dict[str, object]],
    top_k: int,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shared_layer_names = sorted(set(base_layers) & set(target_layers), key=layer_sort_key)
    base_values = np.full((len(shared_layer_names), top_k), np.nan, dtype=np.float32)
    target_values = np.full((len(shared_layer_names), top_k), np.nan, dtype=np.float32)
    diff_values = np.full((len(shared_layer_names), top_k), np.nan, dtype=np.float32)
    neuron_indices = np.full((len(shared_layer_names), top_k), -1, dtype=np.int32)

    for row_idx, layer_name in enumerate(shared_layer_names):
        base_layer = base_layers[layer_name]
        target_layer = target_layers[layer_name]
        base_top_values = base_layer.get("topk_activation_values", []) or []
        base_top_indices = base_layer.get("topk_neuron_indices", []) or []
        target_vector = target_layer.get("avg_activation_vector", []) or []

        limit = min(top_k, len(base_top_values), len(base_top_indices))
        for col_idx in range(limit):
            neuron_index = int(base_top_indices[col_idx])
            neuron_indices[row_idx, col_idx] = neuron_index
            base_value = float(base_top_values[col_idx])
            base_values[row_idx, col_idx] = base_value
            if 0 <= neuron_index < len(target_vector):
                target_value = float(target_vector[neuron_index])
                target_values[row_idx, col_idx] = target_value
                diff_values[row_idx, col_idx] = target_value - base_value
    return shared_layer_names, neuron_indices, base_values, target_values, diff_values


def build_adjacent_similarity(category_layers: Dict[str, Dict[str, object]], top_k: int) -> Tuple[List[str], np.ndarray]:
    layer_names = sorted(category_layers, key=layer_sort_key)
    transitions: List[str] = []
    values: List[float] = []
    for idx in range(len(layer_names) - 1):
        left_name = layer_names[idx]
        right_name = layer_names[idx + 1]
        left_profile = topk_profile(category_layers[left_name], top_k)
        right_profile = topk_profile(category_layers[right_name], top_k)
        transitions.append(f"{short_layer_name(left_name)} -> {short_layer_name(right_name)}")
        values.append(cosine_similarity(left_profile, right_profile))
    return transitions, np.asarray(values, dtype=np.float32)


def build_similarity_diff(
    base_layers: Dict[str, Dict[str, object]],
    target_layers: Dict[str, Dict[str, object]],
    top_k: int,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    base_labels, base_values = build_adjacent_similarity(base_layers, top_k)
    target_labels, target_values = build_adjacent_similarity(target_layers, top_k)
    shared = sorted(set(base_labels) & set(target_labels), key=lambda item: (base_labels + target_labels).index(item))
    base_lookup = {label: value for label, value in zip(base_labels, base_values.tolist())}
    target_lookup = {label: value for label, value in zip(target_labels, target_values.tolist())}
    base_arr = np.asarray([base_lookup[label] for label in shared], dtype=np.float32)
    target_arr = np.asarray([target_lookup[label] for label in shared], dtype=np.float32)
    return shared, base_arr, target_arr, target_arr - base_arr


def _figure_size(rows: int, cols: int, row_scale: float = 0.42, col_scale: float = 0.95) -> Tuple[float, float]:
    width = max(10.0, min(28.0, 4.5 + cols * col_scale))
    height = max(6.0, min(32.0, 2.5 + rows * row_scale))
    return width, height


def plot_heatmap(
    output_path: str | Path,
    values: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    subtitle: str,
    annotation: np.ndarray | None = None,
    cmap: str = "coolwarm",
    center: float | None = 0.0,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = len(row_labels)
    cols = len(col_labels)
    fig, ax = plt.subplots(figsize=_figure_size(rows, cols))

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        vmin, vmax = -1.0, 1.0
    elif center is None:
        vmin, vmax = float(np.min(finite_values)), float(np.max(finite_values))
    else:
        bound = float(np.max(np.abs(finite_values)))
        bound = max(bound, 1e-6)
        vmin, vmax = -bound, bound

    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels([short_layer_name(label) for label in row_labels])
    ax.set_xlabel("Top-k rank")
    ax.set_ylabel("Layer")
    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=13, fontweight="bold")

    if annotation is not None:
        for row_idx in range(rows):
            for col_idx in range(cols):
                if not np.isfinite(values[row_idx, col_idx]):
                    continue
                cell_value = values[row_idx, col_idx]
                text = str(annotation[row_idx, col_idx])
                ax.text(
                    col_idx,
                    row_idx,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.5 if rows > 20 else 7.5,
                    color="black" if abs(cell_value) < max(abs(vmin), abs(vmax)) * 0.65 else "white",
                )

    cbar = fig.colorbar(image, ax=ax, shrink=0.94)
    cbar.ax.set_ylabel("Activation" if center == 0.0 else "Value", rotation=270, labelpad=16)

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_strip(
    output_path: str | Path,
    values: np.ndarray,
    labels: List[str],
    title: str,
    subtitle: str,
    cmap: str = "viridis",
    center: float | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig_width = max(12.0, min(30.0, 5.0 + len(labels) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_width, 3.6))

    if values.size == 0:
        data = np.zeros((1, 1), dtype=np.float32)
        labels = ["No transitions"]
        vmin, vmax = 0.0, 1.0
    else:
        data = values.reshape(1, -1)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            vmin, vmax = -1.0, 1.0
        elif center is None:
            vmin, vmax = float(np.min(finite_values)), float(np.max(finite_values))
        else:
            bound = float(np.max(np.abs(finite_values)))
            bound = max(bound, 1e-6)
            vmin, vmax = -bound, bound

    image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks([0])
    ax.set_yticklabels(["adjacent"])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=13, fontweight="bold")

    for col_idx, value in enumerate(data[0]):
        ax.text(
            col_idx,
            0,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if abs(value) > max(abs(vmin), abs(vmax)) * 0.55 else "black",
        )

    cbar = fig.colorbar(image, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Cosine similarity", rotation=270, labelpad=16)

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_html_overview(output_path: str | Path, title: str, sections: List[Dict[str, object]]) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"  <title>{title}</title>",
        "  <style>",
        "    body { margin: 0; padding: 24px; background: #f6f2ea; color: #1f1a16; font-family: Georgia, serif; }",
        "    main { max-width: 1400px; margin: 0 auto; }",
        "    h1 { margin-bottom: 8px; }",
        "    .lede { color: #6d6258; max-width: 72ch; margin-bottom: 22px; }",
        "    .section { background: #fffdf9; border: 1px solid #dacfc0; border-radius: 16px; padding: 18px; margin-bottom: 22px; }",
        "    .grid { display: grid; grid-template-columns: 1fr; gap: 18px; }",
        "    .card { border: 1px solid #e3d9ca; border-radius: 12px; padding: 14px; background: #fff; }",
        "    img { width: 100%; height: auto; display: block; border-radius: 10px; }",
        "    .caption { color: #6d6258; margin: 6px 0 12px; font-size: 14px; }",
        "    code { background: #f0e8da; padding: 1px 5px; border-radius: 4px; }",
        "  </style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{title}</h1>",
        "<p class='lede'>Each category includes base and target top-k activation heatmaps, an aligned difference heatmap using base neuron ids, adjacent-layer cosine similarity strips, and the corresponding similarity difference strip.</p>",
    ]

    for section in sections:
        parts.append(f"<section class='section'><h2>{section['title']}</h2>")
        parts.append("<div class='grid'>")
        for item in section["items"]:
            parts.append("<div class='card'>")
            parts.append(f"<h3>{item['title']}</h3>")
            parts.append(f"<p class='caption'>{item['caption']}</p>")
            parts.append(f"<img src='{item['src']}' alt='{item['title']}'>")
            parts.append("</div>")
        parts.append("</div></section>")

    parts.extend(["</main>", "</body>", "</html>"])
    output.write_text("\n".join(parts), encoding="utf-8")
