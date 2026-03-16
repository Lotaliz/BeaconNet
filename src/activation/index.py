import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize differences between two activation reports.")
    parser.add_argument("--base", default=None, help="Base activation JSON path.")
    parser.add_argument("--target", default=None, help="Target activation JSON path.")
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


def _report_has_full_vectors(report: Dict[str, Any]) -> bool:
    if report.get("has_full_activation_vector") is True:
        return True
    for item in report.get("results", []):
        if not isinstance(item, dict):
            continue
        layers = item.get("layers", {})
        if not isinstance(layers, dict):
            continue
        for layer in layers.values():
            if isinstance(layer, dict) and isinstance(layer.get("activation_vector"), list):
                return True
    return False


def _prompt_map(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    results = report.get("results", [])
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt", "")
        if isinstance(prompt, str) and prompt.strip():
            mapping[prompt] = item
    return mapping


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _cell_style(value: float, scale: float) -> str:
    if scale <= 0:
        return ""
    alpha = min(abs(value) / scale, 1.0)
    if value > 0:
        return f"background: rgba(198, 74, 28, {0.18 + 0.52 * alpha:.3f});"
    if value < 0:
        return f"background: rgba(26, 93, 171, {0.18 + 0.52 * alpha:.3f});"
    return ""


def _layer_sort_key(layer_name: str) -> tuple[int, int, str]:
    match = re.search(r"model\.layers\.(\d+)\.(.+)$", layer_name)
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


def _lookup_value(layer: Dict[str, Any], neuron_index: int) -> float | None:
    vector = layer.get("activation_vector")
    if isinstance(vector, list) and 0 <= neuron_index < len(vector):
        return float(vector[neuron_index])

    indices = layer.get("topk_neuron_indices", [])
    values = layer.get("topk_activation_values", [])
    if isinstance(indices, list) and isinstance(values, list):
        for idx, value in zip(indices, values):
            if int(idx) == neuron_index:
                return float(value)
    return None


def _build_layer_cards(base_layers: Dict[str, Any], target_layers: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for layer_name in sorted(set(base_layers) & set(target_layers), key=_layer_sort_key):
        base = base_layers[layer_name]
        target = target_layers[layer_name]
        base_topk = [int(x) for x in base.get("topk_neuron_indices", [])]
        target_topk = [int(x) for x in target.get("topk_neuron_indices", [])]
        ordered_union: List[int] = []
        for neuron_index in base_topk + target_topk:
            if neuron_index not in ordered_union:
                ordered_union.append(neuron_index)

        neuron_rows = []
        for neuron_index in ordered_union:
            neuron_rows.append(
                {
                    "neuron_index": neuron_index,
                    "base_value": _lookup_value(base, neuron_index),
                    "target_value": _lookup_value(target, neuron_index),
                    "in_base_topk": neuron_index in base_topk,
                    "in_target_topk": neuron_index in target_topk,
                }
            )

        cards.append(
            {
                "layer_name": layer_name,
                "base_peak_neuron": int(base["max_neuron_index"]),
                "target_peak_neuron": int(target["max_neuron_index"]),
                "base_peak_value": float(base["max_activation_value"]),
                "target_peak_value": float(target["max_activation_value"]),
                "neuron_rows": neuron_rows,
            }
        )
    return cards


def _summarize_prompt(prompt: str, base_item: Dict[str, Any], target_item: Dict[str, Any]) -> Dict[str, Any]:
    cards = _build_layer_cards(base_item.get("layers", {}), target_item.get("layers", {}))
    if not cards:
        return {"prompt": prompt, "layer_cards": [], "layer_count": 0}
    same_peak_count = sum(1 for card in cards if card["base_peak_neuron"] == card["target_peak_neuron"])
    return {
        "prompt": prompt,
        "layer_cards": cards,
        "layer_count": len(cards),
        "same_peak_neuron_count": same_peak_count,
    }


def _render_prompt_section(summary: Dict[str, Any]) -> str:
    cards = summary["layer_cards"]
    if not cards:
        return f"<section class='prompt-card'><h2>{_escape(summary['prompt'])}</h2><p>No shared layers found.</p></section>"

    layer_blocks = []
    for card in cards:
        base_scale = max(abs(card["base_peak_value"]), 1e-6)
        target_scale = max(abs(card["target_peak_value"]), 1e-6)
        neuron_rows = []
        for row in card["neuron_rows"]:
            base_value = row["base_value"]
            target_value = row["target_value"]
            tags: List[str] = []
            if row["in_base_topk"]:
                tags.append("base-topk")
            if row["in_target_topk"]:
                tags.append("target-topk")
            neuron_rows.append(
                "<tr>"
                f"<td>{row['neuron_index']}</td>"
                f"<td style=\"{_cell_style(base_value or 0.0, base_scale)}\">"
                f"{'NA' if base_value is None else f'{base_value:.6f}'}</td>"
                f"<td style=\"{_cell_style(target_value or 0.0, target_scale)}\">"
                f"{'NA' if target_value is None else f'{target_value:.6f}'}</td>"
                f"<td>{', '.join(tags)}</td>"
                "</tr>"
            )

        layer_blocks.append(
            "<details class='layer-block'>"
            f"<summary>{_escape(card['layer_name'])} | "
            f"base peak: neuron {card['base_peak_neuron']} ({card['base_peak_value']:.6f}) | "
            f"target peak: neuron {card['target_peak_neuron']} ({card['target_peak_value']:.6f})</summary>"
            "<div class='table-wrap'>"
            "<table>"
            "<thead><tr><th>Neuron</th><th>Base value</th><th>Target value</th><th>Source</th></tr></thead>"
            f"<tbody>{''.join(neuron_rows)}</tbody>"
            "</table>"
            "</div>"
            "</details>"
        )

    return (
        "<section class='prompt-card'>"
        f"<h2>{_escape(summary['prompt'])}</h2>"
        "<div class='stats'>"
        f"<div><strong>Layers</strong><span>{summary['layer_count']}</span></div>"
        f"<div><strong>Same peak neuron</strong><span>{summary['same_peak_neuron_count']}/{summary['layer_count']}</span></div>"
        "<div><strong>Rule</strong><span>Base top-k plus target values at same neuron ids</span></div>"
        "<div><strong>Display</strong><span>Target top-k neurons are merged in and duplicates are removed</span></div>"
        "</div>"
        f"{''.join(layer_blocks)}"
        "</section>"
    )


def _render_html(base_path: str, target_path: str, summaries: List[Dict[str, Any]]) -> str:
    sections = "".join(_render_prompt_section(summary) for summary in summaries)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Activation Compare</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffdf8;
      --line: #d8cfbe;
      --ink: #1f1a16;
      --muted: #645b53;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(194,106,51,0.12), transparent 28rem),
        linear-gradient(180deg, #f8f5ef 0%, var(--bg) 100%);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 32px 20px 60px;
    }}
    h1, h2 {{ line-height: 1.1; }}
    .lede {{
      color: var(--muted);
      max-width: 70ch;
      margin-bottom: 22px;
    }}
    .meta, .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .meta {{
      margin-bottom: 26px;
    }}
    .meta div, .stats div {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
    }}
    strong {{
      display: block;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .prompt-card {{
      background: rgba(255,255,255,0.74);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      margin-bottom: 22px;
      backdrop-filter: blur(4px);
    }}
    .warning {{
      background: #fff0e4;
      border: 1px solid #d4a37f;
      color: #6c3c1d;
      border-radius: 14px;
      padding: 14px 16px;
      margin-bottom: 22px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
      margin-top: 16px;
    }}
    .layer-block {{
      margin-top: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .layer-block summary {{
      cursor: pointer;
      font-weight: 600;
      line-height: 1.5;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #efe6d6;
    }}
    tr:nth-child(even) td {{
      background: rgba(239,230,214,0.25);
    }}
  </style>
</head>
<body>
  <main>
    <h1>Activation Pattern Compare</h1>
    <p class="lede">This view compares peak neuron identity and activation strength across matching prompts and layers.</p>
    <div class="meta">
      <div><strong>Base report</strong><span>{_escape(base_path)}</span></div>
      <div><strong>Target report</strong><span>{_escape(target_path)}</span></div>
      <div><strong>Shared prompts</strong><span>{len(summaries)}</span></div>
    </div>
    {sections}
  </main>
</body>
</html>"""


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    act_cfg = cfg.activation
    base_path = args.base or act_cfg.compare_base_file
    target_path = args.target or act_cfg.compare_target_file
    output_path = Path(args.output or act_cfg.compare_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_report = _load_report(base_path)
    target_report = _load_report(target_path)
    base_has_full = _report_has_full_vectors(base_report)
    target_has_full = _report_has_full_vectors(target_report)
    base_map = _prompt_map(base_report)
    target_map = _prompt_map(target_report)

    shared_prompts = [prompt for prompt in base_map if prompt in target_map]
    if not shared_prompts:
        raise ValueError("No shared prompts found between activation reports.")

    summaries = [
        _summarize_prompt(prompt, base_map[prompt], target_map[prompt])
        for prompt in shared_prompts
    ]

    html = _render_html(base_path, target_path, summaries)
    if not base_has_full or not target_has_full:
        warning = (
            "<div class='warning'><strong>Legacy activation report detected</strong>"
            "<span>At least one input JSON does not contain full per-layer activation vectors. "
            "When a neuron id appears only in one model's top-k, the other model's exact activation "
            "cannot be recovered from old reports and will still show as NA. "
            "Re-run `python -m src.activation.inspect` for both models with the current code to get exact cross-model values.</span>"
            "</div>"
        )
        html = html.replace("<div class=\"meta\">", warning + "<div class=\"meta\">", 1)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(html)

    if not base_has_full or not target_has_full:
        print("One or both activation reports are legacy files without full activation vectors.")
        print("Re-run `python -m src.activation.inspect` for both models to replace NA with exact values.")
    print(f"Saved activation visualization to: {output_path}")


if __name__ == "__main__":
    main()
