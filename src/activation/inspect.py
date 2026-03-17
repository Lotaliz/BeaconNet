import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import _activation_report_filename, config_to_dict, load_config
from src.activation.hooks import AverageActivationCollector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect average model activations over dataset prompt fields.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional model path override. Defaults to config.activation.model_path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path override. Defaults to report-{ratio}-{model}.json naming.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional dataset JSON path override. Defaults to config.activation.dataset_path.",
    )
    return parser.parse_args()


def _load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )


def _load_dataset(path: str) -> List[Dict[str, object]]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Activation dataset not found: {path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list in {path}")
    return [row for row in data if isinstance(row, dict)]


def _iter_category_prompts(rows: List[Dict[str, object]], field_name: str, max_samples: int) -> List[str]:
    prompts: List[str] = []
    for row in rows:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            prompts.append(value.strip())
    if max_samples > 0:
        prompts = prompts[:max_samples]
    if not prompts:
        raise ValueError(f"No valid prompts found for field {field_name!r}.")
    return prompts


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    act_cfg = cfg.activation
    model_path = args.model_path or act_cfg.model_path
    model_name = Path(model_path).name
    dataset_path = args.dataset_path or act_cfg.dataset_path
    rows = _load_dataset(dataset_path)

    output_path = Path(args.output) if args.output else Path(act_cfg.output_dir) / _activation_report_filename(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=act_cfg.torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(act_cfg.device)
    model.eval()

    collector = AverageActivationCollector(
        model=model,
        target_names=act_cfg.target_linear_names,
        top_k=act_cfg.top_k,
        token_position=act_cfg.token_position,
    )
    collector.register()

    categories: Dict[str, Dict[str, object]] = {}
    try:
        for field_name in act_cfg.prompt_fields:
            collector.reset()
            prompts = _iter_category_prompts(rows, field_name, act_cfg.max_samples_per_field)
            for prompt in tqdm(
                prompts,
                desc=f"activation:{model_name}:{field_name}",
                leave=False,
            ):
                messages = [{"role": "user", "content": prompt}]
                if hasattr(tokenizer, "apply_chat_template"):
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    encoded = tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=act_cfg.max_length,
                        add_special_tokens=False,
                    )
                else:
                    encoded = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=act_cfg.max_length,
                    )
                encoded = {key: value.to(act_cfg.device) for key, value in encoded.items()}

                with torch.no_grad():
                    _ = model(**encoded, use_cache=False)

            category_layers = {}
            for name, summary in collector.summaries().items():
                layer_data = asdict(summary)
                if not act_cfg.save_full_vector:
                    layer_data.pop("avg_activation_vector", None)
                category_layers[name] = layer_data

            global_peak = max(
                collector.summaries().values(),
                key=lambda summary: abs(summary.max_activation_value),
            )
            print(
                f"[activation] category={field_name!r} | samples={len(prompts)} | "
                f"peak_layer={global_peak.layer_name} | "
                f"neuron={global_peak.max_neuron_index} | "
                f"value={global_peak.max_activation_value:.6f}"
            )
            categories[field_name] = {
                "sample_count": len(prompts),
                "layer_count": len(category_layers),
                "layers": category_layers,
            }
    finally:
        collector.remove()
        del model
        if act_cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "config": config_to_dict(cfg),
        "activation_model_path": model_path,
        "dataset_path": dataset_path,
        "has_full_activation_vector": bool(act_cfg.save_full_vector),
        "category_count": len(categories),
        "categories": categories,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved activation report to: {output_path}")


if __name__ == "__main__":
    main()
