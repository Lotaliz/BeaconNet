import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import _activation_report_filename, config_to_dict, load_config
from src.activation.hooks import ActivationCollector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect model activations for specific prompts.")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to inspect. Repeat the flag to inspect multiple prompts.",
    )
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
    return parser.parse_args()


def _load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )


def _prepare_prompts(cli_prompts: List[str] | None, default_prompts: List[str]) -> List[str]:
    prompts = cli_prompts if cli_prompts else default_prompts
    cleaned = [prompt.strip() for prompt in prompts if prompt and prompt.strip()]
    if not cleaned:
        raise ValueError("No valid prompts provided for activation inspection.")
    return cleaned


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    act_cfg = cfg.activation
    model_path = args.model_path or act_cfg.model_path
    prompts = _prepare_prompts(args.prompt, act_cfg.prompts)

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

    collector = ActivationCollector(
        model=model,
        target_names=act_cfg.target_linear_names,
        top_k=act_cfg.top_k,
        token_position=act_cfg.token_position,
    )
    collector.register()

    results = []
    try:
        for prompt in prompts:
            collector.clear()
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

            prompt_result = {
                "prompt": prompt,
                "layer_count": len(collector.summaries()),
                "layers": {},
            }
            for name, summary in collector.summaries().items():
                layer_data = asdict(summary)
                if not act_cfg.save_full_vector:
                    layer_data.pop("activation_vector", None)
                prompt_result["layers"][name] = layer_data
            results.append(prompt_result)

            global_peak = max(
                collector.summaries().values(),
                key=lambda summary: abs(summary.max_activation_value),
            )
            print(
                f"[activation] prompt={prompt[:60]!r} | "
                f"peak_layer={global_peak.layer_name} | "
                f"neuron={global_peak.max_neuron_index} | "
                f"value={global_peak.max_activation_value:.6f}"
            )
    finally:
        collector.remove()
        del model
        if act_cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "config": config_to_dict(cfg),
        "activation_model_path": model_path,
        "has_full_activation_vector": bool(act_cfg.save_full_vector),
        "prompt_count": len(results),
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved activation report to: {output_path}")


if __name__ == "__main__":
    main()
