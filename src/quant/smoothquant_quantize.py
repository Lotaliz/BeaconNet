import argparse
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant.common import (
    build_text_calibration_dataset,
    cleanup_materialized_source,
    load_config,
    load_mixed_calibration_prompts,
    materialize_quantization_source,
    resolve_model_and_adapter_paths,
    save_calibration_artifacts,
    save_quant_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a model with SmoothQuant.")
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("-m", "--model", type=str, default=None, help="Base model directory to quantize.")
    parser.add_argument("-a", "--adapter-path", type=str, default=None, help="Optional LoRA adapter directory to merge.")
    parser.add_argument("--alpha", type=float, default=None, help="SmoothQuant smoothing strength.")
    parser.add_argument("--scheme", type=str, default=None, help="Quantization scheme, e.g. W8A8.")
    return parser.parse_args()


def _get_smoothquant_runtime_dtype(default_dtype: torch.dtype) -> torch.dtype:
    if default_dtype == torch.float16 and torch.cuda.is_available():
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_supported) and is_bf16_supported():
            return torch.bfloat16
    return default_dtype


def _build_smoothquant_recipe(quant_cfg):
    return [
        {
            "modifier": "SmoothQuantModifier",
            "config": {
                "smoothing_strength": quant_cfg.smoothquant_smoothing_strength,
            },
        },
        {
            "modifier": "GPTQModifier",
            "config": {
                "scheme": quant_cfg.smoothquant_scheme,
                "targets": "Linear",
                "ignore": list(quant_cfg.smoothquant_ignore),
                "dampening_frac": float(quant_cfg.gptq_damp_percent),
            },
        },
    ]


def main() -> None:
    args = parse_args()
    cfg = load_config()
    quant_cfg = cfg.quant

    if args.name is not None:
        quant_cfg.model_name = args.name
    if args.model is not None:
        quant_cfg.model_path = args.model
    if args.adapter_path is not None:
        quant_cfg.lora_adapter_path = args.adapter_path
    if args.alpha is not None:
        quant_cfg.smoothquant_smoothing_strength = args.alpha
    if args.scheme is not None:
        quant_cfg.smoothquant_scheme = args.scheme

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        from llmcompressor.modifiers.quantization import GPTQModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "SmoothQuant quantization requires `llmcompressor` and `transformers`. "
            "Install them before running src/quant/smoothquant_quantize.py."
        ) from exc

    resolved_model_path, resolved_adapter_path = resolve_model_and_adapter_paths(
        quant_cfg.model_path,
        quant_cfg.lora_adapter_path,
    )

    random.seed(quant_cfg.seed)
    torch.manual_seed(quant_cfg.seed)

    output_dir = Path(quant_cfg.smoothquant_save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dtype = _get_smoothquant_runtime_dtype(quant_cfg.torch_dtype)

    quant_source_path = None
    source_model_path = resolved_model_path
    try:
        source_model_path, quant_source_path = materialize_quantization_source(
            resolved_model_path,
            resolved_adapter_path,
            runtime_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(source_model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        prompts, calibration_mix = load_mixed_calibration_prompts(
            cfg=cfg,
            limit=quant_cfg.calibration_samples,
            seed=quant_cfg.seed,
        )
        calibration_dataset = build_text_calibration_dataset(prompts)

        model = AutoModelForCausalLM.from_pretrained(
            source_model_path,
            torch_dtype=runtime_dtype,
            device_map="auto",
        )
        recipe_metadata = _build_smoothquant_recipe(quant_cfg)
        recipe = [
            SmoothQuantModifier(
                smoothing_strength=quant_cfg.smoothquant_smoothing_strength,
            ),
            GPTQModifier(
                scheme=quant_cfg.smoothquant_scheme,
                targets="Linear",
                ignore=list(quant_cfg.smoothquant_ignore),
                dampening_frac=float(quant_cfg.gptq_damp_percent),
            ),
        ]
        oneshot(
            model=model,
            recipe=recipe,
            dataset=calibration_dataset,
            tokenizer=tokenizer,
            max_seq_length=quant_cfg.max_length,
            num_calibration_samples=len(prompts),
        )

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        save_calibration_artifacts(output_dir, prompts, calibration_mix)
        save_quant_summary(
            output_dir=output_dir,
            method="smoothquant",
            runtime_config={
                "backend": "llmcompressor",
                "base_model_path": resolved_model_path,
                "lora_adapter_path": resolved_adapter_path or "",
                "quant_source_path": source_model_path,
                "scheme": quant_cfg.smoothquant_scheme,
                "runtime_torch_dtype": str(runtime_dtype),
                "smoothing_strength": quant_cfg.smoothquant_smoothing_strength,
                "damp_percent": quant_cfg.gptq_damp_percent,
                "ignore": list(quant_cfg.smoothquant_ignore),
                "calibration_prompt_count": len(prompts),
                "calibration_mix": calibration_mix,
                "recipe": recipe_metadata,
            },
            cfg=cfg,
        )
    finally:
        cleanup_materialized_source(quant_source_path)

    print(f"SmoothQuant quantization complete. Saved quantized model to: {output_dir}")


if __name__ == "__main__":
    main()
