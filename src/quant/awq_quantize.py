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
    parser = argparse.ArgumentParser(description="Quantize a model with AWQ via llmcompressor.")
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("-m", "--model", type=str, default=None, help="Base model directory to quantize.")
    parser.add_argument("-a", "--adapter-path", type=str, default=None, help="Optional LoRA adapter directory to merge.")
    parser.add_argument("--bits", type=int, default=None, help="Weight bit width.")
    parser.add_argument("--group-size", type=int, default=None, help="AWQ group size.")
    parser.add_argument("--version", type=str, default=None, help="Deprecated AutoAWQ backend flag; kept for config compatibility.")
    parser.add_argument("--disable-zero-point", action="store_true", help="Use symmetric weights instead of asymmetric zero-point quantization.")
    parser.add_argument("--duo-scaling", action="store_true", help="Enable duo scaling grid search.")
    parser.add_argument("--n-grid", type=int, default=None, help="Grid size for AWQ scale search.")
    return parser.parse_args()


def _build_awq_recipe_config(quant_cfg):
    symmetric = not quant_cfg.awq_zero_point
    return {
        "ignore": list(quant_cfg.smoothquant_ignore),
        "duo_scaling": quant_cfg.awq_duo_scaling,
        "n_grid": int(quant_cfg.awq_n_grid),
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": int(quant_cfg.awq_bits),
                    "type": "int",
                    "symmetric": symmetric,
                    "strategy": "group",
                    "group_size": int(quant_cfg.awq_group_size),
                },
            }
        }
    }


def _build_llama_stable_mappings():
    return [
        {
            "smooth_layer": "re:.*input_layernorm$",
            "balance_layers": ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
        },
        {
            "smooth_layer": "re:.*post_attention_layernorm$",
            "balance_layers": ["re:.*gate_proj$", "re:.*up_proj$"],
        },
        {
            "smooth_layer": "re:.*up_proj$",
            "balance_layers": ["re:.*down_proj$"],
        },
    ]


def _get_awq_runtime_dtype(default_dtype: torch.dtype) -> torch.dtype:
    if default_dtype == torch.float16 and torch.cuda.is_available():
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_supported) and is_bf16_supported():
            return torch.bfloat16
    return default_dtype


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
    if args.bits is not None:
        quant_cfg.awq_bits = args.bits
    if args.group_size is not None:
        quant_cfg.awq_group_size = args.group_size
    if args.version is not None:
        quant_cfg.awq_version = args.version
    if args.disable_zero_point:
        quant_cfg.awq_zero_point = False
    if args.duo_scaling:
        quant_cfg.awq_duo_scaling = True
    if args.n_grid is not None:
        quant_cfg.awq_n_grid = args.n_grid

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQMapping, AWQModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "AWQ quantization now uses `llmcompressor` plus `transformers`. "
            "Install them before running src/quant/awq_quantize.py."
        ) from exc

    resolved_model_path, resolved_adapter_path = resolve_model_and_adapter_paths(
        quant_cfg.model_path,
        quant_cfg.lora_adapter_path,
    )

    random.seed(quant_cfg.seed)
    torch.manual_seed(quant_cfg.seed)

    output_dir = Path(quant_cfg.awq_save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dtype = _get_awq_runtime_dtype(quant_cfg.torch_dtype)

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
        recipe_config = _build_awq_recipe_config(quant_cfg)
        if model.__class__.__name__ == "LlamaForCausalLM":
            recipe_config["mappings"] = [
                AWQMapping(**mapping)
                for mapping in _build_llama_stable_mappings()
            ]
        recipe = AWQModifier(**recipe_config)
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
            method="awq",
            runtime_config={
                "backend": "llmcompressor",
                "base_model_path": resolved_model_path,
                "lora_adapter_path": resolved_adapter_path or "",
                "quant_source_path": source_model_path,
                "bits": quant_cfg.awq_bits,
                "group_size": quant_cfg.awq_group_size,
                "runtime_torch_dtype": str(runtime_dtype),
                "zero_point": quant_cfg.awq_zero_point,
                "symmetric": not quant_cfg.awq_zero_point,
                "duo_scaling": quant_cfg.awq_duo_scaling,
                "n_grid": quant_cfg.awq_n_grid,
                "version": quant_cfg.awq_version,
                "calibration_prompt_count": len(prompts),
                "calibration_mix": calibration_mix,
                "recipe": {
                    **{k: v for k, v in recipe_config.items() if k != "mappings"},
                    "mappings": _build_llama_stable_mappings()
                    if model.__class__.__name__ == "LlamaForCausalLM"
                    else "auto",
                },
            },
            cfg=cfg,
        )
    finally:
        cleanup_materialized_source(quant_source_path)

    print(f"AWQ quantization complete. Saved quantized model to: {output_dir}")


if __name__ == "__main__":
    main()
