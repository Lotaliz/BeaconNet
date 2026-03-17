import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config_to_dict, load_config
from src.align.data import build_preference_dataset, load_jsonl_rows


def _missing_dependency_message(exc: Exception) -> str:
    return (
        "Missing training dependency: "
        f"{exc}. Install the required packages first, for example: "
        "`pip install transformers datasets accelerate peft trl`"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPO alignment for llama3.1-8B-Instruct.")
    parser.add_argument("--model-path", default=None, help="Override config.align.model_path.")
    parser.add_argument("--train-dataset", default=None, help="Override config.align.train_dataset_path.")
    parser.add_argument("--eval-dataset", default=None, help="Override config.align.eval_dataset_path.")
    parser.add_argument("--output-dir", default=None, help="Override config.align.output_dir.")
    parser.add_argument(
        "--preference-mode",
        choices=("safer", "better", "safer_then_better"),
        default=None,
        help="How to map PKU-SafeRLHF pairs into chosen/rejected examples.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train sample cap.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional eval sample cap.")
    return parser.parse_args()


def _load_runtime_modules() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer
    except ImportError as exc:
        raise RuntimeError(_missing_dependency_message(exc)) from exc

    try:
        from trl import DPOConfig
    except ImportError:
        DPOConfig = None

    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "TrainingArguments": TrainingArguments,
        "DPOConfig": DPOConfig,
        "DPOTrainer": DPOTrainer,
    }


def _make_training_args(modules: dict[str, Any], align_cfg: Any, output_dir: str):
    dpo_config_cls = modules["DPOConfig"]
    common_kwargs = {
        "output_dir": output_dir,
        "logging_dir": align_cfg.logging_dir,
        "learning_rate": align_cfg.learning_rate,
        "per_device_train_batch_size": align_cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": align_cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": align_cfg.gradient_accumulation_steps,
        "num_train_epochs": align_cfg.num_train_epochs,
        "weight_decay": align_cfg.weight_decay,
        "warmup_ratio": align_cfg.warmup_ratio,
        "logging_steps": align_cfg.logging_steps,
        "save_steps": align_cfg.save_steps,
        "eval_steps": align_cfg.eval_steps,
        "save_total_limit": align_cfg.save_total_limit,
        "seed": align_cfg.seed,
        "report_to": align_cfg.report_to,
        "gradient_checkpointing": align_cfg.gradient_checkpointing,
        "remove_unused_columns": False,
        "bf16": align_cfg.use_bf16,
        "fp16": align_cfg.use_fp16,
    }
    if dpo_config_cls is not None:
        return dpo_config_cls(
            eval_strategy="steps",
            save_strategy="steps",
            max_prompt_length=align_cfg.max_prompt_length,
            max_length=align_cfg.max_length,
            beta=align_cfg.beta,
            **common_kwargs,
        )
    return modules["TrainingArguments"](
        evaluation_strategy="steps",
        save_strategy="steps",
        **common_kwargs,
    )


def _load_model_and_tokenizer(modules: dict[str, Any], model_path: str, align_cfg: Any):
    torch = modules["torch"]
    tokenizer = modules["AutoTokenizer"].from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = modules["AutoModelForCausalLM"].from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if align_cfg.use_bf16 else (torch.float16 if align_cfg.use_fp16 else torch.float32),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model, tokenizer


def _build_peft_config(modules: dict[str, Any], align_cfg: Any):
    if not align_cfg.use_lora:
        return None
    return modules["LoraConfig"](
        r=align_cfg.lora_r,
        lora_alpha=align_cfg.lora_alpha,
        lora_dropout=align_cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(align_cfg.lora_target_modules),
    )


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    align_cfg = cfg.align

    model_path = args.model_path or align_cfg.model_path
    train_dataset_path = args.train_dataset or align_cfg.train_dataset_path
    eval_dataset_path = args.eval_dataset or align_cfg.eval_dataset_path
    output_dir = args.output_dir or align_cfg.output_dir
    preference_mode = args.preference_mode or align_cfg.preference_mode
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else align_cfg.max_train_samples
    max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else align_cfg.max_eval_samples

    modules = _load_runtime_modules()
    model, tokenizer = _load_model_and_tokenizer(modules, model_path=model_path, align_cfg=align_cfg)

    train_rows = load_jsonl_rows(train_dataset_path)
    eval_rows = load_jsonl_rows(eval_dataset_path)
    train_dataset = build_preference_dataset(
        train_rows,
        tokenizer=tokenizer,
        preference_mode=preference_mode,
        max_samples=max_train_samples,
    )
    eval_dataset = build_preference_dataset(
        eval_rows,
        tokenizer=tokenizer,
        preference_mode=preference_mode,
        max_samples=max_eval_samples,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(align_cfg.logging_dir).mkdir(parents=True, exist_ok=True)

    training_args = _make_training_args(modules, align_cfg=align_cfg, output_dir=output_dir)
    peft_config = _build_peft_config(modules, align_cfg=align_cfg)
    ref_model = None
    if peft_config is None:
        ref_model, _ = _load_model_and_tokenizer(modules, model_path=model_path, align_cfg=align_cfg)

    trainer_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    trainer_signature = inspect.signature(modules["DPOTrainer"].__init__)
    if "beta" in trainer_signature.parameters:
        trainer_kwargs["beta"] = align_cfg.beta
    if "max_length" in trainer_signature.parameters:
        trainer_kwargs["max_length"] = align_cfg.max_length
    if "max_prompt_length" in trainer_signature.parameters:
        trainer_kwargs["max_prompt_length"] = align_cfg.max_prompt_length
    if "peft_config" in trainer_signature.parameters:
        trainer_kwargs["peft_config"] = peft_config
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = modules["DPOTrainer"](**trainer_kwargs)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "config": config_to_dict(cfg),
        "resolved": {
            "model_path": model_path,
            "train_dataset_path": train_dataset_path,
            "eval_dataset_path": eval_dataset_path,
            "output_dir": output_dir,
            "preference_mode": preference_mode,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
        },
    }
    metadata_path = Path(output_dir) / "dpo_training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"DPO alignment finished. Model artifacts saved to: {output_dir}")
    print(f"Training metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
