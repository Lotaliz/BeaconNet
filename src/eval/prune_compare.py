from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import config_to_dict, load_config


def _load_alpaca_samples(path: str, sample_size: int, seed: int) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Alpaca dataset not found: {path}")

    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)

    if not rows:
        raise ValueError(f"No valid rows found in dataset: {path}")

    rng = random.Random(seed)
    if sample_size >= len(rows):
        return rows
    return rng.sample(rows, sample_size)


def _build_prompt(row: Dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    model_input = str(row.get("input", "")).strip()
    if instruction and model_input:
        return f"Instruction: {instruction}\nInput: {model_input}"
    if instruction:
        return instruction
    if model_input:
        return model_input
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_model_and_tokenizer(model_path: str, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
    return model, tokenizer


def _generate_responses(
    model_path: str,
    prompts: List[str],
    device: str,
    dtype: torch.dtype,
    max_length: int,
    max_new_tokens: int,
) -> List[str]:
    model, tokenizer = _load_model_and_tokenizer(model_path, device, dtype)
    responses: List[str] = []

    try:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            if hasattr(tokenizer, "apply_chat_template"):
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                )
            else:
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )

            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_len = encoded["input_ids"].shape[-1]
            text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
            responses.append(text)
    finally:
        del model
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return responses


def main() -> None:
    cfg = load_config()
    output_path = Path(cfg.prune_eval_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_alpaca_samples(
        path=cfg.alpaca_dataset_path,
        sample_size=cfg.prune_eval_sample_size,
        seed=cfg.prune_eval_seed,
    )
    prompts = [_build_prompt(row) for row in rows]

    base_outputs = _generate_responses(
        model_path=cfg.model_path,
        prompts=prompts,
        device=cfg.device,
        dtype=cfg.dtype,
        max_length=cfg.max_length,
        max_new_tokens=cfg.prune_eval_max_new_tokens,
    )
    pruned_outputs = _generate_responses(
        model_path=cfg.prune.pruned_model_path,
        prompts=prompts,
        device=cfg.prune.device,
        dtype=cfg.prune.torch_dtype,
        max_length=cfg.prune.max_length,
        max_new_tokens=cfg.prune_eval_max_new_tokens,
    )

    results = []
    for row, prompt, base_output, pruned_output in zip(rows, prompts, base_outputs, pruned_outputs):
        results.append(
            {
                "prompt": prompt,
                "reference_output": row.get("output", ""),
                "base_model_output": base_output,
                "pruned_model_output": pruned_output,
            }
        )

    payload = {
        "config": config_to_dict(cfg),
        "sample_count": len(results),
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved base/pruned model comparison to: {output_path}")


if __name__ == "__main__":
    main()
