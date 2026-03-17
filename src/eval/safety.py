from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import config_to_dict, load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate safety of base and pruned models.")
    parser.add_argument(
        "-m",
        "--model-path",
        default=None,
        help="Optional base model path override. Defaults to cfg.model_path.",
    )
    parser.add_argument(
        "-p",
        "--pruned-model-path",
        default=None,
        help="Optional pruned model path override. Defaults to cfg.prune.pruned_model_path.",
    )
    return parser.parse_args()


def _load_tokenizer(model_path: str, trust_remote_code: bool = True):
    return AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        fix_mistral_regex=True,
    )


def _load_json_dataset(path: str) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list in {path}")
    rows = [row for row in data if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No valid rows found in dataset: {path}")
    return rows


def _extract_prompt(row: Dict[str, Any]) -> str:
    for key in ("prompt", "question", "instruction", "input", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _sample_rows(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    valid_rows = [row for row in rows if _extract_prompt(row)]
    if not valid_rows:
        raise ValueError("No rows with usable prompts found.")
    if sample_size >= len(valid_rows):
        return valid_rows
    rng = random.Random(seed)
    return rng.sample(valid_rows, sample_size)


def _load_model_and_tokenizer(model_path: str, device: str, dtype: torch.dtype):
    tokenizer = _load_tokenizer(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
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
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = tokenizer(
                    prompt_text,
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
            response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
            responses.append(response)
    finally:
        del model
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return responses


def _load_guard(cfg):
    tokenizer = _load_tokenizer(cfg.llama_guard_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llama_guard_model_path,
        torch_dtype=cfg.dtype,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(cfg.device)
    model.eval()
    return model, tokenizer


def _guard_raw_output(
    prompt: str,
    response: str,
    guard_model,
    guard_tokenizer,
    device: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    input_ids = guard_tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)
    with torch.no_grad():
        output = guard_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=guard_tokenizer.pad_token_id or guard_tokenizer.eos_token_id,
            eos_token_id=guard_tokenizer.eos_token_id,
        )
    prompt_len = input_ids.shape[-1]
    return guard_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


def _is_unsafe_guard_output(raw_output: str) -> bool:
    normalized = raw_output.strip().lower()
    return normalized.startswith("unsafe")


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    model_path = args.model_path or cfg.model_path
    pruned_model_path = args.pruned_model_path or cfg.prune.pruned_model_path
    torch.manual_seed(cfg.safety_seed)
    random.seed(cfg.safety_seed)

    output_dir = Path(cfg.safety_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    guard_model, guard_tokenizer = _load_guard(cfg)
    total_count = 0
    total_base_unsafe = 0
    total_pruned_unsafe = 0

    for dataset_name, dataset_path in cfg.safety_dataset_paths.items():
        rows = _load_json_dataset(dataset_path)
        sampled_rows = _sample_rows(rows, cfg.safety_sample_size, cfg.safety_seed)
        prompts = [_extract_prompt(row) for row in sampled_rows]

        base_outputs = _generate_responses(
            model_path=model_path,
            prompts=prompts,
            device=cfg.device,
            dtype=cfg.dtype,
            max_length=cfg.max_length,
            max_new_tokens=cfg.safety_generation_max_new_tokens,
        )
        pruned_outputs = _generate_responses(
            model_path=pruned_model_path,
            prompts=prompts,
            device=cfg.prune.device,
            dtype=cfg.prune.torch_dtype,
            max_length=cfg.prune.max_length,
            max_new_tokens=cfg.safety_generation_max_new_tokens,
        )

        results = []
        for row, prompt, base_output, pruned_output in zip(sampled_rows, prompts, base_outputs, pruned_outputs):
            results.append(
                {
                    "prompt": prompt,
                    "meta": row,
                    "base_model_output": base_output,
                    "pruned_model_output": pruned_output,
                    "base_guard_raw_output": _guard_raw_output(
                        prompt,
                        base_output,
                        guard_model,
                        guard_tokenizer,
                        cfg.device,
                        cfg.safety_guard_max_new_tokens,
                    ),
                    "pruned_guard_raw_output": _guard_raw_output(
                        prompt,
                        pruned_output,
                        guard_model,
                        guard_tokenizer,
                        cfg.device,
                        cfg.safety_guard_max_new_tokens,
                    ),
                }
            )

        payload = {
            "config": config_to_dict(cfg),
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "sample_count": len(results),
            "results": results,
        }
        output_path = output_dir / f"{dataset_name}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"Saved safety comparison results to: {output_path}")

        base_unsafe = sum(1 for row in results if _is_unsafe_guard_output(row["base_guard_raw_output"]))
        pruned_unsafe = sum(1 for row in results if _is_unsafe_guard_output(row["pruned_guard_raw_output"]))
        sample_count = len(results)
        base_unsafe_rate = base_unsafe / sample_count if sample_count else 0.0
        pruned_unsafe_rate = pruned_unsafe / sample_count if sample_count else 0.0
        print(
            f"[{dataset_name}] unsafe rate | "
            f"base: {base_unsafe}/{sample_count} ({base_unsafe_rate:.2%}) | "
            f"pruned: {pruned_unsafe}/{sample_count} ({pruned_unsafe_rate:.2%})"
        )

        total_count += sample_count
        total_base_unsafe += base_unsafe
        total_pruned_unsafe += pruned_unsafe

    del guard_model
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_base_rate = total_base_unsafe / total_count if total_count else 0.0
    total_pruned_rate = total_pruned_unsafe / total_count if total_count else 0.0
    print(
        f"[overall] unsafe rate | "
        f"base: {total_base_unsafe}/{total_count} ({total_base_rate:.2%}) | "
        f"pruned: {total_pruned_unsafe}/{total_count} ({total_pruned_rate:.2%})"
    )


if __name__ == "__main__":
    main()
