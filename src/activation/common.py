from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path: str, adapter_path: str, dtype: torch.dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "LoRA adapter loading requires `peft`. Install it first, for example: `pip install peft`."
            ) from exc
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.to(device)
    model.eval()
    return model


def sanitize_filename(text: str) -> str:
    return "__".join(part for part in text.replace("/", "__").split("__") if part)


def build_prompt_inputs(
    tokenizer,
    prompt: str,
    max_length: int,
) -> Tuple[Dict[str, torch.Tensor], int]:
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
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

    prompt_length = int(encoded["input_ids"].shape[-1])
    prompt_last_token_index = max(prompt_length - 1, 0)
    return encoded, prompt_last_token_index


def build_prompt_response_inputs(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> Tuple[Dict[str, torch.Tensor], Tuple[int, int]]:
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + response
        prompt_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]
        encoded = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
    else:
        prompt_text = prompt
        full_text = f"{prompt}\n{response}"
        prompt_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        encoded = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

    full_ids = encoded["input_ids"]
    full_length = int(full_ids.shape[-1])
    prompt_length = min(int(prompt_ids.shape[-1]), full_length)
    assistant_start = min(prompt_length, max(full_length - 1, 0))
    assistant_end = max(assistant_start + 1, full_length)
    return encoded, (assistant_start, assistant_end)


def move_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
