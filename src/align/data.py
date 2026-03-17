import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Alignment dataset not found: {path}")

    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if isinstance(item, dict):
                rows.append(item)
    if not rows:
        raise ValueError(f"No valid rows found in alignment dataset: {path}")
    return rows

def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _select_pair(row: Dict[str, Any], preference_mode: str) -> tuple[str, str] | None:
    responses = {
        0: _normalize_text(row.get("response_0")),
        1: _normalize_text(row.get("response_1")),
    }
    if not responses[0] or not responses[1]:
        return None

    safe_flags = {
        0: bool(row.get("is_response_0_safe")),
        1: bool(row.get("is_response_1_safe")),
    }

    if preference_mode in {"safer", "safer_then_better"} and safe_flags[0] != safe_flags[1]:
        chosen_id = 0 if safe_flags[0] else 1
        rejected_id = 1 - chosen_id
        return responses[chosen_id], responses[rejected_id]

    key_by_mode = {
        "safer": "safer_response_id",
        "better": "better_response_id",
        "safer_then_better": "better_response_id",
    }
    preference_key = key_by_mode.get(preference_mode)
    if preference_key is None:
        raise ValueError(f"Unsupported preference mode: {preference_mode}")

    preferred_id = row.get(preference_key)
    if preferred_id not in (0, 1):
        return None
    chosen_id = int(preferred_id)
    rejected_id = 1 - chosen_id
    return responses[chosen_id], responses[rejected_id]


def format_user_prompt(prompt: str, tokenizer: Any) -> str:
    content = prompt.strip()
    if not content:
        return ""

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def build_preference_dataset(
    rows: Iterable[Dict[str, Any]],
    tokenizer: Any,
    preference_mode: str,
    max_samples: int = 0,
):
    from datasets import Dataset

    examples: List[Dict[str, str]] = []
    for row in rows:
        prompt = _normalize_text(row.get("prompt"))
        if not prompt:
            continue
        selected = _select_pair(row, preference_mode=preference_mode)
        if selected is None:
            continue
        chosen, rejected = selected
        formatted_prompt = format_user_prompt(prompt, tokenizer)
        if not formatted_prompt:
            continue
        examples.append(
            {
                "prompt": formatted_prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        if max_samples > 0 and len(examples) >= max_samples:
            break

    if not examples:
        raise ValueError("No usable preference pairs were produced for DPO training.")
    return Dataset.from_list(examples)
