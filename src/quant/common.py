import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config_to_dict, load_config  # noqa: E402


ALPACA_RATIO = 0.8
PKU_RATIO = 0.2


def load_rows(path: str) -> List[dict]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Calibration dataset not found: {path}")

    rows: List[dict] = []
    with file_path.open("r", encoding="utf-8") as handle:
        if file_path.suffix == ".jsonl":
            for line in handle:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
        else:
            payload = json.load(handle)
            if isinstance(payload, list):
                rows = [row for row in payload if isinstance(row, dict)]

    if not rows:
        raise ValueError(f"No usable rows found in calibration dataset: {path}")
    return rows


def extract_prompt(row: object) -> str:
    if not isinstance(row, dict):
        return ""

    instruction = str(row.get("instruction", "")).strip()
    model_input = str(row.get("input", "")).strip()
    if instruction and model_input:
        return f"Instruction: {instruction}\nInput: {model_input}"
    if instruction:
        return instruction

    for key in ("prompt", "question", "input", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def sample_prompts(rows: List[dict], sample_size: int, seed: int) -> List[str]:
    prompts = [prompt for prompt in (extract_prompt(row) for row in rows) if prompt]
    if not prompts:
        return []

    rng = random.Random(seed)
    rng.shuffle(prompts)
    if sample_size >= len(prompts):
        return prompts
    return prompts[:sample_size]


def sample_tagged_prompts(rows: List[dict], sample_size: int, seed: int, source: str) -> List[Tuple[str, str]]:
    prompts = sample_prompts(rows, sample_size, seed)
    return [(prompt, source) for prompt in prompts]


def load_mixed_calibration_prompts(cfg, limit: int, seed: int) -> Tuple[List[str], Dict[str, object]]:
    if limit <= 0:
        raise ValueError(f"calibration_samples must be positive, got {limit}")

    alpaca_rows = load_rows(cfg.alpaca_dataset_path)
    pku_rows = load_rows(cfg.align.train_dataset_path)

    alpaca_target = int(round(limit * ALPACA_RATIO))
    alpaca_target = min(max(alpaca_target, 0), limit)
    pku_target = limit - alpaca_target

    alpaca_prompts = sample_prompts(alpaca_rows, alpaca_target, seed)
    pku_prompts = sample_prompts(pku_rows, pku_target, seed + 1)

    deficit = limit - (len(alpaca_prompts) + len(pku_prompts))
    if deficit > 0:
        extra_alpaca = sample_tagged_prompts(alpaca_rows, limit, seed + 2, "alpaca")
        extra_pku = sample_tagged_prompts(pku_rows, limit, seed + 3, "pku")
        used = set(alpaca_prompts) | set(pku_prompts)
        for prompt, source in extra_alpaca + extra_pku:
            if prompt in used:
                continue
            if len(alpaca_prompts) + len(pku_prompts) >= limit:
                break
            used.add(prompt)
            if source == "alpaca":
                alpaca_prompts.append(prompt)
            else:
                pku_prompts.append(prompt)

    prompts = alpaca_prompts + pku_prompts
    if len(prompts) < limit:
        raise ValueError(
            f"Unable to build {limit} calibration prompts from Alpaca and PKU-SafeRLHF; got {len(prompts)} prompts."
        )

    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:limit], {
        "alpaca_dataset_path": cfg.alpaca_dataset_path,
        "pku_dataset_path": cfg.align.train_dataset_path,
        "alpaca_ratio": ALPACA_RATIO,
        "pku_ratio": PKU_RATIO,
        "alpaca_prompt_count": len(alpaca_prompts),
        "pku_prompt_count": len(pku_prompts),
        "total_prompt_count": min(len(prompts), limit),
    }


def looks_like_adapter_dir(path: str | None) -> bool:
    if not path:
        return False
    return (Path(path) / "adapter_config.json").is_file()


def resolve_model_and_adapter_paths(model_path: str, adapter_path: str | None) -> tuple[str, str | None]:
    if adapter_path:
        return model_path, adapter_path

    if looks_like_adapter_dir(model_path):
        adapter_cfg_path = Path(model_path) / "adapter_config.json"
        with adapter_cfg_path.open("r", encoding="utf-8") as handle:
            adapter_cfg = json.load(handle)
        base_model_path = str(adapter_cfg.get("base_model_name_or_path", "")).strip()
        if not base_model_path:
            raise ValueError(f"LoRA adapter config is missing base_model_name_or_path: {adapter_cfg_path}")
        return base_model_path, model_path

    return model_path, None


def materialize_quantization_source(model_path: str, adapter_path: str | None, torch_dtype):
    if not adapter_path:
        return model_path, None

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Quantizing a LoRA-aligned model requires `transformers` and `peft`. Install them before running."
        ) from exc

    merged_dir = Path(tempfile.mkdtemp(prefix="beaconnet-merged-", dir=str(PROJECT_ROOT / "models")))
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    return str(merged_dir), str(merged_dir)


def cleanup_materialized_source(temp_path: str | None) -> None:
    if not temp_path:
        return
    shutil.rmtree(temp_path, ignore_errors=True)


def save_calibration_artifacts(output_dir: Path, prompts: List[str], calibration_mix: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "calibration_prompts.json").open("w", encoding="utf-8") as handle:
        json.dump(prompts, handle, ensure_ascii=False, indent=2)
    with (output_dir / "calibration_mix.json").open("w", encoding="utf-8") as handle:
        json.dump(calibration_mix, handle, ensure_ascii=False, indent=2)


def build_text_calibration_dataset(prompts: List[str]):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Building a calibration dataset for llmcompressor requires the `datasets` package."
        ) from exc
    return Dataset.from_dict({"text": prompts})


def save_quant_summary(
    output_dir: Path,
    method: str,
    runtime_config: Dict[str, object],
    cfg=None,
) -> None:
    summary = {
        "config": config_to_dict(cfg or load_config()),
        "quantization": {
            "method": method,
            **runtime_config,
            "output_dir": str(output_dir),
        },
    }
    with (output_dir / "quant_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
