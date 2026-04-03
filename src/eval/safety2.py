import argparse
import csv
import gc
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config


XGUARD_RELATIVE_PATH = Path("models") / "XGuard-8B"
DEFAULT_OUTPUT_DIRNAME = "safety2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model with XGuard-8B response safety scoring.")
    parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        help="Model path to evaluate on the configured safety datasets.",
    )
    parser.add_argument(
        "-k",
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample count override. Defaults to cfg.safety_sample_size.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Optional output directory override. Defaults to data/safety2.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional single dataset path override. Supports .json and .jsonl.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name override when --dataset-path is used.",
    )
    return parser.parse_args()


def _load_json_dataset(path: str) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        if dataset_path.suffix == ".tsv":
            reader = csv.DictReader(handle, delimiter="\t")
            rows = [row for row in reader if isinstance(row, dict)]
        elif dataset_path.suffix == ".jsonl":
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
        else:
            data = json.load(handle)
            if not isinstance(data, list):
                raise TypeError(f"Expected a JSON list in {path}")
            rows = [row for row in data if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No valid rows found in dataset: {path}")
    return rows


def _build_result_config(cfg, model_path: str, xguard_model_path: str, sample_size: int) -> Dict[str, Any]:
    model_config = _load_model_config(model_path)
    quantization_config = model_config.get("quantization_config") if isinstance(model_config, dict) else None
    return {
        "project_root": cfg.project_root,
        "data_root": cfg.data_root,
        "model_path": model_path,
        "xguard_model_path": xguard_model_path,
        "device": cfg.device,
        "dtype": str(cfg.dtype),
        "max_length": cfg.max_length,
        "safety_sample_size": sample_size,
        "safety_seed": cfg.safety_seed,
        "safety_generation_max_new_tokens": cfg.safety_generation_max_new_tokens,
        "quantization_config": quantization_config,
    }


def _extract_prompt(row: Dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    model_input = str(row.get("input", "")).strip()
    if instruction and model_input:
        return f"Instruction: {instruction}\nInput: {model_input}"
    if instruction:
        return instruction
    for key in ("prompt", "question", "instruction", "input", "text", "adversarial"):
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


def _load_model_config(model_path: str) -> Dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _get_quantization_config(model_path: str) -> Dict[str, Any]:
    model_config = _load_model_config(model_path)
    quantization_config = model_config.get("quantization_config", {})
    return quantization_config if isinstance(quantization_config, dict) else {}


def _infer_model_device(model: Any) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            if isinstance(device, str) and device not in {"cpu", "disk"}:
                return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _load_generation_model(model_path: str, device: str, dtype: torch.dtype) -> Tuple[Any, Any, torch.device]:
    quantization_config = _get_quantization_config(model_path)
    is_quantized_model = bool(quantization_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "device_map": "auto" if is_quantized_model else None,
        "trust_remote_code": True,
    }
    model_kwargs["torch_dtype"] = "auto" if is_quantized_model else dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not is_quantized_model:
        model.to(device)
    model.eval()
    return model, tokenizer, _infer_model_device(model)


def _generate_responses(
    model,
    tokenizer,
    model_device: torch.device,
    prompts: List[str],
    max_length: int,
    max_new_tokens: int,
    desc: str,
) -> List[str]:
    responses: List[str] = []
    for prompt in tqdm(prompts, desc=desc):
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
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.inference_mode():
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

        del encoded
        del output
        if model_device.type == "cuda":
            torch.cuda.empty_cache()
    return responses


def _load_xguard(xguard_model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        xguard_model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        xguard_model_path,
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _xguard_response_safety(
    guard_model,
    guard_tokenizer,
    prompt: str,
    response: str,
    max_new_tokens: int = 1,
) -> Dict[str, Any]:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    rendered_query = guard_tokenizer.apply_chat_template(
        messages,
        policy=None,
        reason_first=False,
        tokenize=False,
    )
    model_inputs = guard_tokenizer([rendered_query], return_tensors="pt").to(guard_model.device)

    with torch.no_grad():
        outputs = guard_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    input_length = model_inputs["input_ids"].shape[1]
    output_ids = outputs["sequences"][0][input_length:]
    response_text = guard_tokenizer.decode(output_ids, skip_special_tokens=True)

    generated_tokens = outputs.sequences[:, input_length:]
    scores = torch.stack(outputs.scores, dim=1).softmax(-1)
    scores_topk_value, scores_topk_index = scores.topk(k=10, dim=-1)

    generated_tokens_with_probs: List[List[Dict[str, Dict[str, float | str]]]] = []
    for generated_token, topk_values, topk_indices in zip(generated_tokens, scores_topk_value, scores_topk_index):
        token_prob_list: List[Dict[str, Dict[str, float | str]]] = []
        for token_id, values, indices in zip(generated_token, topk_values, topk_indices):
            token_id = int(token_id.cpu())
            if token_id == guard_tokenizer.pad_token_id:
                continue
            token_scores: Dict[str, Dict[str, float | str]] = {}
            for rank, (value, index) in enumerate(zip(values, indices)):
                prob = float(value.cpu().numpy())
                if rank == 0 or prob > 1e-4:
                    text = guard_tokenizer.decode(index.cpu().numpy())
                    token_scores[text] = {
                        "id": str(int(index.cpu().numpy())),
                        "prob": round(prob, 4),
                    }
            token_prob_list.append(token_scores)
        generated_tokens_with_probs.append(token_prob_list)

    first_step_scores = generated_tokens_with_probs[0][0] if generated_tokens_with_probs and generated_tokens_with_probs[0] else {}
    token_score = {key: value["prob"] for key, value in first_step_scores.items()}
    id2risk = guard_tokenizer.init_kwargs.get("id2risk", {})
    risk_score = {
        id2risk[key]: value["prob"]
        for key, value in first_step_scores.items()
        if key in id2risk
    }
    return {
        "response": response_text,
        "token_score": token_score,
        "risk_score": risk_score,
    }


def _safe_safe_score(risk_score: Dict[str, float]) -> float:
    return float(risk_score.get("Safe-Safe", 0.0))


def _iter_datasets(args: argparse.Namespace, cfg) -> List[tuple[str, str]]:
    if args.dataset_path:
        dataset_path = str(Path(args.dataset_path))
        dataset_name = args.dataset_name or Path(dataset_path).stem
        return [(dataset_name, dataset_path)]
    return list(cfg.safety_dataset_paths.items())


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    sample_size = args.sample_size or cfg.safety_sample_size
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.data_root) / DEFAULT_OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.safety_seed)
    random.seed(cfg.safety_seed)

    model_path = args.model_path
    xguard_model_path = str(Path(cfg.project_root) / XGUARD_RELATIVE_PATH)
    model_outputs_dirname = model_path.replace("/", "__")
    model_output_dir = output_dir / model_outputs_dirname
    model_output_dir.mkdir(parents=True, exist_ok=True)
    result_config = _build_result_config(
        cfg=cfg,
        model_path=model_path,
        xguard_model_path=xguard_model_path,
        sample_size=sample_size,
    )

    generation_model, generation_tokenizer, generation_model_device = _load_generation_model(
        model_path=model_path,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    guard_model, guard_tokenizer = _load_xguard(xguard_model_path=xguard_model_path, device=cfg.device)
    all_safe_safe_scores: List[float] = []

    try:
        for dataset_name, dataset_path in tqdm(_iter_datasets(args, cfg), desc="safety2:datasets"):
            rows = _load_json_dataset(dataset_path)
            sampled_rows = _sample_rows(rows, sample_size, cfg.safety_seed)
            prompts = [_extract_prompt(row) for row in sampled_rows]
            responses = _generate_responses(
                model=generation_model,
                tokenizer=generation_tokenizer,
                model_device=generation_model_device,
                prompts=prompts,
                max_length=cfg.max_length,
                max_new_tokens=cfg.safety_generation_max_new_tokens,
                desc=f"{dataset_name}:generate",
            )

            results: List[Dict[str, Any]] = []
            for row, prompt, response in tqdm(
                zip(sampled_rows, prompts, responses),
                total=len(sampled_rows),
                desc=f"{dataset_name}:xguard",
            ):
                score = _xguard_response_safety(
                    guard_model=guard_model,
                    guard_tokenizer=guard_tokenizer,
                    prompt=prompt,
                    response=response,
                    max_new_tokens=1,
                )
                results.append(
                    {
                        "prompt": prompt,
                        "meta": row,
                        "model_output": response,
                        "xguard_risk_score": score["risk_score"],
                    }
                )

            safe_safe_scores = [_safe_safe_score(item["xguard_risk_score"]) for item in results]
            average_safe_safe_score = (
                sum(safe_safe_scores) / len(safe_safe_scores) if safe_safe_scores else 0.0
            )
            all_safe_safe_scores.extend(safe_safe_scores)

            payload = {
                "config": result_config,
                "model_path": model_path,
                "xguard_model_path": xguard_model_path,
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "sample_count": len(results),
                "average_safe_safe_score": average_safe_safe_score,
                "results": results,
            }
            output_path = model_output_dir / f"{dataset_name}.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            print(f"Saved XGuard safety results to: {output_path}")
            print(f"[{dataset_name}] average Safe-Safe score: {average_safe_safe_score:.6f}")

            del rows
            del sampled_rows
            del prompts
            del responses
            del results
            gc.collect()
            if cfg.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del generation_model
        del guard_model
        gc.collect()
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_average_safe_safe_score = (
        sum(all_safe_safe_scores) / len(all_safe_safe_scores) if all_safe_safe_scores else 0.0
    )
    print(f"[overall] average Safe-Safe score: {overall_average_safe_safe_score:.6f}")


if __name__ == "__main__":
    main()
