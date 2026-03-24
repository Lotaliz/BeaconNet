from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class SafetyActivationExample:
    sample_id: str
    dataset_name: str
    prompt: str
    response: str
    safe_safe_score: float
    unsafe_score: float
    source_model_path: str
    source_file: str
    metadata: Dict[str, Any]


def _load_json(path: str | Path) -> Dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Safety2 result file not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {target}")
    return data


def load_safety2_examples(source_dir: str | Path, max_samples_per_dataset: int = 0) -> List[SafetyActivationExample]:
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Safety2 source directory not found: {root}")

    examples: List[SafetyActivationExample] = []
    for path in sorted(root.glob("*.json")):
        payload = _load_json(path)
        dataset_name = str(payload.get("dataset_name") or path.stem)
        model_path = str(payload.get("model_path", ""))
        results = payload.get("results", [])
        if not isinstance(results, list):
            continue
        if max_samples_per_dataset > 0:
            results = results[:max_samples_per_dataset]
        for index, row in enumerate(results):
            if not isinstance(row, dict):
                continue
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("model_output", "")).strip()
            risk_score = row.get("xguard_risk_score", {})
            if not prompt or not response or not isinstance(risk_score, dict):
                continue
            safe_safe_score = float(risk_score.get("Safe-Safe", 0.0))
            examples.append(
                SafetyActivationExample(
                    sample_id=f"{dataset_name}-{index:05d}",
                    dataset_name=dataset_name,
                    prompt=prompt,
                    response=response,
                    safe_safe_score=safe_safe_score,
                    unsafe_score=1.0 - safe_safe_score,
                    source_model_path=model_path,
                    source_file=str(path),
                    metadata=row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {},
                )
            )
    if not examples:
        raise ValueError(f"No usable samples found under {root}")
    return examples


def save_examples_jsonl(examples: Iterable[SafetyActivationExample], output_path: str | Path) -> int:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_examples_jsonl(path: str | Path) -> List[SafetyActivationExample]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Example manifest not found: {target}")
    examples: List[SafetyActivationExample] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(SafetyActivationExample(**row))
    if not examples:
        raise ValueError(f"No examples found in {target}")
    return examples
