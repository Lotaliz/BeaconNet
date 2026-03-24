import argparse
import json
from pathlib import Path

from config import load_config
from src.activation.sae_data import load_safety2_examples, save_examples_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SAE training examples from safety2 outputs.")
    parser.add_argument("--source-dir", default=None, help="Directory containing safety2 dataset JSON files.")
    parser.add_argument("--output", default=None, help="Output JSONL path for prepared SAE examples.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional per-dataset cap.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    sae_cfg = cfg.sae

    source_dir = args.source_dir or sae_cfg.source_safety_result_dir
    output_path = Path(args.output or sae_cfg.example_manifest_path)
    max_samples = sae_cfg.max_samples_per_dataset if args.max_samples is None else args.max_samples

    examples = load_safety2_examples(source_dir=source_dir, max_samples_per_dataset=max_samples)
    count = save_examples_jsonl(examples, output_path)

    dataset_counts = {}
    for example in examples:
        dataset_counts[example.dataset_name] = dataset_counts.get(example.dataset_name, 0) + 1

    summary = {
        "source_dir": source_dir,
        "output_path": str(output_path),
        "sample_count": count,
        "dataset_counts": dataset_counts,
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Prepared SAE example manifest: {output_path}")
    print(f"Prepared SAE summary: {summary_path}")
    print(f"Sample count: {count}")


if __name__ == "__main__":
    main()
