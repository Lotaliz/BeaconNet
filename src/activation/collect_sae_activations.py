import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from config import config_to_dict, load_config
from src.activation.common import build_prompt_inputs, load_model, load_tokenizer, move_to_device, sanitize_filename
from src.activation.hooks import SampleActivationCollector
from src.activation.sae_data import load_examples_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-sample activations for SAE training.")
    parser.add_argument("-m", "--model-path", default=None, help="Base model path override.")
    parser.add_argument("-a", "--adapter-path", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--manifest", default=None, help="Prepared SAE example JSONL path.")
    parser.add_argument("--output-dir", default=None, help="Output directory for activation tensors.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit over the prepared manifest.")
    return parser.parse_args()


def _save_metadata(rows: List[Dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    sae_cfg = cfg.sae

    model_path = args.model_path or sae_cfg.model_path
    adapter_path = args.adapter_path if args.adapter_path is not None else (
        sae_cfg.lora_adapter_path if sae_cfg.use_lora_adapter else ""
    )
    manifest_path = Path(args.manifest or sae_cfg.example_manifest_path)
    output_dir = Path(args.output_dir or sae_cfg.activation_dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples_jsonl(manifest_path)
    if args.max_samples is not None and args.max_samples > 0:
        examples = examples[: args.max_samples]

    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path=model_path, adapter_path=adapter_path, dtype=sae_cfg.torch_dtype, device=sae_cfg.device)

    collector = SampleActivationCollector(
        model=model,
        module_names=sae_cfg.capture_module_names,
        token_aggregation=sae_cfg.token_aggregation,
    )
    collector.register()

    per_layer_vectors: Dict[str, List[torch.Tensor]] = {name: [] for name in sae_cfg.capture_module_names}
    metadata_rows: List[Dict[str, object]] = []
    targets_safe_safe: List[float] = []
    targets_unsafe: List[float] = []

    try:
        for example in tqdm(examples, desc="sae:collect"):
            encoded, prompt_last_token_index = build_prompt_inputs(
                tokenizer=tokenizer,
                prompt=example.prompt,
                max_length=sae_cfg.max_length,
            )
            batch = move_to_device(encoded, sae_cfg.device)
            collector.clear()
            collector.set_token_span(prompt_last_token_index, prompt_last_token_index + 1)

            with torch.no_grad():
                _ = model(**batch, use_cache=False)

            sample_vectors = collector.sample_vectors()
            for layer_name in sae_cfg.capture_module_names:
                vector = sample_vectors[layer_name].to(dtype=torch.float16)
                per_layer_vectors[layer_name].append(vector)

            metadata_rows.append(
                {
                    "sample_id": example.sample_id,
                    "dataset_name": example.dataset_name,
                    "prompt": example.prompt,
                    "response": example.response,
                    "safe_safe_score": example.safe_safe_score,
                    "unsafe_score": example.unsafe_score,
                    "source_model_path": example.source_model_path,
                    "activation_target": "prompt_last_token",
                    "prompt_last_token_index": int(prompt_last_token_index),
                }
            )
            targets_safe_safe.append(float(example.safe_safe_score))
            targets_unsafe.append(float(example.unsafe_score))
    finally:
        collector.remove()
        del model
        if sae_cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer_files: Dict[str, str] = {}
    for layer_name, vectors in per_layer_vectors.items():
        if not vectors:
            continue
        tensor = torch.stack(vectors, dim=0)
        filename = sanitize_filename(layer_name) + ".pt"
        output_path = output_dir / filename
        torch.save(
            {
                "layer_name": layer_name,
                "activations": tensor,
                "dtype": str(tensor.dtype),
                "sample_count": int(tensor.shape[0]),
                "feature_dim": int(tensor.shape[1]),
            },
            output_path,
        )
        layer_files[layer_name] = str(output_path)

    metadata_path = output_dir / "metadata.jsonl"
    targets_path = output_dir / "targets.pt"
    manifest_output_path = output_dir / "activation_manifest.json"

    _save_metadata(metadata_rows, metadata_path)
    torch.save(
        {
            "safe_safe_score": torch.tensor(targets_safe_safe, dtype=torch.float32),
            "unsafe_score": torch.tensor(targets_unsafe, dtype=torch.float32),
        },
        targets_path,
    )

    payload = {
        "config": config_to_dict(cfg),
        "model_path": model_path,
        "adapter_path": adapter_path,
        "example_manifest_path": str(manifest_path),
        "activation_dataset_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "targets_path": str(targets_path),
        "layer_files": layer_files,
        "sample_count": len(metadata_rows),
        "activation_target": "prompt_last_token",
    }
    with manifest_output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved SAE activation dataset to: {output_dir}")
    print(f"Saved SAE activation manifest to: {manifest_output_path}")


if __name__ == "__main__":
    main()
