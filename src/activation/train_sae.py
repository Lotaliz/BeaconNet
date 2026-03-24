import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import load_config


@dataclass
class LayerTrainSummary:
    layer_name: str
    input_dim: int
    hidden_dim: int
    train_size: int
    val_size: int
    best_val_loss: float
    best_val_recon_loss: float
    best_val_l1_loss: float
    top_positive_features: List[Dict[str, float]]
    top_negative_features: List[Dict[str, float]]


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        codes = F.relu(self.encoder(inputs))
        recon = self.decoder(codes)
        return recon, codes

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(inputs))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sparse autoencoders on collected activation tensors.")
    parser.add_argument("--dataset-dir", default=None, help="Activation dataset directory from collect_sae_activations.")
    parser.add_argument("--output-dir", default=None, help="Directory to store trained SAE checkpoints.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--layer", action="append", default=None, help="Optional layer name filter. Can be passed multiple times.")
    return parser.parse_args()


def _pearson_corr(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    features = features.float()
    targets = targets.float()
    features_centered = features - features.mean(dim=0, keepdim=True)
    targets_centered = targets - targets.mean()
    numerator = (features_centered * targets_centered.unsqueeze(1)).sum(dim=0)
    denominator = torch.sqrt((features_centered.pow(2).sum(dim=0) * targets_centered.pow(2).sum()).clamp_min(1e-8))
    return numerator / denominator


def _split_indices(total_size: int, val_fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = int(total_size * val_fraction)
    val_size = min(max(val_size, 1), max(total_size - 1, 1)) if total_size > 1 else 0
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    if not train_indices:
        train_indices = indices
        val_indices = []
    return torch.tensor(train_indices, dtype=torch.long), torch.tensor(val_indices, dtype=torch.long)


def _train_one_layer(
    layer_name: str,
    activation_path: Path,
    safe_safe_scores: torch.Tensor,
    cfg,
    output_root: Path,
) -> LayerTrainSummary:
    payload = torch.load(activation_path, map_location="cpu")
    activations = payload["activations"].float()
    input_dim = int(activations.shape[1])
    hidden_dim = max(1, int(math.ceil(input_dim * cfg.feature_multiplier)))

    train_indices, val_indices = _split_indices(len(activations), cfg.val_fraction, cfg.seed)
    train_tensor = activations[train_indices]
    val_tensor = activations[val_indices] if len(val_indices) > 0 else activations[train_indices[:1]]

    mean = train_tensor.mean(dim=0, keepdim=True)
    std = train_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_tensor = (train_tensor - mean) / std
    val_tensor = (val_tensor - mean) / std
    full_tensor = (activations - mean) / std

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=min(cfg.batch_size, len(train_tensor)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=min(cfg.batch_size, len(val_tensor)),
        shuffle=False,
    )

    model = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    best_val_recon = float("inf")
    best_val_l1 = float("inf")

    for epoch in tqdm(range(cfg.num_epochs), desc=f"sae:train:{layer_name}"):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(cfg.device)
            recon, codes = model(batch)
            recon_loss = F.mse_loss(recon, batch)
            l1_loss = codes.abs().mean()
            loss = recon_loss + cfg.l1_coefficient * l1_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_total = 0.0
        val_recon_total = 0.0
        val_l1_total = 0.0
        val_count = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(cfg.device)
                recon, codes = model(batch)
                recon_loss = F.mse_loss(recon, batch)
                l1_loss = codes.abs().mean()
                loss = recon_loss + cfg.l1_coefficient * l1_loss
                batch_size = int(batch.shape[0])
                val_loss_total += float(loss.item()) * batch_size
                val_recon_total += float(recon_loss.item()) * batch_size
                val_l1_total += float(l1_loss.item()) * batch_size
                val_count += batch_size

        avg_val_loss = val_loss_total / max(val_count, 1)
        avg_val_recon = val_recon_total / max(val_count, 1)
        avg_val_l1 = val_l1_total / max(val_count, 1)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_recon = avg_val_recon
            best_val_l1 = avg_val_l1
            best_state = {
                "model": model.state_dict(),
                "mean": mean,
                "std": std,
                "epoch": epoch,
            }

    if best_state is None:
        raise RuntimeError(f"Training failed to produce a checkpoint for {layer_name}")

    model.load_state_dict(best_state["model"])
    model.eval()
    with torch.no_grad():
        codes = model.encode(full_tensor.to(cfg.device)).cpu()
    correlations = _pearson_corr(codes, safe_safe_scores)
    top_positive_values, top_positive_indices = torch.topk(correlations, k=min(10, correlations.numel()))
    top_negative_values, top_negative_indices = torch.topk(-correlations, k=min(10, correlations.numel()))

    top_positive = [
        {"feature_index": int(index.item()), "correlation": float(value.item())}
        for value, index in zip(top_positive_values, top_positive_indices)
    ]
    top_negative = [
        {"feature_index": int(index.item()), "correlation": float(-value.item())}
        for value, index in zip(top_negative_values, top_negative_indices)
    ]

    layer_dir = output_root / activation_path.stem
    layer_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = layer_dir / "sae.pt"
    metrics_path = layer_dir / "metrics.json"

    torch.save(
        {
            "layer_name": layer_name,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "config": {
                "device": cfg.device,
                "feature_multiplier": cfg.feature_multiplier,
                "l1_coefficient": cfg.l1_coefficient,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size,
                "num_epochs": cfg.num_epochs,
                "val_fraction": cfg.val_fraction,
                "seed": cfg.seed,
            },
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
        },
        checkpoint_path,
    )

    summary = LayerTrainSummary(
        layer_name=layer_name,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        train_size=int(len(train_indices)),
        val_size=int(len(val_indices)),
        best_val_loss=best_val_loss,
        best_val_recon_loss=best_val_recon,
        best_val_l1_loss=best_val_l1,
        top_positive_features=top_positive,
        top_negative_features=top_negative,
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    sae_cfg = cfg.sae
    dataset_dir = Path(args.dataset_dir or sae_cfg.activation_dataset_dir)
    output_dir = Path(args.output_dir or sae_cfg.training_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_dir / "activation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Activation manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    targets = torch.load(dataset_dir / "targets.pt", map_location="cpu")
    safe_safe_scores = targets["safe_safe_score"].float()

    class TrainConfig:
        device = sae_cfg.device
        feature_multiplier = sae_cfg.feature_multiplier
        l1_coefficient = sae_cfg.l1_coefficient
        learning_rate = sae_cfg.learning_rate
        weight_decay = sae_cfg.weight_decay
        batch_size = sae_cfg.batch_size
        num_epochs = args.epochs or sae_cfg.num_epochs
        val_fraction = sae_cfg.val_fraction
        seed = sae_cfg.seed

    layer_files = manifest.get("layer_files", {})
    if args.layer:
        requested = set(args.layer)
        layer_files = {name: path for name, path in layer_files.items() if name in requested}
    if not layer_files:
        raise ValueError("No layer activation files selected for SAE training.")

    summaries: List[LayerTrainSummary] = []
    for layer_name, path in layer_files.items():
        summary = _train_one_layer(
            layer_name=layer_name,
            activation_path=Path(path),
            safe_safe_scores=safe_safe_scores,
            cfg=TrainConfig,
            output_root=output_dir,
        )
        summaries.append(summary)
        print(
            f"[sae] layer={layer_name} | best_val_loss={summary.best_val_loss:.6f} | "
            f"top_positive_feature={summary.top_positive_features[0] if summary.top_positive_features else 'NA'}"
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(item) for item in summaries], handle, ensure_ascii=False, indent=2)
    print(f"Saved SAE summaries to: {summary_path}")


if __name__ == "__main__":
    main()
