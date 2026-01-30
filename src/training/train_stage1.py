import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs import load_config
from src.inst.apply_scaler import ScalerApplier
from src.model.layer_encoder import LayerEmbeddingEncoder
from src.model.hypernet import HypernetScaler


class JsonlDataset(Dataset):
    def __init__(self, path: str, fallback: Optional[List[Dict[str, Any]]] = None) -> None:
        self.samples: List[Dict[str, Any]] = []
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.samples.append(json.loads(line))
        elif fallback is not None:
            self.samples = fallback
        else:
            raise FileNotFoundError(f"Dataset path not found: {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class SafetyHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (hidden_states * mask).sum(dim=1) / denom
        return self.proj(pooled).squeeze(-1)


class DecoderHead(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _infer_num_layers_and_hidden_size(model) -> Tuple[int, int]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no config.")
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if num_layers is None or hidden_size is None:
        raise ValueError("Failed to infer num_layers/hidden_size from model.config.")
    return int(num_layers), int(hidden_size)


def _format_sample(sample: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    if "text" in sample:
        text = sample["text"]
        label = sample.get("safety_label", None)
        return text, label

    prompt = sample.get("prompt", "")
    if "response_0" in sample and "response_1" in sample:
        chosen_id = sample.get("safer_response_id", sample.get("better_response_id", 0))
        try:
            chosen_id = int(chosen_id)
        except (TypeError, ValueError):
            chosen_id = 0
        response = sample.get(f"response_{chosen_id}", sample.get("response_0", ""))
        safe_key = f"is_response_{chosen_id}_safe"
        label = sample.get(safe_key, None)
        if label is not None:
            label = 1.0 if bool(label) else 0.0
    else:
        response = sample.get("response", "")
        label = sample.get("safety_label", None)

    text = f"{prompt}\n{response}".strip()
    return text, label


def _collect_layer_weights(model: nn.Module, allowed_components: List[str]) -> Dict[str, Dict[int, torch.Tensor]]:
    layer_pattern = re.compile(r"(?:^|.*\.)layers\.(\d+)\.(.+)$")
    allowed = set(allowed_components)
    weights: Dict[str, Dict[int, torch.Tensor]] = {name: {} for name in allowed_components}
    for name, module in model.named_modules():
        m = layer_pattern.match(name)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        component = m.group(2)
        if component not in allowed:
            continue
        if hasattr(module, "weight"):
            weights[component][layer_idx] = module.weight
    return weights


def _svd_summary(weight: torch.Tensor, rank: int) -> torch.Tensor:
    w = weight.detach().float().cpu()
    try:
        s = torch.linalg.svdvals(w)
    except RuntimeError:
        s = torch.linalg.svdvals(w.to(torch.float64)).float()

    if s.numel() < rank:
        pad = torch.zeros(rank - s.numel(), dtype=s.dtype)
        s = torch.cat([s, pad], dim=0)

    s = torch.clamp(s[:rank], min=1e-6, max=1e6)
    log_s = torch.log(s)
    frob = torch.linalg.norm(w.to(torch.float64))
    mean_abs = w.abs().to(torch.float64).mean()
    stats_raw = torch.tensor([frob, mean_abs], dtype=torch.float64)
    stats = torch.log1p(torch.clamp(stats_raw, min=0.0, max=1e6)).to(dtype=log_s.dtype)
    out = torch.cat([log_s, stats], dim=0)
    return torch.nan_to_num(out, nan=0.0, posinf=20.0, neginf=-20.0)


def _build_layer_summaries(
    model: nn.Module,
    num_layers: int,
    rank: int,
    components: List[str],
) -> Dict[str, torch.Tensor]:
    weights = _collect_layer_weights(model, components)
    summary_dim = rank + 2
    summaries = {
        name: torch.zeros((num_layers, summary_dim), dtype=torch.float32) for name in components
    }
    for component, layer_map in weights.items():
        for layer_idx, weight in layer_map.items():
            if layer_idx < 0 or layer_idx >= num_layers:
                continue
            summaries[component][layer_idx] = _svd_summary(weight, rank)
    return summaries


def _load_or_build_summaries(
    *,
    model: nn.Module,
    num_layers: int,
    rank: int,
    cache_path: str,
) -> Tuple[List[str], Dict[str, torch.Tensor], int]:
    components = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    summary_dim = rank + 2
    if cache_path and os.path.isfile(cache_path):
        data = torch.load(cache_path, map_location="cpu")
        if isinstance(data, dict) and "components" in data and "summaries" in data:
            cached_components = data["components"]
            summaries = data["summaries"]
            cached_dim = data.get("summary_dim", summary_dim)
            if (
                cached_components == components
                and cached_dim == summary_dim
                and isinstance(summaries, dict)
                and all(k in summaries for k in components)
                and all(summaries[k].shape == (num_layers, summary_dim) for k in components)
            ):
                if all(torch.isfinite(summaries[k]).all() for k in components):
                    return components, summaries, summary_dim

    summaries = _build_layer_summaries(model, num_layers, rank, components)
    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save(
            {"components": components, "summaries": summaries, "summary_dim": summary_dim},
            cache_path,
        )
    return components, summaries, summary_dim


def main() -> None:
    cfg = load_config()
    torch.manual_seed(16494)

    tok = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[stage1] model_path={cfg.model_path} device={cfg.device} dtype={cfg.dtype}")
    print(f"[stage1] dataset_path={cfg.dataset_path} save_dir={cfg.save_dir}")

    student = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        dtype=cfg.dtype,
        device_map=cfg.device,
    )
    for p in student.parameters():
        p.requires_grad = False

    num_layers, student_hidden = _infer_num_layers_and_hidden_size(student)

    components, summaries, summary_dim = _load_or_build_summaries(
        model=student,
        num_layers=num_layers,
        rank=cfg.summary_rank,
        cache_path=cfg.summary_cache_path,
    )
    print(f"[stage1] summary_cache={cfg.summary_cache_path} summary_dim={summary_dim}")
    print(f"[stage1] components={components}")
    summary_tensors = {
        name: summaries[name].to(device=cfg.device, dtype=torch.float32) for name in components
    }

    encoder = LayerEmbeddingEncoder(
        num_layers=num_layers,
        summary_dim=summary_dim,
        component_names=components,
        d_layer_emb=cfg.d_layer_emb,
        d_hidden=cfg.d_layer_emb,
    ).to(device=cfg.device, dtype=torch.float32)

    hypernet = HypernetScaler(
        num_layers=num_layers,
        hidden_size=student_hidden,
        d_layer_emb=cfg.d_layer_emb,
        d_guidance=cfg.d_attn,
        d_model=cfg.d_layer_emb,
        n_heads=8,
    ).to(device=cfg.device, dtype=torch.float32)

    decoder = DecoderHead(cfg.d_layer_emb, len(components) * summary_dim).to(
        device=cfg.device, dtype=torch.float32
    )

    safety_head = SafetyHead(hidden_size=student_hidden).to(device=cfg.device, dtype=torch.float32)

    applier = ScalerApplier(strength_o=1.0, strength_d=1.0, clamp_min=0.5, clamp_max=1.5)
    applier.attach_llama_o_proj_and_down_proj(student)
    print(f"[stage1] num_layers={num_layers} hidden_size={student_hidden}")

    fallback = [
        {"text": "User: Explain why unsafe requests should be refused.\nAssistant:", "safety_label": 1.0},
        {"text": "User: Summarize the benefits of seatbelts.\nAssistant:", "safety_label": 1.0},
    ]
    dataset = JsonlDataset(cfg.dataset_path, fallback=fallback)
    print(f"[stage1] dataset_size={len(dataset)} batch_size={cfg.batch_size}")

    def collate(batch: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        texts: List[str] = []
        safety_labels: List[Optional[float]] = []
        for sample in batch:
            text, label = _format_sample(sample)
            texts.append(text)
            safety_labels.append(label)

        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels

        if any(label is not None for label in safety_labels):
            lab = [float(label) if label is not None else 0.0 for label in safety_labels]
            safety = torch.tensor(lab, dtype=torch.float32)
        else:
            safety = None
        return enc, safety

    steps_per_epoch = max(1, (len(dataset) + cfg.batch_size - 1) // cfg.batch_size)
    total_steps = max(cfg.num_steps, steps_per_epoch * cfg.stage1_epochs)
    print(
        f"[stage1] steps_per_epoch={steps_per_epoch} num_epochs={cfg.stage1_epochs} total_steps={total_steps}"
    )

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    data_iter: Iterable[Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]] = iter(loader)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        + list(hypernet.parameters())
        + list(decoder.parameters())
        + list(safety_head.parameters()),
        lr=cfg.lr,
    )
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in hypernet.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in safety_head.parameters() if p.requires_grad)
    print(f"[stage1] trainable_params={trainable} lr={cfg.lr}")

    student.train()
    encoder.train()
    hypernet.train()
    decoder.train()
    safety_head.train()

    reg_weight = float(os.environ.get("REG_LOSS_WEIGHT", "1.0"))
    print(f"[stage1] summary_weight={cfg.summary_loss_weight} reg_weight={reg_weight}")

    for step in tqdm(range(total_steps), desc="Training Stage 1"):
        try:
            batch, safety = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch, safety = next(data_iter)

        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        if safety is not None:
            safety = safety.to(cfg.device)

        B = batch["input_ids"].size(0)
        summaries_b = {name: t.unsqueeze(0).expand(B, -1, -1) for name, t in summary_tensors.items()}

        summaries_b_model = {name: t.to(dtype=torch.float32) for name, t in summaries_b.items()}
        layer_emb, _summary_pred = encoder(summaries_b_model)
        z0 = torch.zeros((B, cfg.d_attn), device=cfg.device, dtype=torch.float32)
        hn_out = hypernet(guidance=z0, layer_emb=layer_emb)
        scalers = {}
        for key, value in hn_out.scalers.items():
            clean = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
            scalers[key] = torch.clamp(clean, min=-0.5, max=0.5)
        applier.set_scalers(scalers)

        need_safety = safety is not None and cfg.safety_weight > 0.0
        outputs = student(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            output_hidden_states=need_safety,
        )

        loss_safe = outputs.loss.float()
        safety_loss = None
        if need_safety:
            hidden = outputs.hidden_states[-1].float()
            logits = safety_head(hidden, batch["attention_mask"]).float()
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            safety_loss = nn.functional.binary_cross_entropy_with_logits(logits, safety.float())
            loss_safe = loss_safe + cfg.safety_weight * safety_loss

        target_summary = torch.cat([summaries_b[name] for name in components], dim=-1)
        pred_summary = decoder(layer_emb)
        loss_e = nn.functional.mse_loss(pred_summary, target_summary)

        reg_terms = []
        for key in ("o_proj", "down_proj"):
            scaler = scalers[key].float()
            factor = 1.0 + scaler
            reg_terms.append((factor - 1.0).pow(2).mean())
        loss_reg = torch.stack(reg_terms).mean()

        loss = loss_safe + cfg.summary_loss_weight * loss_e + reg_weight * loss_reg

        if not torch.isfinite(loss):
            def _stat(t: torch.Tensor, name: str) -> None:
                t = t.detach().float()
                print(
                    f"[nan-debug] {name}: finite={torch.isfinite(t).all().item()} "
                    f"min={t.min().item():.3e} max={t.max().item():.3e} mean={t.mean().item():.3e}"
                )

            print(f"[nan-debug] step={step+1}")
            _stat(loss_safe, "loss_safe")
            _stat(loss_e, "loss_e")
            _stat(loss_reg, "loss_reg")
            if safety_loss is not None:
                _stat(safety_loss, "safety_loss")
                _stat(logits, "logits")
            _stat(target_summary, "target_summary")
            _stat(pred_summary, "pred_summary")
            _stat(hn_out.scalers["o_proj"], "scaler_o")
            _stat(hn_out.scalers["down_proj"], "scaler_d")
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(hypernet.parameters()) + list(decoder.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        if (step + 1) % cfg.logging_steps == 0:
            msg = (
                f"step={step+1} loss={loss.item():.4f} safe={loss_safe.item():.4f} "
                f"L_e={loss_e.item():.4f} L_reg={loss_reg.item():.4f}"
            )
            if safety_loss is not None:
                msg += f" safety={safety_loss.item():.4f}"
            print(msg)

        log_path = os.path.join(cfg.save_dir, "stage1_metrics.jsonl")
        os.makedirs(cfg.save_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            record = {
                "step": step + 1,
                "loss": float(loss.item()),
                "loss_safe": float(loss_safe.item()),
                "loss_e": float(loss_e.item()),
                "loss_reg": float(loss_reg.item()),
                "safety_loss": float(safety_loss.item()) if safety_loss is not None else None,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    ckpt = {
        "encoder": encoder.state_dict(),
        "hypernet": hypernet.state_dict(),
        "decoder": decoder.state_dict(),
        "safety_head": safety_head.state_dict(),
        "config": {
            "summary_rank": cfg.summary_rank,
            "summary_dim": summary_dim,
            "components": components,
        },
    }
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save(ckpt, cfg.stage1_ckpt_path)

    applier.remove()


if __name__ == "__main__":
    main()
