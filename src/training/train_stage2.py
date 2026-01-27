from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs import load_config
from src.inst.apply_scaler import ScalerApplier
from src.inst.hooks import ActivationHookManager
from src.model.layer_encoder import LayerEmbeddingEncoder
from src.model.hypernet import HypernetScaler
from src.utils.activation_aggregator import ActivationAggregator


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


class AttentionModule(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        teacher_layers: int,
        student_hidden: int,
        teacher_hidden: int,
        d_attn: int,
        num_basis: Optional[int] = None,
        train_basis: bool = True,
        lambda_pos: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.teacher_layers = int(teacher_layers)
        self.d_attn = int(d_attn)
        self.num_basis = int(num_basis) if num_basis is not None else self.teacher_layers
        self.lambda_pos = float(lambda_pos)

        self.q_proj = nn.Sequential(
            nn.Linear(student_hidden * 2, d_attn),
            nn.GELU(),
            nn.LayerNorm(d_attn),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(teacher_hidden * 2, d_attn),
            nn.GELU(),
            nn.LayerNorm(d_attn),
        )

        basis = torch.zeros(self.teacher_layers, self.d_attn)
        nn.init.normal_(basis, mean=0.0, std=0.02)
        if train_basis:
            self.basis = nn.Parameter(basis)
        else:
            self.register_buffer("basis", basis)

        bias = self._build_depth_bias(self.num_layers, self.teacher_layers, self.lambda_pos)
        self.register_buffer("depth_bias", bias)

    def forward(
        self,
        student_o: torch.Tensor,
        student_d: torch.Tensor,
        teacher_o: torch.Tensor,
        teacher_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if student_o.shape != student_d.shape or teacher_o.shape != teacher_d.shape:
            raise ValueError("student_o/d and teacher_o/d must have the same shape.")
        if student_o.size(1) != self.num_layers:
            raise ValueError("student activations must have num_layers on dim=1.")
        if teacher_o.size(1) != self.teacher_layers:
            raise ValueError("teacher activations must have teacher_layers on dim=1.")

        s_feat = torch.cat([student_o, student_d], dim=-1)
        t_feat = torch.cat([teacher_o, teacher_d], dim=-1)

        q = self.q_proj(s_feat)
        k = self.k_proj(t_feat)

        scale = 1.0 / math.sqrt(self.d_attn)
        scores = torch.einsum("bld,bmd->blm", q, k) * scale
        scores = scores + self.depth_bias
        alpha = torch.softmax(scores, dim=-1)

        z = torch.einsum("blm,md->bld", alpha, self.basis)
        return z, alpha

    @staticmethod
    def _build_depth_bias(num_layers: int, teacher_layers: int, lambda_pos: float) -> torch.Tensor:
        l_idx = torch.linspace(0.0, 1.0, steps=num_layers)
        t_idx = torch.linspace(0.0, 1.0, steps=teacher_layers)
        bias = -lambda_pos * (l_idx[:, None] - t_idx[None, :]).abs()
        return bias


def _infer_num_layers_and_hidden_size(model) -> Tuple[int, int]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no config.")
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if num_layers is None or hidden_size is None:
        raise ValueError("Failed to infer num_layers/hidden_size from model.config.")
    return int(num_layers), int(hidden_size)


def _format_sample(sample: Dict[str, Any]) -> Tuple[str, str, Optional[float]]:
    if "text" in sample:
        text = sample["text"]
        label = sample.get("safety_label", None)
        return text, "", label

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
    return text, prompt, label


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
                and all(torch.isfinite(summaries[k]).all() for k in components)
            ):
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


def _load_stage1_checkpoint(
    path: str,
    encoder: LayerEmbeddingEncoder,
    hypernet: HypernetScaler,
    safety_head: Optional[SafetyHead] = None,
) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stage1 checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    if "hypernet" in ckpt:
        hypernet.load_state_dict(ckpt["hypernet"])
    if safety_head is not None and "safety_head" in ckpt:
        safety_head.load_state_dict(ckpt["safety_head"])
    return ckpt


def main() -> None:
    cfg = load_config()
    torch.manual_seed(16494)

    tok = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[stage2] model_path={cfg.model_path} device={cfg.device} dtype={cfg.dtype}")
    print(f"[stage2] dataset_path={cfg.dataset_path} save_dir={cfg.save_dir}")

    student = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        dtype=cfg.dtype,
        device_map=cfg.device,
    )
    for p in student.parameters():
        p.requires_grad = False

    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_path,
        dtype=cfg.dtype,
        device_map=cfg.device,
    )
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    num_layers, student_hidden = _infer_num_layers_and_hidden_size(student)
    teacher_layers, teacher_hidden = _infer_num_layers_and_hidden_size(teacher)
    if teacher_layers <= 0 or num_layers <= 0:
        raise ValueError("Failed to infer valid layer counts for student/teacher.")

    components, summaries, summary_dim = _load_or_build_summaries(
        model=student,
        num_layers=num_layers,
        rank=cfg.summary_rank,
        cache_path=cfg.summary_cache_path,
    )
    print(f"[stage2] summary_cache={cfg.summary_cache_path} summary_dim={summary_dim}")
    print(f"[stage2] components={components}")

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

    attn_module = AttentionModule(
        num_layers=num_layers,
        teacher_layers=teacher_layers,
        student_hidden=student_hidden,
        teacher_hidden=teacher_hidden,
        d_attn=cfg.d_attn,
        num_basis=teacher_layers,
        train_basis=True,
        lambda_pos=cfg.lambda_pos,
    ).to(device=cfg.device, dtype=torch.float32)

    safety_head = SafetyHead(hidden_size=student_hidden).to(device=cfg.device, dtype=cfg.dtype)
    for p in safety_head.parameters():
        p.requires_grad = False

    ckpt_path = os.environ.get("STAGE1_CKPT", os.path.join(cfg.save_dir, "stage1.pt"))
    ckpt = _load_stage1_checkpoint(ckpt_path, encoder, hypernet, safety_head)
    print(f"[stage2] loaded_stage1={ckpt_path}")
    if isinstance(ckpt, dict) and "config" in ckpt and "components" in ckpt["config"]:
        ckpt_components = ckpt["config"]["components"]
        if ckpt_components != components:
            print("[stage2] warning: stage1 components do not match current components list.")

    tune_encoder = bool(int(os.environ.get("TUNE_ENCODER", "0")))
    if not tune_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    hook_teacher = ActivationHookManager(detach=True, to_cpu=False, keep_sequence_dim=False)
    hook_student = ActivationHookManager(detach=True, to_cpu=False, keep_sequence_dim=False)
    hook_teacher.register_llama_o_proj_and_down_proj(teacher)
    hook_student.register_llama_o_proj_and_down_proj(student)

    applier = ScalerApplier(strength_o=1.0, strength_d=1.0, clamp_min=0.5, clamp_max=1.5)
    applier.attach_llama_o_proj_and_down_proj(student)
    print(f"[stage2] num_layers={num_layers} hidden_size={student_hidden}")

    fallback = [
        {"text": "User: Explain why unsafe requests should be refused.\nAssistant:", "safety_label": 1.0},
        {"text": "User: Summarize the benefits of seatbelts.\nAssistant:", "safety_label": 1.0},
    ]
    dataset = JsonlDataset(cfg.dataset_path, fallback=fallback)
    print(f"[stage2] dataset_size={len(dataset)} batch_size={cfg.batch_size}")

    def collate(batch: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        texts: List[str] = []
        prompts: List[str] = []
        safety_labels: List[Optional[float]] = []
        for sample in batch:
            text, prompt, label = _format_sample(sample)
            texts.append(text)
            prompts.append(prompt)
            safety_labels.append(label)

        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        prompt_enc = tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
        labels = enc["input_ids"].clone()
        for i, plen in enumerate(prompt_lens):
            end = int(plen.item())
            if end > 0:
                labels[i, :end] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        enc["prompt_lens"] = prompt_lens

        if any(label is not None for label in safety_labels):
            lab = [float(label) if label is not None else 0.0 for label in safety_labels]
            safety = torch.tensor(lab, dtype=torch.float32)
        else:
            safety = None
        return enc, safety

    steps_per_epoch = max(1, (len(dataset) + cfg.batch_size - 1) // cfg.batch_size)
    total_steps = max(cfg.num_steps, steps_per_epoch * cfg.stage2_epochs)
    print(
        f"[stage2] steps_per_epoch={steps_per_epoch} num_epochs={cfg.stage2_epochs} total_steps={total_steps}"
    )

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    data_iter: Iterable[Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]] = iter(loader)

    train_params: List[nn.Parameter] = list(attn_module.parameters()) + list(hypernet.parameters())
    if tune_encoder:
        train_params += list(encoder.parameters())
    optimizer = torch.optim.AdamW(train_params, lr=cfg.lr)

    trainable = sum(p.numel() for p in train_params if p.requires_grad)
    print(f"[stage2] trainable_params={trainable} lr={cfg.lr} tune_encoder={tune_encoder}")

    encoder.eval()
    hypernet.train()
    attn_module.train()
    safety_head.eval()

    reg_weight = float(cfg.reg_weight)
    attn_sharp_weight = float(cfg.attn_sharp_weight)
    attn_bal_weight = float(cfg.attn_bal_weight)
    kd_weight = float(cfg.kd_weight)
    kd_temp = float(cfg.kd_temp)
    print(
        f"[stage2] reg_weight={reg_weight} attn_sharp={attn_sharp_weight} attn_bal={attn_bal_weight} "
        f"kd_weight={kd_weight} kd_temp={kd_temp}"
    )

    aggregator = ActivationAggregator(strict=True)

    for step in tqdm(range(total_steps), desc="stage2"):
        try:
            batch, safety = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch, safety = next(data_iter)

        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        if safety is not None:
            safety = safety.to(cfg.device)

        hook_teacher.clear()
        with torch.no_grad():
            teacher_out = teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        teacher_o = hook_teacher.get_point("o_proj")
        teacher_d = hook_teacher.get_point("down_proj")

        hook_student.clear()
        applier.clear_scalers()
        with torch.no_grad():
            _ = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        student_o = hook_student.get_point("o_proj")
        student_d = hook_student.get_point("down_proj")

        agg_to = aggregator.dict_to_blD(teacher_o, num_layers=teacher_layers, pooling="last")
        agg_td = aggregator.dict_to_blD(teacher_d, num_layers=teacher_layers, pooling="last")
        agg_so = aggregator.dict_to_blD(student_o, num_layers=num_layers, pooling="last")
        agg_sd = aggregator.dict_to_blD(student_d, num_layers=num_layers, pooling="last")
        hook_teacher.clear()
        hook_student.clear()

        B = batch["input_ids"].size(0)
        summaries_b = {name: t.unsqueeze(0).expand(B, -1, -1) for name, t in summary_tensors.items()}
        summaries_b_model = {name: t.to(dtype=torch.float32) for name, t in summaries_b.items()}

        with torch.set_grad_enabled(tune_encoder):
            layer_emb, _summary_pred = encoder(summaries_b_model)

        z, alpha = attn_module(
            student_o=agg_so.x.float(),
            student_d=agg_sd.x.float(),
            teacher_o=agg_to.x.float(),
            teacher_d=agg_td.x.float(),
        )
        hn_out = hypernet(guidance=z, layer_emb=layer_emb)

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

        loss_lm = outputs.loss.float()
        safety_loss = None
        if need_safety:
            hidden = outputs.hidden_states[-1]
            logits = safety_head(hidden, batch["attention_mask"]).float()
            safety_loss = nn.functional.binary_cross_entropy_with_logits(logits, safety.float())
            loss_lm = loss_lm + cfg.safety_weight * safety_loss

        with torch.no_grad():
            logits_T = teacher_out.logits.float()
        logits_S = outputs.logits.float()
        loss_mask = (batch["labels"] != -100)
        denom = loss_mask.sum().clamp(min=1)
        logits_T = logits_T[loss_mask]
        logits_S = logits_S[loss_mask]
        log_pS = F.log_softmax(logits_S / kd_temp, dim=-1)
        pT = F.softmax(logits_T / kd_temp, dim=-1)
        loss_kd = F.kl_div(log_pS, pT, reduction="batchmean") * (kd_temp * kd_temp)

        eps = 1e-8
        alpha_clamped = torch.clamp(alpha, min=eps)
        entropy = -(alpha_clamped * alpha_clamped.log()).sum(dim=-1).mean()
        mean_alpha = alpha.mean(dim=(0, 1))
        uniform = torch.full_like(mean_alpha, 1.0 / mean_alpha.numel())
        kl_bal = (mean_alpha * (mean_alpha + eps).log()).sum() - (
            mean_alpha * (uniform + eps).log()
        ).sum()
        loss_attn = attn_sharp_weight * entropy + attn_bal_weight * kl_bal

        reg_terms = []
        for key in ("o_proj", "down_proj"):
            factor = 1.0 + scalers[key].float()
            reg_terms.append((factor - 1.0).pow(2).mean())
        loss_reg = torch.stack(reg_terms).mean()

        loss = loss_lm + kd_weight * loss_kd + loss_attn + reg_weight * loss_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
        optimizer.step()

        if (step + 1) % cfg.logging_steps == 0:
            top1 = alpha.max(dim=-1).values.mean().item()
            msg = (
                f"step={step+1} loss={loss.item():.4f} lm={loss_lm.item():.4f} "
                f"L_kd={loss_kd.item():.4f} L_attn={loss_attn.item():.4f} "
                f"L_reg={loss_reg.item():.4f} entropy={entropy.item():.4f} "
                f"balance_kl={kl_bal.item():.4f} top1={top1:.4f} ans_tokens={int(denom.item())}"
            )
            if safety_loss is not None:
                msg += f" safety={safety_loss.item():.4f}"
            print(msg)

        log_path = os.path.join(cfg.save_dir, "stage2_metrics.jsonl")
        os.makedirs(cfg.save_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            record = {
                "step": step + 1,
                "loss": float(loss.item()),
                "loss_lm": float(loss_lm.item()),
                "loss_kd": float(loss_kd.item()),
                "loss_attn": float(loss_attn.item()),
                "loss_reg": float(loss_reg.item()),
                "entropy": float(entropy.item()),
                "top1": float(alpha.max(dim=-1).values.mean().item()),
                "balance_kl": float(kl_bal.item()),
                "answer_tokens": float(denom.item()),
                "safety_loss": float(safety_loss.item()) if safety_loss is not None else None,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    ckpt = {
        "encoder": encoder.state_dict(),
        "hypernet": hypernet.state_dict(),
        "attention": attn_module.state_dict(),
        "config": {
            "summary_rank": cfg.summary_rank,
            "summary_dim": summary_dim,
            "components": components,
            "tune_encoder": tune_encoder,
        },
    }
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(cfg.save_dir, "stage2.pt"))

    hook_teacher.remove()
    hook_student.remove()
    applier.remove()


if __name__ == "__main__":
    main()
