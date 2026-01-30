from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs import load_config
from src.inst.apply_scaler import ScalerApplier
from src.inst.hooks import ActivationHookManager
from src.model.layer_encoder import LayerEmbeddingEncoder
from src.model.hypernet import HypernetScaler
from src.inst.infer import LightZNet  # reuse same architecture
from src.utils.activation_aggregator import ActivationAggregator


class HexPhiDataset(Dataset):
    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Validation dataset not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError("hex-phi.json must be a JSON list.")
        self.samples: List[Dict[str, Any]] = []
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            row = dict(row)
            row["__idx"] = i
            self.samples.append(row)

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


def _infer_num_layers_and_hidden_size(model) -> Tuple[int, int]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no config.")
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if num_layers is None or hidden_size is None:
        raise ValueError("Failed to infer num_layers/hidden_size from model.config.")
    return int(num_layers), int(hidden_size)


def _extract_prompt(row: Dict[str, Any]) -> str:
    for k in ("prompt", "question", "input", "text"):
        v = row.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for v in row.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _load_stage1(path: str, encoder: LayerEmbeddingEncoder, safety_head: SafetyHead) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stage1 checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    if "safety_head" in ckpt:
        safety_head.load_state_dict(ckpt["safety_head"])


def _load_stage3(path: str, hypernet: HypernetScaler, lightz: LightZNet) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stage3 checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if "hypernet" in ckpt:
        hypernet.load_state_dict(ckpt["hypernet"])
    if "lightz" in ckpt:
        lightz.load_state_dict(ckpt["lightz"])


def _guard_classify(
    prompt: str,
    response: str,
    guard_model,
    guard_tokenizer,
    device: str,
    max_new_tokens: int,
) -> Tuple[str, str]:
    message = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    input_ids = guard_tokenizer.apply_chat_template(message, return_tensors="pt").to(device)
    with torch.no_grad():
        output = guard_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=guard_tokenizer.pad_token_id or guard_tokenizer.eos_token_id,
            eos_token_id=guard_tokenizer.eos_token_id,
        )
    prompt_len = input_ids.shape[-1]
    decoded = guard_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    label = "SAFE" if decoded.lower().startswith("safe") else "UNSAFE"
    return label, decoded


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompts: List[str]
    indices: List[int]
    types: List[Optional[str]]


def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg.val_seed)
    random.seed(cfg.val_seed)

    os.makedirs(os.path.dirname(cfg.val_output_path) or ".", exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, dtype=cfg.dtype, device_map=cfg.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0

    num_layers, hidden_size = _infer_num_layers_and_hidden_size(model)

    encoder = LayerEmbeddingEncoder(
        num_layers=num_layers,
        summary_dim=cfg.summary_rank + 2,
        component_names=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        d_layer_emb=cfg.d_layer_emb,
        d_hidden=cfg.d_layer_emb,
    ).to(device=cfg.device, dtype=torch.float32)

    hypernet = HypernetScaler(
        num_layers=num_layers,
        hidden_size=hidden_size,
        d_layer_emb=cfg.d_layer_emb,
        d_guidance=cfg.d_attn,
        d_model=cfg.d_layer_emb,
        n_heads=8,
    ).to(device=cfg.device, dtype=torch.float32)

    lightz = LightZNet(hidden_size, cfg.d_attn).to(device=cfg.device, dtype=torch.float32)
    safety_head = SafetyHead(hidden_size).to(device=cfg.device, dtype=torch.float32)

    _load_stage1(cfg.stage1_ckpt_path, encoder, safety_head)
    _load_stage3(cfg.stage3_ckpt_path, hypernet, lightz)

    encoder.eval()
    hypernet.eval()
    lightz.eval()
    safety_head.eval()

    # Build and cache per-layer embeddings e_l once (static).
    # This mirrors the training-time xi_l construction based on low-rank summaries.
    summary_path = cfg.summary_cache_path
    data = torch.load(summary_path, map_location="cpu")
    if not isinstance(data, dict) or "summaries" not in data:
        raise ValueError(f"Invalid summary cache at {summary_path}")
    cache_dim = int(data.get("summary_dim", cfg.summary_rank + 2))
    if cache_dim != encoder.summary_dim:  # type: ignore[attr-defined]
        raise ValueError(f"Summary dim mismatch: cache_dim={cache_dim} encoder_dim={encoder.summary_dim}")
    summaries = data["summaries"]
    summary_tensors = {
        name: summaries[name].to(device=cfg.device, dtype=torch.float32).unsqueeze(0)
        for name in encoder.component_names  # type: ignore[attr-defined]
    }
    with torch.no_grad():
        layer_emb_1, _ = encoder(summary_tensors)
    layer_emb_1 = layer_emb_1.detach()  # (1, L, d_layer_emb)

    guard_tokenizer = AutoTokenizer.from_pretrained(cfg.llama_guard_model_path, trust_remote_code=True)
    guard_tokenizer.padding_side = "left"
    if guard_tokenizer.pad_token is None:
        guard_tokenizer.pad_token = guard_tokenizer.eos_token
    guard_model = AutoModelForCausalLM.from_pretrained(
        cfg.llama_guard_model_path,
        dtype=torch.float16,
        device_map=cfg.device,
        trust_remote_code=True,
    )
    guard_model.eval()

    hook_mgr = ActivationHookManager(detach=True, to_cpu=False, keep_sequence_dim=False)
    hook_mgr.register_llama_o_proj_and_down_proj(model)
    applier = ScalerApplier(strength_o=1.0, strength_d=1.0, clamp_min=0.5, clamp_max=1.5)
    applier.attach_llama_o_proj_and_down_proj(model)
    aggregator = ActivationAggregator(strict=True)

    dataset = HexPhiDataset(cfg.val_dataset_path)
    n = len(dataset)
    k = min(int(cfg.val_sample_size), n)
    all_idx = list(range(n))
    random.shuffle(all_idx)
    sel = set(all_idx[:k])
    subset = [dataset[i] for i in range(n) if i in sel]

    def collate(samples: List[Dict[str, Any]]) -> Batch:
        prompts = [_extract_prompt(s) for s in samples]
        indices = [int(s["__idx"]) for s in samples]
        types = [str(s.get("type")) if s.get("type") is not None else None for s in samples]
        enc = tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        return Batch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            prompts=prompts,
            indices=indices,
            types=types,
        )

    loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    results: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc="val-hex-phi"):
        input_ids = batch.input_ids.to(cfg.device)
        attention_mask = batch.attention_mask.to(cfg.device)
        B = input_ids.size(0)

        with torch.no_grad():
            applier.clear_scalers()
            base_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.gen_max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        hook_mgr.clear()
        applier.clear_scalers()
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        acts_o = hook_mgr.get_point("o_proj")
        acts_d = hook_mgr.get_point("down_proj")
        agg_o = aggregator.dict_to_blD(acts_o, num_layers=num_layers, pooling="last")
        agg_d = aggregator.dict_to_blD(acts_d, num_layers=num_layers, pooling="last")
        hook_mgr.clear()

        with torch.no_grad():
            z = lightz(agg_o.x.float(), agg_d.x.float())
            layer_emb = layer_emb_1.expand(B, -1, -1)
            hn = hypernet(guidance=z, layer_emb=layer_emb)
            scalers = {}
            for key, value in hn.scalers.items():
                clean = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
                scalers[key] = torch.clamp(clean, min=-0.5, max=0.5)
            applier.set_scalers(scalers)

        with torch.no_grad():
            tuned_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.gen_max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        applier.clear_scalers()

        base_texts = tok.batch_decode(base_ids, skip_special_tokens=True)
        tuned_texts = tok.batch_decode(tuned_ids, skip_special_tokens=True)

        def safety_scores(seqs: torch.Tensor) -> Tuple[List[float], List[float]]:
            attn = (seqs != tok.pad_token_id).to(dtype=torch.long)
            with torch.no_grad():
                out = model(
                    input_ids=seqs.to(cfg.device),
                    attention_mask=attn.to(cfg.device),
                    output_hidden_states=True,
                )
                hidden = out.hidden_states[-1].float()
                logit = safety_head(hidden, attn.to(cfg.device)).float()
                prob = torch.sigmoid(logit)
            return logit.detach().cpu().tolist(), prob.detach().cpu().tolist()

        base_slogit, base_sprob = safety_scores(base_ids)
        tuned_slogit, tuned_sprob = safety_scores(tuned_ids)

        for i in range(B):
            prompt = batch.prompts[i]
            base_resp = base_texts[i]
            tuned_resp = tuned_texts[i]
            base_guard, base_guard_raw = _guard_classify(
                prompt, base_resp, guard_model, guard_tokenizer, cfg.device, cfg.guard_max_new_tokens
            )
            tuned_guard, tuned_guard_raw = _guard_classify(
                prompt, tuned_resp, guard_model, guard_tokenizer, cfg.device, cfg.guard_max_new_tokens
            )
            results.append(
                {
                    "idx": batch.indices[i],
                    "type": batch.types[i],
                    "prompt": prompt,
                    "baseline_response": base_resp,
                    "tuned_response": tuned_resp,
                    "baseline_safety_logit": float(base_slogit[i]),
                    "baseline_safety_prob": float(base_sprob[i]),
                    "tuned_safety_logit": float(tuned_slogit[i]),
                    "tuned_safety_prob": float(tuned_sprob[i]),
                    "baseline_guard_label": base_guard,
                    "baseline_guard_raw": base_guard_raw,
                    "tuned_guard_label": tuned_guard,
                    "tuned_guard_raw": tuned_guard_raw,
                }
            )

    with open(cfg.val_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "val_dataset_path": cfg.val_dataset_path,
                    "val_sample_size": cfg.val_sample_size,
                    "val_seed": cfg.val_seed,
                    "model_path": cfg.model_path,
                    "stage1_ckpt_path": cfg.stage1_ckpt_path,
                    "stage3_ckpt_path": cfg.stage3_ckpt_path,
                    "llama_guard_model_path": cfg.llama_guard_model_path,
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    hook_mgr.remove()
    applier.remove()


if __name__ == "__main__":
    main()
