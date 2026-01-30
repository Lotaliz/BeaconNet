import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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
from src.utils.activation_aggregator import ActivationAggregator


class JsonlDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.samples: List[Dict[str, Any]] = []
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = dict(self.samples[idx])
        s["__idx"] = idx
        return s


class SafetyHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (hidden_states * mask).sum(dim=1) / denom
        return self.proj(pooled).squeeze(-1)


class LightZNet(nn.Module):
    def __init__(self, hidden_size: int, d_attn: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, d_attn),
            nn.GELU(),
            nn.Linear(d_attn, d_attn),
        )

    def forward(self, o_proj: torch.Tensor, down_proj: torch.Tensor) -> torch.Tensor:
        x = torch.cat([o_proj, down_proj], dim=-1)
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


def _extract_prompt(sample: Dict[str, Any]) -> str:
    if "prompt" in sample:
        return str(sample["prompt"])
    if "text" in sample:
        return str(sample["text"])
    return ""


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


def _load_stage1(path: str, encoder: LayerEmbeddingEncoder, safety_head: SafetyHead) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stage1 checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    if "safety_head" in ckpt:
        safety_head.load_state_dict(ckpt["safety_head"])
    return ckpt


def _load_stage3(path: str, hypernet: HypernetScaler, lightz: LightZNet) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stage3 checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if "hypernet" in ckpt:
        hypernet.load_state_dict(ckpt["hypernet"])
    if "lightz" in ckpt:
        lightz.load_state_dict(ckpt["lightz"])
    return ckpt


@dataclass
class BatchPrompts:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_texts: List[str]
    indices: List[int]


def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg.test_seed)
    random.seed(cfg.test_seed)

    print(f"[infer] model_path={cfg.model_path} device={cfg.device} dtype={cfg.dtype}")
    print(f"[infer] dataset_path={cfg.dataset_path} output={cfg.inference_output_path}")
    print(f"[infer] sample_size={cfg.test_sample_size} seed={cfg.test_seed} max_new_tokens={cfg.gen_max_new_tokens}")

    tok = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
    # Decoder-only models should use left-padding for correct generation with batched prompts.
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        dtype=cfg.dtype,
        device_map=cfg.device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Silence generation warnings by explicitly using greedy settings and neutral sampling params.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0

    num_layers, hidden_size = _infer_num_layers_and_hidden_size(model)

    components, summaries, summary_dim = _load_or_build_summaries(
        model=model,
        num_layers=num_layers,
        rank=cfg.summary_rank,
        cache_path=cfg.summary_cache_path,
    )

    encoder = LayerEmbeddingEncoder(
        num_layers=num_layers,
        summary_dim=summary_dim,
        component_names=components,
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

    safety_head = SafetyHead(hidden_size=hidden_size).to(device=cfg.device, dtype=torch.float32)

    _load_stage1(cfg.stage1_ckpt_path, encoder, safety_head)
    _load_stage3(cfg.stage3_ckpt_path, hypernet, lightz)

    encoder.eval()
    hypernet.eval()
    lightz.eval()
    safety_head.eval()

    summary_tensors = {
        name: summaries[name].to(device=cfg.device, dtype=torch.float32) for name in components
    }
    summaries_1 = {name: t.unsqueeze(0) for name, t in summary_tensors.items()}
    with torch.no_grad():
        layer_emb_1, _ = encoder(summaries_1)
    layer_emb_1 = layer_emb_1.detach()  # (1, L, d_layer_emb)

    hook_mgr = ActivationHookManager(detach=True, to_cpu=False, keep_sequence_dim=False)
    hook_mgr.register_llama_o_proj_and_down_proj(model)

    applier = ScalerApplier(strength_o=1.0, strength_d=1.0, clamp_min=0.5, clamp_max=1.5)
    applier.attach_llama_o_proj_and_down_proj(model)

    aggregator = ActivationAggregator(strict=True)

    dataset = JsonlDataset(cfg.dataset_path)
    n = len(dataset)
    k = min(int(cfg.test_sample_size), n)
    indices = list(range(n))
    random.shuffle(indices)
    sel = indices[:k]

    def collate(samples: List[Dict[str, Any]]) -> BatchPrompts:
        prompt_texts = [_extract_prompt(s) for s in samples]
        idxs = [int(s["__idx"]) for s in samples]
        enc = tok(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        return BatchPrompts(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            prompt_texts=prompt_texts,
            indices=idxs,
        )

    subset = [dataset[i] for i in sel]
    loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    results: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc="infer"):
        input_ids = batch.input_ids.to(cfg.device)
        attention_mask = batch.attention_mask.to(cfg.device)
        B = input_ids.size(0)

        with torch.no_grad():
            applier.clear_scalers()
            baseline_ids = model.generate(
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

        def _decode(seqs: torch.Tensor) -> List[str]:
            return tok.batch_decode(seqs, skip_special_tokens=True)

        baseline_texts = _decode(baseline_ids)
        tuned_texts = _decode(tuned_ids)

        def _safety_score(seqs: torch.Tensor) -> Tuple[List[float], List[float]]:
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

        base_logit, base_prob = _safety_score(baseline_ids)
        tuned_logit, tuned_prob = _safety_score(tuned_ids)

        for i in range(B):
            results.append(
                {
                    "idx": int(batch.indices[i]),
                    "prompt": batch.prompt_texts[i],
                    "baseline_text": baseline_texts[i],
                    "tuned_text": tuned_texts[i],
                    "baseline_safety_logit": float(base_logit[i]),
                    "baseline_safety_prob": float(base_prob[i]),
                    "tuned_safety_logit": float(tuned_logit[i]),
                    "tuned_safety_prob": float(tuned_prob[i]),
                }
            )

    os.makedirs(os.path.dirname(cfg.inference_output_path) or ".", exist_ok=True)
    with open(cfg.inference_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "model_path": cfg.model_path,
                    "stage1_ckpt_path": cfg.stage1_ckpt_path,
                    "stage3_ckpt_path": cfg.stage3_ckpt_path,
                    "dataset_path": cfg.dataset_path,
                    "sample_size": cfg.test_sample_size,
                    "seed": cfg.test_seed,
                    "max_new_tokens": cfg.gen_max_new_tokens,
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
