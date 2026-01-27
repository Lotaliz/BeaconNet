from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inst.hooks import ActivationHookManager
from src.inst.apply_scaler import ScalerApplier
from src.utils.activation_aggregator import ActivationAggregator
from src.model.hypernet import HypernetScaler
from configs import load_config


def _infer_num_layers_and_hidden_size(model) -> tuple[int, int]:
    # Infer num_layers and hidden_size from common HF config fields
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no config.")

    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if num_layers is None or hidden_size is None:
        raise ValueError("Failed to infer num_layers/hidden_size from model.config.")

    return int(num_layers), int(hidden_size)


def main() -> None:
    # Configure paths and runtime
    cfg = load_config()

    # Load model/tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=cfg.dtype, device_map=cfg.device)
    model.eval()

    # Build instrumentation
    hook_mgr = ActivationHookManager(detach=True, to_cpu=False)
    hook_mgr.register_llama_o_proj_and_down_proj(model)

    applier = ScalerApplier(
        strength_o=0.5,
        strength_d=0.5,
        clamp_min=0.5,
        clamp_max=1.5,
    )
    applier.attach_llama_o_proj_and_down_proj(model)

    # Prepare a minimal prompt batch
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain why unsafe requests should be refused in one sentence."},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(cfg.device)

    # First forward: capture activations without scaling
    hook_mgr.clear()
    applier.clear_scalers()

    with torch.no_grad():
        out0 = model(**inputs)
        logits0 = out0.logits

    acts_o: Dict[int, torch.Tensor] = hook_mgr.get_point("o_proj")
    acts_d: Dict[int, torch.Tensor] = hook_mgr.get_point("down_proj")
    print(f"[act] acts_o: {len(acts_o)}, act_d: {len(acts_d)}")
    # Aggregate activations to (B, L, D)
    num_layers, hidden_size = _infer_num_layers_and_hidden_size(model)
    aggregator = ActivationAggregator(strict=True)

    agg_o = aggregator.dict_to_blD(acts_o, num_layers=num_layers, pooling="last")
    agg_d = aggregator.dict_to_blD(acts_d, num_layers=num_layers, pooling="last")

    print(f"[agg] o_proj:   {tuple(agg_o.x.shape)} (expected B,L,D={hidden_size})")
    print(f"[agg] down_proj:{tuple(agg_d.x.shape)} (expected B,L,D={hidden_size})")

    # Build a simple guidance vector for smoke testing
    B = agg_o.x.size(0)
    guidance_raw = agg_o.x.mean(dim=1)  # (B, D)

    proj = torch.nn.Linear(hidden_size, 512, bias=False).to(device=cfg.device, dtype=cfg.dtype)
    torch.nn.init.normal_(proj.weight, mean=0.0, std=0.02)

    with torch.no_grad():
        guidance = proj(guidance_raw)  # (B, 512)

    # Hypernet: predict scalers (B, L, D)
    hypernet = HypernetScaler(
        num_layers=num_layers,
        hidden_size=hidden_size,
        d_layer_emb=512,
        d_guidance=512,
        d_model=512,
        n_heads=8,
    ).to(device=cfg.device, dtype=cfg.dtype)
    hypernet.eval()

    with torch.no_grad():
        hn_out = hypernet(guidance=guidance)
        scalers = hn_out.scalers
        with torch.no_grad():
          scalers["o_proj"] = torch.randn_like(scalers["o_proj"]) * 0.2
          scalers["down_proj"] = torch.randn_like(scalers["down_proj"]) * 0.2

        print(f"[hn] o_proj scaler:   {tuple(scalers['o_proj'].shape)}")
        print(f"[hn] down_proj scaler:{tuple(scalers['down_proj'].shape)}")

    # Second forward: apply scaling via hooks and re-run
    applier.set_scalers({"o_proj": scalers["o_proj"], "down_proj": scalers["down_proj"]})

    with torch.no_grad():
        out1 = model(**inputs)
        logits1 = out1.logits

    # Basic sanity check: logits should change if scaling is active
    diff = (logits1 - logits0).abs().mean().item()
    print(f"[check] mean(|logits1-logits0|) = {diff:.6e}")

    # Cleanup hooks
    hook_mgr.remove()
    applier.remove()


if __name__ == "__main__":
    main()
