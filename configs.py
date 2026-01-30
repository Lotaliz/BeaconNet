from __future__ import annotations

from dataclasses import dataclass
import os
import torch


@dataclass(frozen=True)
class AppConfig:
    model_name: str
    model_path: str
    teacher_model_path: str
    dataset_path: str
    save_dir: str
    device: str
    dtype: torch.dtype
    max_length: int
    batch_size: int
    num_steps: int
    num_epochs: int
    stage1_epochs: int
    stage2_epochs: int
    stage3_epochs: int
    logging_steps: int
    lr: float
    safety_weight: float
    reg_weight: float
    attn_sharp_weight: float
    attn_bal_weight: float
    kd_weight: float
    kd_temp: float
    lambda_pos: float
    safe_light_weight: float
    unfreeze_hypernet: bool
    stage2_scaler_cache: str
    stage1_ckpt_path: str
    stage2_ckpt_path: str
    stage3_ckpt_path: str
    inference_output_path: str
    test_sample_size: int
    test_seed: int
    gen_max_new_tokens: int
    val_dataset_path: str
    val_output_path: str
    val_sample_size: int
    val_seed: int
    llama_guard_model_path: str
    guard_max_new_tokens: int
    num_basis: int
    d_attn: int
    summary_rank: int
    summary_cache_path: str
    summary_loss_weight: float
    d_layer_emb: int


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config() -> AppConfig:
    device = os.environ.get("DEVICE", _default_device())
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model_name = "llama3"
    summary_rank = int(os.environ.get("SUMMARY_RANK", "8"))

    return AppConfig(
        model_name=model_name,
        model_path="models/llama3.2-1B-Instruct",
        teacher_model_path="models/llama3.1-8B-Instruct",
        dataset_path="datasets/PKU-SafeRLHF/data/Alpaca-7B/train.jsonl",
        save_dir="models/BN_output/glue",
        device=device,
        dtype=dtype,
        max_length=512,
        batch_size=8,
        num_steps=100,
        num_epochs=3,
        stage1_epochs=3,
        stage2_epochs=3,
        stage3_epochs=3,
        logging_steps=250,
        lr=1e-5,
        safety_weight=1.0,
        reg_weight=1e-3,
        attn_sharp_weight=0.5,
        attn_bal_weight=0.5,
        kd_weight=0.5,
        kd_temp=2.0,
        lambda_pos=1.0,
        safe_light_weight=1.0,
        unfreeze_hypernet=True,
        stage2_scaler_cache="",
        stage1_ckpt_path="models/BN_output/glue/stage1.pt",
        stage2_ckpt_path="models/BN_output/glue/stage2.pt",
        stage3_ckpt_path="models/BN_output/glue/stage3.pt",
        inference_output_path="models/BN_output/glue/inference_results.json",
        test_sample_size=512,
        test_seed=0,
        gen_max_new_tokens=128,
        val_dataset_path="datasets/hex-phi/hex-phi.json",
        val_output_path="models/BN_output/glue/val_results_hex_phi.json",
        val_sample_size=300,
        val_seed=0,
        llama_guard_model_path="models/llama-guard-3",
        guard_max_new_tokens=64,
        num_basis=64,
        d_attn=256,
        summary_rank=summary_rank,
        summary_cache_path=f"datasets/layer_summaries/{model_name}_rank{summary_rank}.pt",
        summary_loss_weight=1.0,
        d_layer_emb=512,
    )
