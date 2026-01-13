from __future__ import annotations

from dataclasses import dataclass
import os
import torch


@dataclass(frozen=True)
class AppConfig:
    model_name: str
    model_path: str
    dataset_path: str
    save_dir: str
    device: str
    dtype: torch.dtype


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config() -> AppConfig:
    device = os.environ.get("DEVICE", _default_device())
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    return AppConfig(
        model_name=os.environ.get("MODEL_NAME", "slm"),
        model_path=os.environ.get("SLM_PATH", "YOUR_SLM_PATH_HERE"),
        dataset_path=os.environ.get("DATASET_PATH", "YOUR_DATASET_PATH_HERE"),
        save_dir=os.environ.get("SAVE_DIR", "./outputs"),
        device=device,
        dtype=dtype,
    )
