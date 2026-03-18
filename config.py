import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
MODELS_ROOT = PROJECT_ROOT / "models"
DATA_ROOT = PROJECT_ROOT / "data"
DATASETS_ROOT = PROJECT_ROOT / "datasets"


def _cuda_available() -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False


CUDA_AVAILABLE = _cuda_available()


def _format_ratio(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _activation_report_filename(model_path: str) -> str:
    model_dir = Path(model_path).name
    marker = "-wanda-"
    if marker in model_dir:
        model_name, ratio = model_dir.rsplit(marker, 1)
        return f"report-{ratio}-{model_name}.json"
    return f"report-base-{model_dir}.json"


def _activation_report_filename_with_adapter(model_path: str, adapter_path: str = "") -> str:
    base_name = _activation_report_filename(model_path).removesuffix(".json")
    if adapter_path:
        return f"{base_name}-adapter-{Path(adapter_path).name}.json"
    return f"{base_name}.json"


@dataclass
class PruneConfig:
    method: str = "wanda"
    model_name: str = "llama3.1-8B-Instruct"
    model_path: str = str(MODELS_ROOT / "llama3.1-8B-Instruct")
    lora_adapter_path: str = str(MODELS_ROOT / "aligned" / "llama3.1-8B-Instruct-dpo")
    calibration_data_path: str = str(SRC_ROOT / "test.jsonl")
    output_root: str = str(MODELS_ROOT / "pruned")
    output_name_prefix: str = "llama3.1-8B-Instruct-dpo-wanda"
    device: str = "cuda" if CUDA_AVAILABLE else "cpu"
    torch_dtype: torch.dtype = torch.float16 if CUDA_AVAILABLE else torch.float32
    sparsity_ratio: float = 0.60
    calibration_samples: int = 128
    batch_size: int = 1
    max_length: int = 512
    seed: int = 42
    target_linear_names: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    @property
    def formatted_sparsity_ratio(self) -> str:
        return _format_ratio(self.sparsity_ratio)

    @property
    def output_name(self) -> str:
        return f"{self.output_name_prefix}-{self.formatted_sparsity_ratio}"

    @property
    def save_path(self) -> str:
        return str(Path(self.output_root) / self.output_name)

    @property
    def pruned_model_path(self) -> str:
        return self.save_path


@dataclass
class ActivationConfig:
    model_name: str = "llama3.1-8B-Instruct-0.6"
    # model_path: str = str(MODELS_ROOT / "pruned" / "llama3.1-8B-Instruct-wanda-0.6")
    model_path: str = str(MODELS_ROOT / "llama3.1-8B-Instruct")
    lora_adapter_path: str = str(MODELS_ROOT / "aligned" / "llama3.1-8B-Instruct-dpo")
    use_lora_adapter: bool = False
    output_dir: str = str(DATA_ROOT / "activation")
    compare_base_file: str = str(DATA_ROOT / "activation" / "report-base-llama3.1-8B-Instruct.json")
    compare_target_file: str = str(DATA_ROOT / "activation" / "report-0.6-llama3.1-8B-Instruct.json")
    compare_output_file: str = "activation_compare.html"
    path_compare_dir: str = str(DATA_ROOT / "activation" / "path_compare")
    path_compare_index_file: str = "activation_path_compare.html"
    dataset_path: str = str(DATASETS_ROOT / "advbench" / "data" / "advbench_clean.json")
    prompt_fields: tuple[str, ...] = ("original_prompt", "paraphrase")
    max_samples_per_field: int = 0
    target_linear_names: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    max_length: int = 512
    top_k: int = 20
    token_position: str = "last"
    save_full_vector: bool = True
    device: str = "cuda" if CUDA_AVAILABLE else "cpu"
    torch_dtype: torch.dtype = torch.float16 if CUDA_AVAILABLE else torch.float32

    @property
    def output_path(self) -> str:
        adapter_path = self.lora_adapter_path if self.use_lora_adapter else ""
        return str(Path(self.output_dir) / _activation_report_filename_with_adapter(self.model_path, adapter_path))

    @property
    def output_file(self) -> str:
        return Path(self.output_path).name

    @property
    def compare_output_path(self) -> str:
        return str(Path(self.output_dir) / self.compare_output_file)

    @property
    def path_compare_index_path(self) -> str:
        return str(Path(self.path_compare_dir) / self.path_compare_index_file)


@dataclass
class AlignmentConfig:
    model_name: str = "llama3.1-8B-Instruct"
    model_path: str = str(MODELS_ROOT / "llama3.1-8B-Instruct")
    dataset_name: str = "PKU-SafeRLHF-Alpaca2-7B"
    train_dataset_path: str = str(DATASETS_ROOT / "PKU-SafeRLHF" / "data" / "Alpaca2-7B" / "train.jsonl")
    eval_dataset_path: str = str(DATASETS_ROOT / "PKU-SafeRLHF" / "data" / "Alpaca2-7B" / "test.jsonl")
    output_dir: str = str(MODELS_ROOT / "aligned" / "llama3.1-8B-Instruct-dpo")
    logging_dir: str = str(DATA_ROOT / "align" / "logs")
    report_to: str = "none"
    preference_mode: str = "safer"
    max_prompt_length: int = 1024
    max_length: int = 1536
    max_train_samples: int = 0
    max_eval_samples: int = 0
    beta: float = 0.1
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 2
    seed: int = 42
    gradient_checkpointing: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    use_bf16: bool = CUDA_AVAILABLE and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    use_fp16: bool = CUDA_AVAILABLE and not bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())


@dataclass
class AppConfig:
    project_root: str = str(PROJECT_ROOT)
    src_root: str = str(SRC_ROOT)
    data_root: str = str(DATA_ROOT)
    datasets_root: str = str(DATASETS_ROOT)
    models_root: str = str(MODELS_ROOT)

    model_name: str = "llama3.1-8B-Instruct"
    model_path: str = str(MODELS_ROOT / "llama3.1-8B-Instruct")
    llama_guard_model_path: str = str(MODELS_ROOT / "llama-guard-3")

    eval_dataset_name: str = "strongreject"
    eval_dataset_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "strongreject": str(DATA_ROOT / "eval_results_strongreject.json"),
            "hex-phi": str(DATA_ROOT / "eval_results_hex-phi.json"),
        }
    )
    eval_output_dir: str = str(DATA_ROOT)
    eval_seed: int = 42
    eval_sample_size: int = 64
    max_length: int = 512
    batch_size: int = 1
    gen_max_new_tokens: int = 256
    guard_max_new_tokens: int = 128
    alpaca_dataset_path: str = str(DATASETS_ROOT / "alpaca" / "alpaca.jsonl")
    prune_eval_output_path: str = str(DATA_ROOT / "alpaca_prune_compare.json")
    prune_eval_sample_size: int = 100
    prune_eval_seed: int = 42
    prune_eval_max_new_tokens: int = 256
    safety_dataset_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "advbench": str(DATASETS_ROOT / "advbench" / "data" / "advbench.json"),
            "hex-phi": str(DATASETS_ROOT / "hex-phi" / "hex-phi.json"),
            "strongreject": str(DATASETS_ROOT / "strongreject" / "data" / "strongreject.json"),
        }
    )
    safety_output_dir: str = str(DATA_ROOT / "safety")
    safety_sample_size: int = 100
    safety_seed: int = 42
    safety_generation_max_new_tokens: int = 256
    safety_guard_max_new_tokens: int = 128

    dtype: torch.dtype = torch.float16 if CUDA_AVAILABLE else torch.float32
    device: str = "cuda" if CUDA_AVAILABLE else "cpu"

    prune: PruneConfig = field(default_factory=PruneConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    align: AlignmentConfig = field(default_factory=AlignmentConfig)


def load_config() -> AppConfig:
    return AppConfig()


def config_to_dict(cfg: AppConfig) -> Dict[str, Any]:
    prune = cfg.prune
    activation = cfg.activation
    align = cfg.align
    return {
        "project_root": cfg.project_root,
        "models_root": cfg.models_root,
        "datasets_root": cfg.datasets_root,
        "model_name": cfg.model_name,
        "model_path": cfg.model_path,
        "device": cfg.device,
        "dtype": str(cfg.dtype),
        "alpaca_dataset_path": cfg.alpaca_dataset_path,
        "prune_eval_output_path": cfg.prune_eval_output_path,
        "prune_eval_sample_size": cfg.prune_eval_sample_size,
        "prune_eval_seed": cfg.prune_eval_seed,
        "prune_eval_max_new_tokens": cfg.prune_eval_max_new_tokens,
        "safety_dataset_paths": cfg.safety_dataset_paths,
        "safety_output_dir": cfg.safety_output_dir,
        "safety_sample_size": cfg.safety_sample_size,
        "safety_seed": cfg.safety_seed,
        "safety_generation_max_new_tokens": cfg.safety_generation_max_new_tokens,
        "safety_guard_max_new_tokens": cfg.safety_guard_max_new_tokens,
        "activation": {
            "model_name": activation.model_name,
            "model_path": activation.model_path,
            "lora_adapter_path": activation.lora_adapter_path,
            "use_lora_adapter": activation.use_lora_adapter,
            "output_dir": activation.output_dir,
            "output_file": activation.output_file,
            "output_path": activation.output_path,
            "compare_base_file": activation.compare_base_file,
            "compare_target_file": activation.compare_target_file,
            "compare_output_file": activation.compare_output_file,
            "compare_output_path": activation.compare_output_path,
            "path_compare_dir": activation.path_compare_dir,
            "path_compare_index_file": activation.path_compare_index_file,
            "path_compare_index_path": activation.path_compare_index_path,
            "dataset_path": activation.dataset_path,
            "prompt_fields": list(activation.prompt_fields),
            "max_samples_per_field": activation.max_samples_per_field,
            "target_linear_names": list(activation.target_linear_names),
            "max_length": activation.max_length,
            "top_k": activation.top_k,
            "token_position": activation.token_position,
            "save_full_vector": activation.save_full_vector,
            "device": activation.device,
            "torch_dtype": str(activation.torch_dtype),
        },
        "align": {
            "model_name": align.model_name,
            "model_path": align.model_path,
            "dataset_name": align.dataset_name,
            "train_dataset_path": align.train_dataset_path,
            "eval_dataset_path": align.eval_dataset_path,
            "output_dir": align.output_dir,
            "logging_dir": align.logging_dir,
            "report_to": align.report_to,
            "preference_mode": align.preference_mode,
            "max_prompt_length": align.max_prompt_length,
            "max_length": align.max_length,
            "max_train_samples": align.max_train_samples,
            "max_eval_samples": align.max_eval_samples,
            "beta": align.beta,
            "learning_rate": align.learning_rate,
            "weight_decay": align.weight_decay,
            "num_train_epochs": align.num_train_epochs,
            "per_device_train_batch_size": align.per_device_train_batch_size,
            "per_device_eval_batch_size": align.per_device_eval_batch_size,
            "gradient_accumulation_steps": align.gradient_accumulation_steps,
            "warmup_ratio": align.warmup_ratio,
            "logging_steps": align.logging_steps,
            "save_steps": align.save_steps,
            "eval_steps": align.eval_steps,
            "save_total_limit": align.save_total_limit,
            "seed": align.seed,
            "gradient_checkpointing": align.gradient_checkpointing,
            "use_lora": align.use_lora,
            "lora_r": align.lora_r,
            "lora_alpha": align.lora_alpha,
            "lora_dropout": align.lora_dropout,
            "lora_target_modules": list(align.lora_target_modules),
            "use_bf16": align.use_bf16,
            "use_fp16": align.use_fp16,
        },
        "prune": {
            "method": prune.method,
            "model_name": prune.model_name,
            "model_path": prune.model_path,
            "lora_adapter_path": prune.lora_adapter_path,
            "calibration_data_path": prune.calibration_data_path,
            "output_root": prune.output_root,
            "output_name": prune.output_name,
            "output_name_prefix": prune.output_name_prefix,
            "formatted_sparsity_ratio": prune.formatted_sparsity_ratio,
            "save_path": prune.save_path,
            "pruned_model_path": prune.pruned_model_path,
            "device": prune.device,
            "torch_dtype": str(prune.torch_dtype),
            "sparsity_ratio": prune.sparsity_ratio,
            "calibration_samples": prune.calibration_samples,
            "batch_size": prune.batch_size,
            "max_length": prune.max_length,
            "seed": prune.seed,
            "target_linear_names": list(prune.target_linear_names),
        },
    }
