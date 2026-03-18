from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
import torch.nn as nn


@dataclass
class ActivationSummary:
    layer_name: str
    max_neuron_index: int
    max_activation_value: float
    topk_neuron_indices: List[int]
    topk_activation_values: List[float]
    activation_vector: List[float]


@dataclass
class AverageActivationSummary:
    layer_name: str
    sample_count: int
    max_neuron_index: int
    max_activation_value: float
    topk_neuron_indices: List[int]
    topk_activation_values: List[float]
    avg_activation_vector: List[float]


def _is_supported_linear(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    base_layer = getattr(module, "base_layer", None)
    return isinstance(base_layer, nn.Linear)


def _matches_target_suffix(name: str, target_names: Iterable[str]) -> bool:
    parts = name.split(".")
    if not parts:
        return False
    last = parts[-1]
    prev = parts[-2] if len(parts) >= 2 else ""
    for target_name in target_names:
        if last == target_name:
            return True
        if last == "base_layer" and prev == target_name:
            return True
    return False


def find_target_modules(model: nn.Module, target_names: Iterable[str]) -> Dict[str, nn.Module]:
    suffixes = tuple(target_names)
    modules: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if _is_supported_linear(module) and _matches_target_suffix(name, suffixes):
            modules[name] = module
    if not modules:
        raise ValueError(
            "No target linear modules found for activation inspection. "
            "Check target_linear_names or whether the model is wrapped by PEFT/LoRA."
        )
    return modules


class ActivationCollector:
    def __init__(self, model: nn.Module, target_names: Iterable[str], top_k: int, token_position: str) -> None:
        self.model = model
        self.target_names = tuple(target_names)
        self.top_k = top_k
        self.token_position = token_position
        self._summaries: Dict[str, ActivationSummary] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self) -> None:
        modules = find_target_modules(self.model, self.target_names)
        for name, module in modules.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def clear(self) -> None:
        self._summaries.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summaries(self) -> Dict[str, ActivationSummary]:
        return dict(self._summaries)

    def _make_hook(self, layer_name: str):
        def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> None:
            hidden = output.detach().float()
            if hidden.ndim == 3:
                if self.token_position == "last":
                    vector = hidden[0, -1]
                else:
                    vector = hidden[0, 0]
            elif hidden.ndim == 2:
                vector = hidden[-1]
            else:
                vector = hidden.reshape(-1)

            abs_vector = vector.abs()
            k = min(self.top_k, abs_vector.numel())
            top_values, top_indices = torch.topk(abs_vector, k=k)
            max_idx = int(top_indices[0].item())
            self._summaries[layer_name] = ActivationSummary(
                layer_name=layer_name,
                max_neuron_index=max_idx,
                max_activation_value=float(vector[max_idx].item()),
                topk_neuron_indices=[int(idx.item()) for idx in top_indices],
                topk_activation_values=[float(vector[int(idx.item())].item()) for idx in top_indices],
                activation_vector=vector.tolist(),
            )

        return hook


class AverageActivationCollector:
    def __init__(self, model: nn.Module, target_names: Iterable[str], top_k: int, token_position: str) -> None:
        self.model = model
        self.target_names = tuple(target_names)
        self.top_k = top_k
        self.token_position = token_position
        self._sum_vectors: Dict[str, torch.Tensor] = {}
        self._counts: Dict[str, int] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self) -> None:
        modules = find_target_modules(self.model, self.target_names)
        for name, module in modules.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def reset(self) -> None:
        self._sum_vectors.clear()
        self._counts.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summaries(self) -> Dict[str, AverageActivationSummary]:
        summaries: Dict[str, AverageActivationSummary] = {}
        for layer_name, sum_vector in self._sum_vectors.items():
            count = self._counts[layer_name]
            mean_vector = sum_vector / count
            abs_vector = mean_vector.abs()
            k = min(self.top_k, abs_vector.numel())
            _top_values, top_indices = torch.topk(abs_vector, k=k)
            max_idx = int(top_indices[0].item())
            summaries[layer_name] = AverageActivationSummary(
                layer_name=layer_name,
                sample_count=count,
                max_neuron_index=max_idx,
                max_activation_value=float(mean_vector[max_idx].item()),
                topk_neuron_indices=[int(idx.item()) for idx in top_indices],
                topk_activation_values=[float(mean_vector[int(idx.item())].item()) for idx in top_indices],
                avg_activation_vector=mean_vector.tolist(),
            )
        return summaries

    def _make_hook(self, layer_name: str):
        def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> None:
            hidden = output.detach().float()
            if hidden.ndim == 3:
                if self.token_position == "last":
                    vector = hidden[0, -1]
                else:
                    vector = hidden[0, 0]
            elif hidden.ndim == 2:
                vector = hidden[-1]
            else:
                vector = hidden.reshape(-1)

            vector = vector.cpu()
            if layer_name not in self._sum_vectors:
                self._sum_vectors[layer_name] = torch.zeros_like(vector)
                self._counts[layer_name] = 0
            self._sum_vectors[layer_name] += vector
            self._counts[layer_name] += 1

        return hook
