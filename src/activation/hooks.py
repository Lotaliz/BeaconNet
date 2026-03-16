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


def find_target_modules(model: nn.Module, target_names: Iterable[str]) -> Dict[str, nn.Module]:
    suffixes = tuple(target_names)
    modules: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(suffixes):
            modules[name] = module
    if not modules:
        raise ValueError("No target linear modules found for activation inspection.")
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
