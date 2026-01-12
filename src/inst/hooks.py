import re
from dataclasses import dataclass
from typing import Dict, Pattern, Tuple, Any

import torch
import torch.nn as nn


@dataclass
class HookSpec:
    """Describe how to match a target module and how to parse its layer index."""
    point: str
    name_pattern: Pattern[str]  # regex with a capturing group for layer index


class ActivationHookManager:
    """
    Register forward hooks to capture activations at selected module outputs.

    Typical use:
      - manager = ActivationHookManager()
      - manager.register_llama_o_proj_and_down_proj(model)
      - manager.clear()
      - _ = model(**batch)
      - acts_o = manager.get_point("o_proj")      # dict[layer_idx -> tensor]
      - acts_d = manager.get_point("down_proj")   # dict[layer_idx -> tensor]
      - manager.remove()
    """

    def __init__(
        self,
        *,
        store_input: bool = False,
        detach: bool = True,
        to_cpu: bool = False,
        keep_sequence_dim: bool = True,
    ) -> None:
        # Whether to store module inputs as well (usually not needed)
        self.store_input = store_input

        # Whether to detach captured tensors from autograd graph
        self.detach = detach

        # Whether to move captured tensors to CPU to save GPU memory
        self.to_cpu = to_cpu

        # Whether to keep (B, T, D) vs pool later; hooks store raw outputs by default
        self.keep_sequence_dim = keep_sequence_dim

        # Internal cache: cache[point][layer_idx] = tensor (from latest forward)
        self._cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # Optional input cache: in_cache[point][layer_idx] = input tensor(s)
        self._in_cache: Dict[str, Dict[int, Any]] = {}

        # Hook handles for removal
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        # Track registered specs to avoid confusion
        self._specs: Dict[str, HookSpec] = {}

    # --------------------------
    # Registration helpers
    # --------------------------

    def register_o_down_proj(self, model: nn.Module) -> None:
        """
        Register hooks for Llama-like module names:
          - model.layers.{i}.self_attn.o_proj
          - model.layers.{i}.mlp.down_proj
        """
        specs = [
            HookSpec(
                point="o_proj",
                name_pattern=re.compile(r"(?:^|.*\.)layers\.(\d+)\.self_attn\.o_proj$"),
            ),
            HookSpec(
                point="down_proj",
                name_pattern=re.compile(r"(?:^|.*\.)layers\.(\d+)\.mlp\.down_proj$"),
            ),
        ]
        self.register_by_specs(model, specs)

    def register_by_specs(self, model: nn.Module, specs: list[HookSpec]) -> None:
        """Register hooks by (point, regex) specs."""
        for spec in specs:
            self._specs[spec.point] = spec
            if spec.point not in self._cache:
                self._cache[spec.point] = {}
            if self.store_input and spec.point not in self._in_cache:
                self._in_cache[spec.point] = {}

        # Walk modules and attach hooks when name matches any spec
        for name, module in model.named_modules():
            for spec in specs:
                m = spec.name_pattern.match(name)
                if m is None:
                    continue
                layer_idx = int(m.group(1))
                handle = module.register_forward_hook(self._make_hook(spec.point, layer_idx))
                self._handles.append(handle)

    # --------------------------
    # Cache APIs
    # --------------------------

    def clear(self) -> None:
        """Clear cached activations for the next forward."""
        for point in self._cache:
            self._cache[point].clear()
        if self.store_input:
            for point in self._in_cache:
                self._in_cache[point].clear()

    def remove(self) -> None:
        """Remove all hooks."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def get_point(self, point: str) -> Dict[int, torch.Tensor]:
        """Return {layer_idx: activation_tensor} for a given point from the latest forward."""
        if point not in self._cache:
            raise KeyError(f"Unknown point '{point}'. Registered points: {list(self._cache.keys())}")
        return self._cache[point]

    def get_input_point(self, point: str) -> Dict[int, Any]:
        """Return {layer_idx: input(s)} for a given point (only if store_input=True)."""
        if not self.store_input:
            raise RuntimeError("store_input=False; no input cache is available.")
        if point not in self._in_cache:
            raise KeyError(f"Unknown point '{point}'. Registered points: {list(self._in_cache.keys())}")
        return self._in_cache[point]

    # --------------------------
    # Internal hook function
    # --------------------------

    def _make_hook(self, point: str, layer_idx: int):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
            # Store module inputs if requested (coarse-grained; keep original tuple)
            if self.store_input:
                self._in_cache[point][layer_idx] = inputs

            # Most targets (Linear) output a Tensor; keep minimal compatibility for tuple outputs
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(out):
                return

            # Optionally detach and/or move to CPU
            if self.detach:
                out = out.detach()
            if self.to_cpu:
                out = out.cpu()

            # Keep raw shape by default (B,T,D or B,D)
            self._cache[point][layer_idx] = out

        return hook
