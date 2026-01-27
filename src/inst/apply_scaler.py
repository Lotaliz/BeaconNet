import re
from dataclasses import dataclass
from typing import Dict, Optional, Pattern, Any, Tuple
import torch
import torch.nn as nn


@dataclass
class ScalerHookSpec:
    """Describe how to match a target module and parse its layer index."""
    point: str  # "o_proj" or "down_proj"
    name_pattern: Pattern[str]  # regex with a capturing group for layer index


class ScalerApplier:
    """
    Apply per-layer, per-channel scaling to SLM module outputs via forward hooks.
    """

    def __init__(
        self,
        *,
        strength_o: float = 1.0,
        strength_d: float = 1.0,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        eps: float = 1e-6,
    ) -> None:
        # Scaling strengths for each injection point
        self.strength_o = float(strength_o)
        self.strength_d = float(strength_d)

        # Optional clamp range for multiplicative factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Numerical stability
        self.eps = float(eps)

        # scalers[point] = Tensor(B, L, D) for the current forward
        self._scalers: Dict[str, torch.Tensor] = {}

        # Hook handles for removal
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        # Remember which specs are active
        self._specs: Dict[str, ScalerHookSpec] = {}


    def attach_o_down_proj(self, model: nn.Module) -> None:
        """
        Attach hooks for Llama-like module names:
          - model.layers.{i}.self_attn.o_proj
          - model.layers.{i}.mlp.down_proj
        """
        specs = [
            ScalerHookSpec(
                point="o_proj",
                name_pattern=re.compile(r"(?:^|.*\.)layers\.(\d+)\.self_attn\.o_proj$"),
            ),
            ScalerHookSpec(
                point="down_proj",
                name_pattern=re.compile(r"(?:^|.*\.)layers\.(\d+)\.mlp\.down_proj$"),
            ),
        ]
        self.attach_by_specs(model, specs)

    def attach_llama_o_proj_and_down_proj(self, model: nn.Module) -> None:
        """Backward-compatible alias for attach_o_down_proj."""
        self.attach_o_down_proj(model)

    def attach_by_specs(self, model: nn.Module, specs: list[ScalerHookSpec]) -> None:
        """Attach scaler hooks by (point, regex) specs."""
        for spec in specs:
            self._specs[spec.point] = spec

        # Walk modules and attach hooks when name matches any spec
        for name, module in model.named_modules():
            for spec in specs:
                m = spec.name_pattern.match(name)
                if m is None:
                    continue
                layer_idx = int(m.group(1))
                handle = module.register_forward_hook(self._make_scaler_hook(spec.point, layer_idx))
                self._handles.append(handle)

    def set_scalers(self, scalers: Dict[str, torch.Tensor]) -> None:
        """
        Set scalers for the next forward.

        Expected:
          - scalers["o_proj"]    : (B, L, D)
          - scalers["down_proj"] : (B, L, D)
        """
        self._scalers = scalers

    def clear_scalers(self) -> None:
        """Clear scalers so the next forward is not modified."""
        self._scalers = {}

    def remove(self) -> None:
        """Remove all attached hooks."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    # --------------------------
    # Internal hook function
    # --------------------------

    def _make_scaler_hook(self, point: str, layer_idx: int):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
            # If no scaler is provided for this point, do nothing
            if point not in self._scalers:
                return output

            s_all = self._scalers[point]  # (B, L, D)
            if not torch.is_tensor(s_all):
                return output

            # Select per-layer scaler: s = (B, D)
            if s_all.dim() != 3:
                raise ValueError(f"Scaler for point '{point}' must be (B, L, D), got {tuple(s_all.shape)}")
            if layer_idx < 0 or layer_idx >= s_all.size(1):
                raise IndexError(f"layer_idx={layer_idx} out of range for scaler shape {tuple(s_all.shape)}")

            s = s_all[:, layer_idx, :]  # (B, D)

            # Determine scaling strength by point
            strength = self.strength_o if point == "o_proj" else self.strength_d

            # Prepare output tensor
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(out):
                return output

            # Align dtype/device (scaler is usually produced by hypernet on same device)
            s = s.to(device=out.device, dtype=out.dtype)

            # Broadcast scaler over sequence dimension if needed
            # out: (B, T, D) => scale: (B, 1, D)
            # out: (B, D)    => scale: (B, D)
            if out.dim() == 3:
                scale = s.unsqueeze(1)
            elif out.dim() == 2:
                scale = s
            else:
                # Fallback for unexpected shapes; try to broadcast on last dim
                scale = s
                while scale.dim() < out.dim():
                    scale = scale.unsqueeze(1)

            # Compute multiplicative factor: 1 + strength * scale
            factor = 1.0 + strength * scale

            # Optional clamp for stability
            if self.clamp_min is not None or self.clamp_max is not None:
                factor = torch.clamp(
                    factor,
                    min=self.clamp_min if self.clamp_min is not None else -float("inf"),
                    max=self.clamp_max if self.clamp_max is not None else float("inf"),
                )

            # Apply scaling (avoid in-place to keep autograd safe)
            out_scaled = out * (factor + self.eps)

            # Preserve tuple structure if needed
            if isinstance(output, (tuple, list)):
                output = list(output)
                output[0] = out_scaled
                return type(output)(output) if isinstance(output, tuple) else output
            return out_scaled

        return hook
