from dataclasses import dataclass
from typing import Dict, Optional, Literal
import torch

PoolingMode = Literal["last", "mean"]

@dataclass
class AggregatedActivations:
    """Container for aggregated activations."""
    x: torch.Tensor  # (B, L, D)
    pooling: str
    num_layers: int
    hidden_size: int


class ActivationAggregator:
    """
    Convert per-layer activation dicts into a single (B, L, D) tensor.
    """
    def __init__(self, *, strict: bool = True) -> None:
        # If strict, missing layers raise; otherwise fill with zeros
        self.strict = strict

    def dict_to_blD(
        self,
        acts: Dict[int, torch.Tensor],
        *,
        num_layers: int,
        pooling: PoolingMode = "last",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> AggregatedActivations:
        # Validate presence and infer (B, D) shape from first available layer
        if len(acts) == 0:
            raise ValueError("Empty activation dict.")

        ref_layer = min(acts.keys())
        ref = acts[ref_layer]
        if not torch.is_tensor(ref):
            raise TypeError(f"Activation for layer {ref_layer} is not a Tensor.")

        ref_bd = self._pool_to_bd(ref, pooling=pooling)
        B, D = ref_bd.shape

        dev = device if device is not None else ref_bd.device
        dt = dtype if dtype is not None else ref_bd.dtype

        # Allocate output
        out = torch.zeros((B, num_layers, D), device=dev, dtype=dt)

        # Fill per-layer activations
        for l in range(num_layers):
            if l not in acts:
                if self.strict:
                    raise KeyError(f"Missing layer {l} in activation dict (keys={sorted(acts.keys())}).")
                continue

            t = acts[l]
            bd = self._pool_to_bd(t, pooling=pooling)

            if bd.shape != (B, D):
                raise ValueError(
                    f"Layer {l} pooled shape mismatch: got {tuple(bd.shape)}, expected {(B, D)}."
                )

            out[:, l, :] = bd.to(device=dev, dtype=dt)

        return AggregatedActivations(x=out, pooling=pooling, num_layers=num_layers, hidden_size=D)

    def _pool_to_bd(self, x: torch.Tensor, *, pooling: PoolingMode) -> torch.Tensor:
        # Convert (B, T, D) -> (B, D) by pooling, or pass through if already (B, D)
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            if pooling == "last":
                return x[:, -1, :]
            if pooling == "mean":
                return x.mean(dim=1)
            raise ValueError(f"Unsupported pooling: {pooling}")
        raise ValueError(f"Unsupported activation tensor rank: {x.dim()} (expected 2 or 3).")
