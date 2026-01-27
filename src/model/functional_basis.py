from __future__ import annotations

import torch
import torch.nn as nn


class FunctionalBasis(nn.Module):
    """
    Trainable per-layer functional basis vectors.

    Shape: (num_layers, num_basis, hidden_size)
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_basis: int,
        hidden_size: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_basis = int(num_basis)
        self.hidden_size = int(hidden_size)

        basis = torch.zeros(self.num_layers, self.num_basis, self.hidden_size)
        self.basis = nn.Parameter(basis)
        nn.init.normal_(self.basis, mean=0.0, std=init_std)

    def forward(self) -> torch.Tensor:
        return self.basis
