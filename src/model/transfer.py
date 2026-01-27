from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from .functional_basis import FunctionalBasis
from .cross_attention import CrossModelAttention
from .hypernet import HypernetScaler, HypernetOutput


@dataclass
class TransferOutput:
    scalers: dict[str, torch.Tensor]
    guidance: torch.Tensor


class SafetyTransfer(nn.Module):
    """
    Teacher-student transfer module:
      - embed student/teacher activations
      - cross-attend to functional basis
      - produce per-layer scalers via hypernet
    """

    def __init__(
        self,
        *,
        num_layers: int,
        student_hidden_size: int,
        teacher_hidden_size: int,
        num_basis: int = 8,
        d_attn: int = 256,
        embed_dim: int | None = None,
        d_layer_emb: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.student_hidden_size = int(student_hidden_size)
        self.teacher_hidden_size = int(teacher_hidden_size)
        self.num_basis = int(num_basis)
        self.d_attn = int(d_attn)
        self.embed_dim = int(embed_dim) if embed_dim is not None else self.student_hidden_size

        self.student_embed = nn.Linear(self.student_hidden_size * 2, self.embed_dim)
        self.teacher_embed = nn.Linear(self.teacher_hidden_size * 2, self.embed_dim)

        self.basis = FunctionalBasis(
            num_layers=self.num_layers,
            num_basis=self.num_basis,
            hidden_size=self.embed_dim,
        )

        self.cross_attn = CrossModelAttention(
            hidden_size=self.embed_dim,
            d_attn=self.d_attn,
            dropout=dropout,
        )

        self.hypernet = HypernetScaler(
            num_layers=self.num_layers,
            hidden_size=self.student_hidden_size,
            d_layer_emb=d_layer_emb,
            d_guidance=self.d_attn,
            d_model=d_model,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
        )

    def forward(
        self,
        *,
        student_o: torch.Tensor,
        student_d: torch.Tensor,
        teacher_o: torch.Tensor,
        teacher_d: torch.Tensor,
    ) -> TransferOutput:
        if student_o.shape != student_d.shape:
            raise ValueError("student_o and student_d must have the same shape (B, L, D).")
        if teacher_o.shape != teacher_d.shape:
            raise ValueError("teacher_o and teacher_d must have the same shape (B, L, D).")

        s_embed = self.student_embed(torch.cat([student_o, student_d], dim=-1))
        t_embed = self.teacher_embed(torch.cat([teacher_o, teacher_d], dim=-1))

        basis = self.basis()
        z = self.cross_attn(student_embed=s_embed, teacher_embed=t_embed, basis=basis)

        hn_out: HypernetOutput = self.hypernet(guidance=z)
        return TransferOutput(scalers=hn_out.scalers, guidance=z)

    def encoder_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        params += list(self.student_embed.parameters())
        params += list(self.teacher_embed.parameters())
        params += list(self.basis.parameters())
        params += list(self.cross_attn.parameters())
        return params

    def hypernet_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        params += list(self.hypernet.parameters())
        return params
