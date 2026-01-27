import math
import torch
import torch.nn as nn


class CrossModelAttention(nn.Module):
    """
    Cross-model attention over a per-layer functional basis.

    Inputs:
      student_embed: (B, L, D)
      teacher_embed: (B, L, D)
      basis:         (L, R, D)

    Output:
      z: (B, L, d_attn)
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        d_attn: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.d_attn = int(d_attn)

        self.q_proj = nn.Linear(self.hidden_size, self.d_attn)
        self.k_proj = nn.Linear(self.hidden_size, self.d_attn)
        self.qk_proj = nn.Linear(self.d_attn * 2, self.d_attn)

        self.basis_key = nn.Linear(self.hidden_size, self.d_attn)
        self.basis_val = nn.Linear(self.hidden_size, self.d_attn)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        *,
        student_embed: torch.Tensor,
        teacher_embed: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        if student_embed.dim() != 3 or teacher_embed.dim() != 3:
            raise ValueError("student_embed and teacher_embed must be (B, L, D).")
        if basis.dim() != 3:
            raise ValueError("basis must be (L, R, D).")

        q = self.q_proj(student_embed)
        k = self.k_proj(teacher_embed)
        qk = self.qk_proj(torch.cat([q, k], dim=-1))

        basis_k = self.basis_key(basis)
        basis_v = self.basis_val(basis)

        scale = 1.0 / math.sqrt(self.d_attn)
        scores = torch.einsum("bld,lrd->blr", qk, basis_k) * scale
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        z = torch.einsum("blr,lrd->bld", weights, basis_v)
        return z
