from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn


@dataclass
class HypernetOutput:
    """Container for hypernet scalers."""
    scalers: Dict[str, torch.Tensor]  # keys: "o_proj", "down_proj"; values: (B, L, D)


class LayerEmbedding(nn.Module):
    """
    Learnable per-layer embeddings for SLM blocks.
    """

    def __init__(self, num_layers: int, d_emb: int = 512) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.d_emb = int(d_emb)

        self.emb = nn.Embedding(self.num_layers, self.d_emb)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Produce (B, L, d_emb)
        idx = torch.arange(self.num_layers, device=device)
        e = self.emb(idx).to(dtype=dtype)  # (L, d_emb)
        return e.unsqueeze(0).expand(batch_size, -1, -1)


class HypernetScaler(nn.Module):
    """
    Minimal hypernet:
      - layer embeddings as tokens (sequence length = num_layers)
      - optional guidance vector concatenated to each layer token
      - cross-layer MHFA
      - two heads to predict (B, L, hidden_size) scalers for o_proj and down_proj
    """

    def __init__(
        self,
        *,
        num_layers: int,
        hidden_size: int,
        d_layer_emb: int = 512,
        d_guidance: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.d_layer_emb = int(d_layer_emb)
        self.d_guidance = int(d_guidance)
        self.d_model = int(d_model)

        self.layer_emb = LayerEmbedding(num_layers=self.num_layers, d_emb=self.d_layer_emb)

        self.in_proj = nn.Linear(self.d_layer_emb + self.d_guidance, self.d_model, bias=True)

        self.mhfa = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=n_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(self.d_model * mlp_ratio, self.d_model),
        )

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        self.head_o = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.hidden_size),
        )
        self.head_d = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.hidden_size),
        )

        # Initialize final projections to near-zero output for stability
        nn.init.zeros_(self.head_o[-1].weight)
        nn.init.zeros_(self.head_o[-1].bias)
        nn.init.zeros_(self.head_d[-1].weight)
        nn.init.zeros_(self.head_d[-1].bias)

    def forward(
        self,
        *,
        guidance: Optional[torch.Tensor] = None,  # (B, d_guidance)
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> HypernetOutput:
        # Resolve batch/device/dtype
        if guidance is None:
            if batch_size is None or device is None or dtype is None:
                raise ValueError("If guidance is None, batch_size/device/dtype must be provided.")
            B = int(batch_size)
            g = torch.zeros((B, self.d_guidance), device=device, dtype=dtype)
        else:
            if guidance.dim() != 2 or guidance.size(-1) != self.d_guidance:
                raise ValueError(f"guidance must be (B, {self.d_guidance}), got {tuple(guidance.shape)}")
            g = guidance
            B = g.size(0)
            device = g.device
            dtype = g.dtype

        # Build layer tokens and concatenate guidance
        e = self.layer_emb(batch_size=B, device=device, dtype=dtype)  # (B, L, d_layer_emb)
        g_tok = g.unsqueeze(1).expand(-1, self.num_layers, -1)        # (B, L, d_guidance)
        x = torch.cat([e, g_tok], dim=-1)                             # (B, L, d_layer_emb+d_guidance)

        # Project to MHFA space
        x = self.in_proj(x)                                           # (B, L, d_model)

        # Cross-layer attention + FFN blocks
        x1 = self.ln1(x)
        attn_out, _ = self.mhfa(x1, x1, x1, need_weights=False)        # (B, L, d_model)
        x = x + attn_out

        x2 = self.ln2(x)
        x = x + self.ffn(x2)                                          # (B, L, d_model)

        # Predict scalers for the two injection points
        s_o = self.head_o(x)                                          # (B, L, hidden_size)
        s_d = self.head_d(x)                                          # (B, L, hidden_size)

        return HypernetOutput(scalers={"o_proj": s_o, "down_proj": s_d})
