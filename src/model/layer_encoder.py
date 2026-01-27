from __future__ import annotations

from typing import Dict, List
import torch
import torch.nn as nn


class LayerEmbeddingEncoder(nn.Module):
    """
    Encode per-layer structure using layer index, component type, and low-rank summaries.

    Inputs:
      summaries: {component_name: (B, L, R) or (L, R)}
    Outputs:
      layer_emb: (B, L, d_layer_emb)
      summary_pred: {component_name: (B, L, R)}
    """

    def __init__(
        self,
        *,
        num_layers: int,
        summary_dim: int,
        component_names: List[str],
        d_layer_emb: int = 512,
        d_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.summary_dim = int(summary_dim)
        self.d_layer_emb = int(d_layer_emb)
        self.component_names = list(component_names)

        self.layer_emb = nn.Embedding(self.num_layers, self.d_layer_emb)
        self.comp_emb = nn.Embedding(len(self.component_names), self.d_layer_emb)
        nn.init.normal_(self.layer_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.comp_emb.weight, mean=0.0, std=0.02)

        in_dim = self.d_layer_emb * 2 + self.summary_dim
        self.comp_encoder = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, self.d_layer_emb),
        )

        self.fuse = nn.Sequential(
            nn.Linear(self.d_layer_emb * 2, self.d_layer_emb),
            nn.GELU(),
            nn.Linear(self.d_layer_emb, self.d_layer_emb),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.d_layer_emb, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, self.summary_dim),
        )
        self._component_index = {name: i for i, name in enumerate(self.component_names)}

    def forward(
        self,
        summaries: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        summaries = self._normalize_inputs(summaries)

        first = next(iter(summaries.values()))
        B, L, _ = first.shape
        device = first.device
        dtype = first.dtype

        layer_idx = torch.arange(self.num_layers, device=device)
        layer_tok = self.layer_emb(layer_idx).to(dtype=dtype).unsqueeze(0).expand(B, -1, -1)

        comp_embs: List[torch.Tensor] = []
        summary_pred: Dict[str, torch.Tensor] = {}
        for name in self.component_names:
            summary = summaries[name]
            idx = self._component_index[name]
            comp_tok = self.comp_emb.weight[idx].to(dtype=dtype).view(1, 1, -1).expand(B, L, -1)
            x = torch.cat([layer_tok, comp_tok, summary], dim=-1)
            emb = self.comp_encoder(x)
            comp_embs.append(emb)
            summary_pred[name] = self.decoder(emb)

        comp_stack = torch.stack(comp_embs, dim=2)  # (B, L, C, d_layer_emb)
        layer_emb = comp_stack.mean(dim=2)
        return layer_emb, summary_pred

    def _normalize_inputs(
        self,
        summaries: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not summaries:
            raise ValueError("summaries must be a non-empty dict.")
        if set(summaries.keys()) != set(self.component_names):
            raise ValueError("summaries keys must match component_names.")

        normalized: Dict[str, torch.Tensor] = {}
        for name, summary in summaries.items():
            if summary.dim() == 2:
                summary = summary.unsqueeze(0)
            if summary.dim() != 3:
                raise ValueError("summary inputs must be (B, L, R) or (L, R).")
            if summary.size(1) != self.num_layers:
                raise ValueError("summary inputs must have num_layers on dim=1.")
            if summary.size(2) != self.summary_dim:
                raise ValueError("summary inputs must have summary_dim on dim=2.")
            normalized[name] = summary

        return normalized
