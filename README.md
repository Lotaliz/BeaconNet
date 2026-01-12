# HyperNet-based Safety Alignment Transfer

This repository implements a hypernetwork-conditioned framework
for transferring safety alignment from large language models (LLMs)
to small language models (SLMs) under model compression.

Key ideas:
- Do NOT directly copy LLM parameters;
- Learn functional basis vectors that represent parameter effects;
- Use cross-model attention (Q=SLM activations, K=LLM activations, V=functional basis);
- Concatenate attention outputs with layer embeddings as hypernet inputs;
- Hypernetwork generates controllable parameter modulation (scalers / deltas).

Highlights:
- Parameter-level safety alignment
- Low-rank and interpretable structure
- Compatible with distillation and compression settings

Structure:
data/        datasets and preprocessing
models/      functional basis, attention, hypernet
training/    training loop and losses
utils/       configs and evaluation
