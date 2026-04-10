# !/bin/bash

python -m src.activation.sae_compare \
  --baseline-model-path models/llama3.1-8B-Instruct \
  --baseline-adapter-path models/aligned/llama3.1-8B-Instruct-dpo \
  --baseline-safety-dir data/safety2/models__aligned__llama3.1-8B-Instruct-dpo \
  --compressed-model-path models/pruned/llama3.1-8B-Instruct-dpo-wanda-0.6 \
  --compressed-safety-dir data/safety2/models__pruned__llama3.1-8B-Instruct-dpo-wanda-0.6 \
  --sae-checkpoint-dir data/activation/sae/checkpoints \
  --output-dir data/activation/sae_compare_prune06