#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RATIO="${1:-}"
if [[ -z "${RATIO}" ]]; then
  echo "Usage: bash scripts/run_prune_safety_sae_once.sh <ratio>"
  echo "Example: bash scripts/run_prune_safety_sae_once.sh 0.6"
  exit 1
fi

readarray -t CFG < <(python - <<PY
from pathlib import Path
from config import load_config
cfg = load_config()
ratio = "${RATIO}"
ratio_tag = ratio[2:] if ratio.startswith("0.") else ratio.replace('.', '')
pruned_model_path = str(Path(cfg.prune.output_root) / f"{cfg.prune.output_name_prefix}-{ratio}")
safety_model_dir = pruned_model_path.replace('/', '__')
safety_output_dir = str(Path(cfg.batch.safety_output_root) / safety_model_dir)
sae_compare_output_dir = str(Path(cfg.batch.sae_compare_output_root) / f"{cfg.batch.sae_compare_output_prefix}{ratio_tag}")
sae_vis_output_dir = str(Path(cfg.batch.sae_vis_output_root) / f"{cfg.batch.sae_vis_output_prefix}{ratio_tag}")
print(cfg.batch.python_bin)
print(cfg.prune.model_path)
print(cfg.prune.lora_adapter_path)
print(pruned_model_path)
print(cfg.batch.baseline_model_path)
print(cfg.batch.baseline_lora_adapter_path)
print(cfg.sae_compare.baseline_safety_dir)
print(safety_output_dir)
print(cfg.sae_compare.sae_checkpoint_dir)
print(sae_compare_output_dir)
print(sae_vis_output_dir)
PY
)

PYTHON_BIN="${CFG[0]}"
BASE_MODEL_PATH="${CFG[1]}"
BASE_ADAPTER_PATH="${CFG[2]}"
PRUNED_MODEL_PATH="${CFG[3]}"
SAE_BASELINE_MODEL_PATH="${CFG[4]}"
SAE_BASELINE_ADAPTER_PATH="${CFG[5]}"
BASELINE_SAFETY_DIR="${CFG[6]}"
COMPRESSED_SAFETY_DIR="${CFG[7]}"
SAE_CHECKPOINT_DIR="${CFG[8]}"
SAE_COMPARE_OUTPUT_DIR="${CFG[9]}"
SAE_VIS_OUTPUT_DIR="${CFG[10]}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

log "Step 1/4: Wanda pruning, ratio=${RATIO}"
"${PYTHON_BIN}" -m src.prune.wanda \
  --model "${BASE_MODEL_PATH}" \
  --adapter-path "${BASE_ADAPTER_PATH}" \
  --sparsity-ratio "${RATIO}"

log "Step 2/4: safety2 evaluation for ${PRUNED_MODEL_PATH}"
"${PYTHON_BIN}" -m src.eval.safety2 \
  --model-path "${PRUNED_MODEL_PATH}" \
  --output-dir "$(dirname "${COMPRESSED_SAFETY_DIR}")"

log "Step 3/4: SAE compare for ratio=${RATIO}"
"${PYTHON_BIN}" -m src.activation.sae_compare \
  --baseline-model-path "${SAE_BASELINE_MODEL_PATH}" \
  --baseline-adapter-path "${SAE_BASELINE_ADAPTER_PATH}" \
  --baseline-safety-dir "${BASELINE_SAFETY_DIR}" \
  --compressed-model-path "${PRUNED_MODEL_PATH}" \
  --compressed-safety-dir "${COMPRESSED_SAFETY_DIR}" \
  --sae-checkpoint-dir "${SAE_CHECKPOINT_DIR}" \
  --output-dir "${SAE_COMPARE_OUTPUT_DIR}"

log "Step 4/4: SAE visualization for ratio=${RATIO}"
"${PYTHON_BIN}" -m src.activation.sae_vis \
  --compare-dir "${SAE_COMPARE_OUTPUT_DIR}" \
  --output-dir "${SAE_VIS_OUTPUT_DIR}"

log "Completed ratio=${RATIO}"
echo "Pruned model: ${PRUNED_MODEL_PATH}"
echo "Safety results: ${COMPRESSED_SAFETY_DIR}"
echo "SAE compare: ${SAE_COMPARE_OUTPUT_DIR}"
echo "SAE vis: ${SAE_VIS_OUTPUT_DIR}"
