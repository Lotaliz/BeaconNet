#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

readarray -t CFG < <(python - <<'PY'
from config import load_config
cfg = load_config()
print(cfg.batch.python_bin)
print(cfg.sae_patch.sae_checkpoint_dir)
print(cfg.sae_patch.output_dir)
print(cfg.sae_patch.sweep_output_prefix)
print(cfg.sae_patch.sample_size)
print(cfg.sae_patch.patch_strength)
print(cfg.sae_patch.safe_threshold)
print(cfg.sae_patch.unsafe_threshold)
for ratio in cfg.batch.prune_ratios:
    print(f"RATIO={ratio}")
for mode in cfg.sae_patch.patch_modes:
    print(f"MODE={mode}")
PY
)

PYTHON_BIN="${CFG[0]}"
SAE_CHECKPOINT_DIR="${CFG[1]}"
PATCH_OUTPUT_ROOT="${CFG[2]}"
OUTPUT_PREFIX="${CFG[3]}"
SAMPLE_SIZE="${CFG[4]}"
PATCH_STRENGTH="${CFG[5]}"
SAFE_THRESHOLD="${CFG[6]}"
UNSAFE_THRESHOLD="${CFG[7]}"
RATIOS=()
PATCH_MODES=()
for ((i=8; i<${#CFG[@]}; i++)); do
  if [[ "${CFG[i]}" == RATIO=* ]]; then
    RATIOS+=("${CFG[i]#RATIO=}")
  elif [[ "${CFG[i]}" == MODE=* ]]; then
    PATCH_MODES+=("${CFG[i]#MODE=}")
  fi
done

MODEL_PATHS=()
for RATIO in "${RATIOS[@]}"; do
  MODEL_PATH="models/pruned/llama3.1-8B-Instruct-dpo-wanda-${RATIO}"
  if [[ -d "${MODEL_PATH}" ]]; then
    MODEL_PATHS+=("${MODEL_PATH}")
  else
    echo "Skip ratio=${RATIO}: model directory not found at ${MODEL_PATH}"
  fi
done

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  echo "No configured pruned models were found."
  exit 1
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

TOTAL=${#MODEL_PATHS[@]}
INDEX=0
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  INDEX=$((INDEX + 1))
  MODEL_NAME="$(basename "${MODEL_PATH}")"
  RATIO_TAG="${MODEL_NAME##*-}"
  OUTPUT_DIR="${PATCH_OUTPUT_ROOT}/${OUTPUT_PREFIX}${RATIO_TAG/./}"

  log "Patch ${INDEX}/${TOTAL}: ${MODEL_NAME}"
  echo "Output dir: ${OUTPUT_DIR}"

  CMD=("${PYTHON_BIN}" -m src.activation.sae_patch
    -m "${MODEL_PATH}"
    -a ""
    --sae-checkpoint-dir "${SAE_CHECKPOINT_DIR}"
    --sample-size "${SAMPLE_SIZE}"
    --patch-strength "${PATCH_STRENGTH}"
    --safe-threshold "${SAFE_THRESHOLD}"
    --unsafe-threshold "${UNSAFE_THRESHOLD}"
    --output-dir "${OUTPUT_DIR}")

  for MODE in "${PATCH_MODES[@]}"; do
    CMD+=(--patch-mode "${MODE}")
  done

  "${CMD[@]}"
done

log "All patch jobs completed"
