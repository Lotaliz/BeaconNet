#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

readarray -t CFG < <(python - <<'PY'
from pathlib import Path
from config import load_config
cfg = load_config()
print(cfg.batch.python_bin)
print(cfg.sae_patch.output_dir)
print(cfg.sae_patch.sweep_model_glob)
print(cfg.sae_patch.sweep_output_prefix)
print(cfg.sae_patch.sample_size)
print(cfg.sae_patch.feature_top_k)
print(cfg.sae_patch.patch_strength)
for mode in cfg.sae_patch.patch_modes:
    print(f"MODE={mode}")
PY
)

PYTHON_BIN="${CFG[0]}"
PATCH_OUTPUT_ROOT="${CFG[1]}"
MODEL_GLOB="${CFG[2]}"
OUTPUT_PREFIX="${CFG[3]}"
SAMPLE_SIZE="${CFG[4]}"
FEATURE_TOP_K="${CFG[5]}"
PATCH_STRENGTH="${CFG[6]}"
PATCH_MODES=()
for ((i=7; i<${#CFG[@]}; i++)); do
  PATCH_MODES+=("${CFG[i]#MODE=}")
done

readarray -t MODEL_PATHS < <(find models/pruned -maxdepth 1 -mindepth 1 -type d -name "${MODEL_GLOB}" | sort)

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  echo "No pruned models matched: models/pruned/${MODEL_GLOB}"
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
    --sample-size "${SAMPLE_SIZE}"
    --feature-top-k "${FEATURE_TOP_K}"
    --patch-strength "${PATCH_STRENGTH}"
    --output-dir "${OUTPUT_DIR}")

  for MODE in "${PATCH_MODES[@]}"; do
    CMD+=(--patch-mode "${MODE}")
  done

  "${CMD[@]}"
done

log "All patch jobs completed"
