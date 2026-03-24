#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v lm_eval >/dev/null 2>&1; then
  echo "lm_eval not found in PATH"
  exit 1
fi

readarray -t CFG < <(python - <<'PY'
from pathlib import Path
from config import load_config
cfg = load_config()
print(cfg.utility_eval.baseline_model_path)
print(cfg.utility_eval.pruned_model_root)
print(cfg.utility_eval.pruned_model_prefix)
print(cfg.utility_eval.output_root)
print(','.join(cfg.utility_eval.tasks))
print(cfg.utility_eval.device)
print(cfg.utility_eval.batch_size)
print('1' if cfg.utility_eval.apply_chat_template else '0')
for ratio in cfg.batch.prune_ratios:
    print(f'RATIO={ratio}')
PY
)

BASELINE_MODEL_PATH="${CFG[0]}"
PRUNED_MODEL_ROOT="${CFG[1]}"
PRUNED_MODEL_PREFIX="${CFG[2]}"
OUTPUT_ROOT="${CFG[3]}"
TASKS="${CFG[4]}"
DEVICE="${CFG[5]}"
BATCH_SIZE="${CFG[6]}"
APPLY_CHAT_TEMPLATE="${CFG[7]}"
RATIOS=()
for ((i=8; i<${#CFG[@]}; i++)); do
  RATIOS+=("${CFG[i]#RATIO=}")
done

mkdir -p "${OUTPUT_ROOT}"

run_eval() {
  local model_path="$1"
  local output_path="$2"
  local label="$3"

  echo
  echo "============================================================"
  echo "[lm_eval] ${label}"
  echo "model: ${model_path}"
  echo "output: ${output_path}"
  echo "tasks: ${TASKS}"
  echo "============================================================"

  local -a cmd=(
    lm_eval --model hf
    --model_args "pretrained=${model_path},tokenizer=${model_path}"
    --tasks "${TASKS}"
    --device "${DEVICE}"
    --batch_size "${BATCH_SIZE}"
    --output_path "${output_path}"
  )

  if [[ "${APPLY_CHAT_TEMPLATE}" == "1" ]]; then
    cmd+=(--apply_chat_template)
  fi

  "${cmd[@]}"
}

run_eval "${BASELINE_MODEL_PATH}" "${OUTPUT_ROOT}/dpo_baseline" "dpo_baseline"

for ratio in "${RATIOS[@]}"; do
  model_path="${PRUNED_MODEL_ROOT}/${PRUNED_MODEL_PREFIX}-${ratio}"
  ratio_tag="${ratio}"
  ratio_tag="${ratio_tag#0.}"
  if [[ ! -d "${model_path}" ]]; then
    echo "Skip ratio=${ratio}: model directory not found at ${model_path}"
    continue
  fi
  run_eval "${model_path}" "${OUTPUT_ROOT}/prune${ratio_tag}" "pruned_${ratio}"
done

echo
echo "All lm_eval jobs completed. Results saved under ${OUTPUT_ROOT}"
