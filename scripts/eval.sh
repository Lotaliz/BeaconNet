#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SHOW_HELP=0
RUN_BASELINE=1
RUN_PRUNED=1
RUN_QUANTIZED=0
ONLY_QUANTIZED=0
ONLY_MODEL_PATH=""
ONLY_MODEL_LABEL=""

usage() {
  cat <<'EOF'
Usage: scripts/eval.sh [options]

Options:
  --with-quantized       Also evaluate models under models/quantized.
  --quantized-only       Evaluate only quantized models.
  --skip-baseline        Skip the baseline model.
  --skip-pruned          Skip pruned models.
  --only-model PATH      Evaluate only one specific model directory.
  --label LABEL          Optional label for --only-model output dir.
  -h, --help             Show this help message.

Examples:
  scripts/eval.sh
  scripts/eval.sh --with-quantized
  scripts/eval.sh --quantized-only
  scripts/eval.sh --only-model models/quantized/llama3.1-8B-Instruct-dpo-awq-4bit-g128 --label awq4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-quantized)
      RUN_QUANTIZED=1
      shift
      ;;
    --quantized-only)
      RUN_BASELINE=0
      RUN_PRUNED=0
      RUN_QUANTIZED=1
      ONLY_QUANTIZED=1
      shift
      ;;
    --skip-baseline)
      RUN_BASELINE=0
      shift
      ;;
    --skip-pruned)
      RUN_PRUNED=0
      shift
      ;;
    --only-model)
      ONLY_MODEL_PATH="${2:-}"
      if [[ -z "${ONLY_MODEL_PATH}" ]]; then
        echo "--only-model requires a model path"
        exit 1
      fi
      shift 2
      ;;
    --label)
      ONLY_MODEL_LABEL="${2:-}"
      if [[ -z "${ONLY_MODEL_LABEL}" ]]; then
        echo "--label requires a value"
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      SHOW_HELP=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "${SHOW_HELP}" == "1" ]]; then
  usage
  exit 0
fi

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
print(cfg.quant.output_root)
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
QUANTIZED_MODEL_ROOT="${CFG[3]}"
OUTPUT_ROOT="${CFG[4]}"
TASKS="${CFG[5]}"
DEVICE="${CFG[6]}"
BATCH_SIZE="${CFG[7]}"
APPLY_CHAT_TEMPLATE="${CFG[8]}"
RATIOS=()
for ((i=9; i<${#CFG[@]}; i++)); do
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

normalize_label() {
  local raw="$1"
  raw="${raw//\//__}"
  raw="${raw// /_}"
  echo "${raw}"
}

run_quantized_models() {
  if [[ ! -d "${QUANTIZED_MODEL_ROOT}" ]]; then
    echo "Skip quantized models: directory not found at ${QUANTIZED_MODEL_ROOT}"
    return
  fi

  local found_any=0
  while IFS= read -r model_path; do
    [[ -z "${model_path}" ]] && continue
    found_any=1
    local model_name
    local label
    model_name="$(basename "${model_path}")"
    label="quant_$(normalize_label "${model_name}")"
    run_eval "${model_path}" "${OUTPUT_ROOT}/${label}" "${label}"
  done < <(find "${QUANTIZED_MODEL_ROOT}" -maxdepth 1 -mindepth 1 -type d | sort)

  if [[ "${found_any}" == "0" ]]; then
    echo "Skip quantized models: no model directories found under ${QUANTIZED_MODEL_ROOT}"
  fi
}

if [[ -n "${ONLY_MODEL_PATH}" ]]; then
  if [[ ! -d "${ONLY_MODEL_PATH}" ]]; then
    echo "Model directory not found: ${ONLY_MODEL_PATH}"
    exit 1
  fi
  if [[ -z "${ONLY_MODEL_LABEL}" ]]; then
    ONLY_MODEL_LABEL="$(normalize_label "$(basename "${ONLY_MODEL_PATH}")")"
  fi
  run_eval "${ONLY_MODEL_PATH}" "${OUTPUT_ROOT}/${ONLY_MODEL_LABEL}" "${ONLY_MODEL_LABEL}"
  echo
  echo "Single-model lm_eval job completed. Results saved under ${OUTPUT_ROOT}/${ONLY_MODEL_LABEL}"
  exit 0
fi

if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_eval "${BASELINE_MODEL_PATH}" "${OUTPUT_ROOT}/dpo_baseline" "dpo_baseline"
fi

if [[ "${RUN_PRUNED}" == "1" ]]; then
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
fi

if [[ "${RUN_QUANTIZED}" == "1" ]]; then
  run_quantized_models
fi

echo
echo "All lm_eval jobs completed. Results saved under ${OUTPUT_ROOT}"
