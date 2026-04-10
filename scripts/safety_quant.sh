#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<'EOF'
Usage: scripts/safety_quant.sh [options]

Evaluate the six quantized models with the XGuard-based safety pipeline.
This script invokes src/eval/safety2.py, which uses XGuard for response
safety scoring and saves results under data/safety2 by default.

Default model order:
  1. llama3.1-8B-Instruct-dpo-awq-4bit-g128
  2. llama3.1-8B-Instruct-dpo-awq-8bit-g128
  3. llama3.1-8B-Instruct-dpo-gptq-4bit-g128
  4. llama3.1-8B-Instruct-dpo-gptq-8bit-g128
  5. llama3.1-8B-Instruct-dpo-smoothquant-w4a4-a0.8
  6. llama3.1-8B-Instruct-dpo-smoothquant-w8a8-a0.8

Options:
  --python BIN          Python interpreter to use.
  --models-root PATH    Quantized model root. Default: config.quant.output_root
  --output-dir PATH     Safety evaluation output dir. Default: data/safety2
  --sample-size N       Override safety2 sample size.
  --only-model PATH     Evaluate only one specific quantized model.
  --dry-run             Print commands without executing them.
  -h, --help            Show this help message.

Examples:
  bash scripts/safety_quant.sh
  bash scripts/safety_quant.sh --dry-run
  bash scripts/safety_quant.sh --only-model models/quantized/llama3.1-8B-Instruct-dpo-awq-4bit-g128
EOF
}

PYTHON_BIN="python"
MODELS_ROOT=""
OUTPUT_DIR=""
SAMPLE_SIZE=""
ONLY_MODEL=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --models-root)
      MODELS_ROOT="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="${2:-}"
      shift 2
      ;;
    --only-model)
      ONLY_MODEL="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

readarray -t CFG < <("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from config import load_config
cfg = load_config()
print(cfg.batch.python_bin)
print(cfg.quant.output_root)
print(Path(cfg.data_root) / "safety2")
print(cfg.safety_sample_size)
PY
)

if [[ "${PYTHON_BIN}" == "python" && -n "${CFG[0]}" ]]; then
  PYTHON_BIN="${CFG[0]}"
fi
if [[ -z "${MODELS_ROOT}" ]]; then
  MODELS_ROOT="${CFG[1]}"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${CFG[2]}"
fi
if [[ -z "${SAMPLE_SIZE}" ]]; then
  SAMPLE_SIZE="${CFG[3]}"
fi

mkdir -p "${OUTPUT_DIR}"

run_cmd() {
  local -a cmd=("$@")
  echo
  printf '>>'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "${cmd[@]}"
}

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

MODEL_PATHS=()
if [[ -n "${ONLY_MODEL}" ]]; then
  MODEL_PATHS=("${ONLY_MODEL}")
else
  DEFAULT_MODEL_NAMES=(
    "llama3.1-8B-Instruct-dpo-awq-4bit-g128"
    "llama3.1-8B-Instruct-dpo-awq-8bit-g128"
    "llama3.1-8B-Instruct-dpo-gptq-4bit-g128"
    "llama3.1-8B-Instruct-dpo-gptq-8bit-g128"
    "llama3.1-8B-Instruct-dpo-smoothquant-w4a4-a0.8"
    "llama3.1-8B-Instruct-dpo-smoothquant-w8a8-a0.8"
  )
  for model_name in "${DEFAULT_MODEL_NAMES[@]}"; do
    MODEL_PATHS+=("${MODELS_ROOT}/${model_name}")
  done
fi

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  echo "No quantized models were found."
  exit 1
fi

TOTAL=${#MODEL_PATHS[@]}
INDEX=0
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Skip missing model directory: ${MODEL_PATH}"
    continue
  fi

  INDEX=$((INDEX + 1))
  MODEL_NAME="$(basename "${MODEL_PATH}")"
  log "Safety eval ${INDEX}/${TOTAL}: ${MODEL_NAME}"

  CMD=("${PYTHON_BIN}" -m src.eval.safety2
    --model-path "${MODEL_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --sample-size "${SAMPLE_SIZE}")

  run_cmd "${CMD[@]}"
done

echo
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete. No safety evaluation jobs were executed."
else
  echo "All quantized safety evaluation jobs completed. Results saved under ${OUTPUT_DIR}."
fi
