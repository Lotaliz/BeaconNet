#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<'EOF'
Usage: scripts/quantize_models.sh [options]

Batch-quantize the base Llama 3.1 8B model with AWQ, GPTQ, and SmoothQuant.
By default this script:
  - uses BeaconNet/models/llama3.1-8B-Instruct
  - disables the DPO LoRA adapter
  - saves outputs under BeaconNet/models/quantized
  - runs two precisions per method

Options:
  --python BIN          Python interpreter to use.
  --model PATH          Base model path. Default: config.quant.model_path
  --output-root PATH    Quantized model output root. Default: config.quant.output_root
  --with-adapter PATH   Optional LoRA adapter path. Default: disabled
  --awq-only            Run only AWQ jobs.
  --gptq-only           Run only GPTQ jobs.
  --smoothquant-only    Run only SmoothQuant jobs.
  --dry-run             Print commands without executing them.
  -h, --help            Show this help message.

Examples:
  bash scripts/quantize_models.sh
  bash scripts/quantize_models.sh --dry-run
  bash scripts/quantize_models.sh --awq-only
  bash scripts/quantize_models.sh --model /path/to/model --output-root /path/to/out
EOF
}

PYTHON_BIN="python"
BASE_MODEL_PATH=""
OUTPUT_ROOT=""
ADAPTER_PATH=""
RUN_AWQ=1
RUN_GPTQ=1
RUN_SMOOTHQUANT=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --model)
      BASE_MODEL_PATH="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --with-adapter)
      ADAPTER_PATH="${2:-}"
      shift 2
      ;;
    --awq-only)
      RUN_AWQ=1
      RUN_GPTQ=0
      RUN_SMOOTHQUANT=0
      shift
      ;;
    --gptq-only)
      RUN_AWQ=0
      RUN_GPTQ=1
      RUN_SMOOTHQUANT=0
      shift
      ;;
    --smoothquant-only)
      RUN_AWQ=0
      RUN_GPTQ=0
      RUN_SMOOTHQUANT=1
      shift
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
from config import load_config
cfg = load_config()
print(cfg.quant.model_path)
print(cfg.quant.output_root)
PY
)

if [[ -z "${BASE_MODEL_PATH}" ]]; then
  BASE_MODEL_PATH="${CFG[0]}"
fi
if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${CFG[1]}"
fi

mkdir -p "${OUTPUT_ROOT}"

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

add_common_args() {
  local -n _target=$1
  _target+=(--model "${BASE_MODEL_PATH}")
  _target+=(--adapter-path "${ADAPTER_PATH}")
}

# Two precisions per method.
# AWQ: 4-bit and 8-bit, shared group size g128.
# GPTQ: 4-bit and 8-bit, shared group size g128.
# SmoothQuant: W8A8 and W4A4, shared alpha a0.8.
AWQ_BITS=(4 8)
GPTQ_BITS=(4 8)
SMOOTHQUANT_SCHEMES=("W8A8" "W4A4")

if [[ "${RUN_AWQ}" == "1" ]]; then
  for bits in "${AWQ_BITS[@]}"; do
    cmd=("${PYTHON_BIN}" "${PROJECT_ROOT}/src/quant/awq_quantize.py" --bits "${bits}" --group-size 128)
    add_common_args cmd
    run_cmd "${cmd[@]}"
  done
fi

if [[ "${RUN_GPTQ}" == "1" ]]; then
  for bits in "${GPTQ_BITS[@]}"; do
    cmd=("${PYTHON_BIN}" "${PROJECT_ROOT}/src/quant/gptq_quantize.py" --bits "${bits}" --group-size 128)
    add_common_args cmd
    run_cmd "${cmd[@]}"
  done
fi

if [[ "${RUN_SMOOTHQUANT}" == "1" ]]; then
  for scheme in "${SMOOTHQUANT_SCHEMES[@]}"; do
    cmd=("${PYTHON_BIN}" "${PROJECT_ROOT}/src/quant/smoothquant_quantize.py" --alpha 0.8 --scheme "${scheme}")
    add_common_args cmd
    run_cmd "${cmd[@]}"
  done
fi

echo
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete. No quantization jobs were executed."
else
  echo "All quantization jobs completed. Outputs are saved under ${OUTPUT_ROOT}."
fi
