#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SAFE_SAMPLE_SIZE="${1:-200}"
UNSAFE_SAMPLE_SIZE="${2:-200}"
RUN_NAME="${3:-mixed_prompt_last_s${SAFE_SAMPLE_SIZE}_u${UNSAFE_SAMPLE_SIZE}}"

readarray -t CFG < <(python - <<'PY'
from pathlib import Path
from config import load_config
cfg = load_config()
print(cfg.batch.python_bin)
print(cfg.model_path)
print(cfg.align.output_dir)
print(cfg.align.train_dataset_path)
print(Path(cfg.project_root) / "datasets" / "wildjb" / "eval" / "eval.tsv")
print(cfg.data_root)
PY
)

PYTHON_BIN="${CFG[0]}"
BASE_MODEL_PATH="${CFG[1]}"
ALIGNED_MODEL_PATH="${CFG[2]}"
PKU_DATASET_PATH="${CFG[3]}"
WILDJB_DATASET_PATH="${CFG[4]}"
DATA_ROOT="${CFG[5]}"

SAFETY_OUTPUT_ROOT="${DATA_ROOT}/safety2_${RUN_NAME}"
SAE_OUTPUT_ROOT="${DATA_ROOT}/activation/${RUN_NAME}"
SAE_MANIFEST_PATH="${SAE_OUTPUT_ROOT}/examples.jsonl"
SAE_ACTIVATION_DIR="${SAE_OUTPUT_ROOT}/activations"
SAE_CHECKPOINT_DIR="${SAE_OUTPUT_ROOT}/checkpoints"
SAFE_DATASET_NAME="pku_safe_rlhf_${SAFE_SAMPLE_SIZE}"
UNSAFE_DATASET_NAME="wildjb_${UNSAFE_SAMPLE_SIZE}"
MODEL_OUTPUT_DIRNAME="$("${PYTHON_BIN}" - <<PY
model_path = r"""${ALIGNED_MODEL_PATH}"""
print(model_path.replace("/", "__"))
PY
)"
SAFETY_RESULT_DIR="${SAFETY_OUTPUT_ROOT}/${MODEL_OUTPUT_DIRNAME}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

if [[ ! -f "${PKU_DATASET_PATH}" ]]; then
  echo "PKU dataset not found: ${PKU_DATASET_PATH}"
  exit 1
fi

if [[ ! -f "${WILDJB_DATASET_PATH}" ]]; then
  echo "WildJB dataset not found: ${WILDJB_DATASET_PATH}"
  exit 1
fi

log "Run name: ${RUN_NAME}"
echo "Safe dataset: ${PKU_DATASET_PATH}"
echo "Unsafe dataset: ${WILDJB_DATASET_PATH}"
echo "Safe sample size: ${SAFE_SAMPLE_SIZE}"
echo "Unsafe sample size: ${UNSAFE_SAMPLE_SIZE}"
echo "Safety output: ${SAFETY_OUTPUT_ROOT}"
echo "SAE output: ${SAE_OUTPUT_ROOT}"

mkdir -p "${SAFETY_OUTPUT_ROOT}" "${SAE_OUTPUT_ROOT}"

log "Step 1/5: Evaluate aligned model on PKU-SafeRLHF prompts with XGuard supervision"
"${PYTHON_BIN}" -m src.eval.safety2 \
  --model-path "${ALIGNED_MODEL_PATH}" \
  --dataset-path "${PKU_DATASET_PATH}" \
  --dataset-name "${SAFE_DATASET_NAME}" \
  --sample-size "${SAFE_SAMPLE_SIZE}" \
  --output-dir "${SAFETY_OUTPUT_ROOT}"

log "Step 2/5: Evaluate aligned model on WildJB jailbreak prompts with XGuard supervision"
"${PYTHON_BIN}" -m src.eval.safety2 \
  --model-path "${ALIGNED_MODEL_PATH}" \
  --dataset-path "${WILDJB_DATASET_PATH}" \
  --dataset-name "${UNSAFE_DATASET_NAME}" \
  --sample-size "${UNSAFE_SAMPLE_SIZE}" \
  --output-dir "${SAFETY_OUTPUT_ROOT}"

log "Step 3/5: Build SAE manifest from mixed safety2 outputs"
"${PYTHON_BIN}" -m src.activation.prepare_sae_dataset \
  --source-dir "${SAFETY_RESULT_DIR}" \
  --output "${SAE_MANIFEST_PATH}"

log "Step 4/5: Collect prompt-last-token activations for SAE training"
"${PYTHON_BIN}" -m src.activation.collect_sae_activations \
  --model-path "${BASE_MODEL_PATH}" \
  --adapter-path "${ALIGNED_MODEL_PATH}" \
  --manifest "${SAE_MANIFEST_PATH}" \
  --output-dir "${SAE_ACTIVATION_DIR}"

log "Step 5/5: Train SAE checkpoints on mixed safe/unsafe data"
"${PYTHON_BIN}" -m src.activation.train_sae \
  --dataset-dir "${SAE_ACTIVATION_DIR}" \
  --output-dir "${SAE_CHECKPOINT_DIR}"

log "Completed mixed SAE retraining pipeline"
echo "Safety results: ${SAFETY_RESULT_DIR}"
echo "SAE manifest: ${SAE_MANIFEST_PATH}"
echo "Activation dataset: ${SAE_ACTIVATION_DIR}"
echo "SAE checkpoints: ${SAE_CHECKPOINT_DIR}"
