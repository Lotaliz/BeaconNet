#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

readarray -t RATIOS < <(python - <<'PY'
from config import load_config
cfg = load_config()
for item in cfg.batch.prune_ratios:
    print(item)
PY
)

if [[ ${#RATIOS[@]} -eq 0 ]]; then
  echo "No prune ratios configured in cfg.batch.prune_ratios"
  exit 1
fi

TOTAL=${#RATIOS[@]}
INDEX=0
START_TS=$(date +%s)

for RATIO in "${RATIOS[@]}"; do
  INDEX=$((INDEX + 1))
  echo
  echo "============================================================"
  echo "[Sweep ${INDEX}/${TOTAL}] ratio=${RATIO}"
  echo "============================================================"
  bash scripts/wanda_sae_single.sh "${RATIO}"
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo
echo "All sweep jobs completed in ${ELAPSED}s"
