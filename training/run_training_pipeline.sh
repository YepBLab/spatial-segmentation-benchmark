#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 PROJECT_ROOT TRAINING_MANIFEST" >&2
  exit 64
fi

PROJECT_ROOT="$(mkdir -p "$1" && cd "$1" && pwd)"
MANIFEST="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-cpsam_v2_finetuned_all_regions}"
N_EPOCHS="${N_EPOCHS:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BSIZE="${BSIZE:-256}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-9}"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

run_step() {
  local step="$1"
  shift
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ${step}"
  "$@" 2>&1 | tee "${LOG_DIR}/${step}.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE ${step}"
}

PREPARE_ARGS=(
  --project-root "${PROJECT_ROOT}"
  --manifest "${MANIFEST}"
)
if [[ "${OVERWRITE_PREPARED_DATA:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--overwrite)
fi

run_step 01_prepare_training_data \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_training_data.py" "${PREPARE_ARGS[@]}"

run_step 02_preflight \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight_training.py" \
  --project-root "${PROJECT_ROOT}" \
  --pretrained-model cpsam_v2 \
  --require-cuda

echo "NOTICE: this reproduction uses all regions because manual labels are limited."
echo "With sufficient labels, reserve independent regions/specimens for validation."

run_step 03_train \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_cpsam_v2.py" \
  --project-root "${PROJECT_ROOT}" \
  --pretrained-model cpsam_v2 \
  --model-name "${MODEL_NAME}" \
  --device cuda \
  --n-epochs "${N_EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --batch-size "${BATCH_SIZE}" \
  --bsize "${BSIZE}"

run_step 04_validate_model \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_trained_model.py" \
  --project-root "${PROJECT_ROOT}" \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --bsize "${BSIZE}"

run_step 05_plot_training_history \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_training_history.py" \
  --loss-csv "${PROJECT_ROOT}/training/final/losses.csv" \
  --output "${PROJECT_ROOT}/training/final/training_history_dark.png" \
  --smooth-window "${SMOOTH_WINDOW}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TRAINING PIPELINE COMPLETE"
