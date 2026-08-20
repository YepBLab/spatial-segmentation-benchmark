#!/usr/bin/env bash
set -euo pipefail

MANUAL_ONLY=0
if [[ "${1:-}" == "--manual-only" ]]; then
  MANUAL_ONLY=1
  shift
fi

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 [--manual-only] PROJECT_ROOT BENCHMARK_CONFIG MODEL_REGISTRY" >&2
  exit 64
fi

PROJECT_ROOT="$(cd "$1" && pwd)"
CONFIG="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
REGISTRY="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.matplotlib"

run_step() {
  local step="$1"
  shift
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ${step}"
  "$@" 2>&1 | tee "${LOG_DIR}/${step}.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE ${step}"
}

run_step 00_unit_tests "${PYTHON_BIN}" -m unittest discover -s "${REPO_ROOT}/tests" -v
if [[ "${MANUAL_ONLY}" -eq 1 ]]; then
  run_step 01_preflight "${PYTHON_BIN}" "${REPO_ROOT}/scripts/00_preflight_inventory.py" \
    --project-root "${PROJECT_ROOT}" --config "${CONFIG}" --registry "${REGISTRY}" \
    --skip-reimport-check
else
  run_step 01_preflight "${PYTHON_BIN}" "${REPO_ROOT}/scripts/00_preflight_inventory.py" \
    --project-root "${PROJECT_ROOT}" --config "${CONFIG}" --registry "${REGISTRY}"
fi
run_step 02_prepare_rois "${PYTHON_BIN}" "${REPO_ROOT}/scripts/01_prepare_common_rois.py" \
  --project-root "${PROJECT_ROOT}" --config "${CONFIG}" --registry "${REGISTRY}"
run_step 03_boundary_tolerance "${PYTHON_BIN}" "${REPO_ROOT}/scripts/02_calibrate_boundary_tolerance.py" \
  --project-root "${PROJECT_ROOT}" --config "${CONFIG}"
run_step 04_manual_metrics "${PYTHON_BIN}" "${REPO_ROOT}/scripts/03_compute_manual_reference_metrics.py" \
  --project-root "${PROJECT_ROOT}" --config "${CONFIG}" --registry "${REGISTRY}"
if [[ "${MANUAL_ONLY}" -eq 0 ]]; then
  run_step 05_global_medullary "${PYTHON_BIN}" "${REPO_ROOT}/scripts/04_compute_global_medullary_metrics.py" \
    --project-root "${PROJECT_ROOT}" --config "${CONFIG}" --registry "${REGISTRY}"
fi
run_step 06_stratified "${PYTHON_BIN}" "${REPO_ROOT}/scripts/05_compute_stratified_metrics.py" \
  --project-root "${PROJECT_ROOT}" --config "${CONFIG}" --registry "${REGISTRY}"
run_step 07_failure_cases "${PYTHON_BIN}" "${REPO_ROOT}/scripts/06_classify_error_cases.py" \
  --project-root "${PROJECT_ROOT}" --config "${CONFIG}"
if [[ "${MANUAL_ONLY}" -eq 1 ]]; then
  run_step 08_figures "${PYTHON_BIN}" "${REPO_ROOT}/scripts/07_render_benchmark_figures.py" \
    --project-root "${PROJECT_ROOT}" --config "${CONFIG}" --registry "${REGISTRY}" \
    --manual-only
else
  run_step 08_figures "${PYTHON_BIN}" "${REPO_ROOT}/scripts/07_render_benchmark_figures.py" \
    --project-root "${PROJECT_ROOT}" --config "${CONFIG}" --registry "${REGISTRY}"
fi
run_step 09_report "${PYTHON_BIN}" "${REPO_ROOT}/scripts/08_build_benchmark_report.py" \
  --project-root "${PROJECT_ROOT}" --config "${CONFIG}" --registry "${REGISTRY}"
run_step 10_validate "${PYTHON_BIN}" "${REPO_ROOT}/scripts/09_validate_benchmark.py" \
  --project-root "${PROJECT_ROOT}" --config "${CONFIG}" --registry "${REGISTRY}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PIPELINE COMPLETE"
