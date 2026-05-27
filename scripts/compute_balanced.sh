#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  bash scripts/compute_balanced.sh INPUT_WIDE_CSV [OUTPUT_DIR]

Example:
  bash run_compute_scores.sh /path/to/best_sample_model_metrics_wide.csv runs/paper

Environment:
  LONGAV_ALLOW_PARTIAL=1  Average available components instead of requiring all six shared video metrics.
USAGE
  exit 0
fi

INPUT_CSV="$(resolve_longav_path "$1")"
OUTPUT_DIR="$(resolve_longav_path "${2:-${LONGAV_RUNS_ROOT}/paper}")"
SAMPLE_OUTPUT="${OUTPUT_DIR}/balanced_sample_model_scores.csv"
GROUP_OUTPUT="${OUTPUT_DIR}/scenario_balanced_scores.csv"

require_longav_file "${INPUT_CSV}"
mkdir -p "${OUTPUT_DIR}"

ARGS=()
if [[ "${LONGAV_ALLOW_PARTIAL:-0}" == "1" ]]; then
  ARGS+=(--allow-partial)
fi

echo "[LongAV] input      : ${INPUT_CSV}"
echo "[LongAV] sample out : ${SAMPLE_OUTPUT}"
echo "[LongAV] group out  : ${GROUP_OUTPUT}"

"${LONGAV_PYTHON}" -m longav_eval compute-balanced \
  --input "${INPUT_CSV}" \
  --output "${SAMPLE_OUTPUT}" \
  --group-output "${GROUP_OUTPUT}" \
  --group-by task category model \
  "${ARGS[@]}"
