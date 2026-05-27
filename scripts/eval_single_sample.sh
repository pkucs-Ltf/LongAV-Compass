#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  bash scripts/eval_single_sample.sh SAMPLE_DIR [OUTPUT_DIR]

Example:
  bash run_eval_single.sh samples/t2av__advertising__sample_id runs/single_sample

Environment variables are the same as scripts/eval_batch_samples.sh, except
LONGAV_REUSE_ROOT is not used for single-sample evaluation.
USAGE
  exit 0
fi

SAMPLE_DIR="$(resolve_longav_path "$1")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$(resolve_longav_path "${2:-${LONGAV_RUNS_ROOT}/single_eval_${TIMESTAMP}}")"
API_KEYS="$(resolve_longav_path "${LONGAV_API_KEYS}")"

require_longav_dir "${SAMPLE_DIR}"
require_longav_file "${API_KEYS}"
mkdir -p "${OUTPUT_DIR}"

append_model_args
LONGAV_OPTIONAL_ARGS=()
if [[ "${LONGAV_SKIP_AUDIO:-1}" != "0" ]]; then
  LONGAV_OPTIONAL_ARGS+=(--skip-audio)
fi
if [[ "${LONGAV_ONLY_MISSING:-0}" == "1" ]]; then
  LONGAV_OPTIONAL_ARGS+=(--only-missing)
fi
if [[ -n "${LONGAV_QA_ROOT:-}" ]]; then
  LONGAV_OPTIONAL_ARGS+=(--qa-root "$(resolve_longav_path "${LONGAV_QA_ROOT}")")
fi
if [[ -n "${LONGAV_MAX_EVENTS:-}" ]]; then
  LONGAV_OPTIONAL_ARGS+=(--max-events "${LONGAV_MAX_EVENTS}")
fi
if [[ -n "${LONGAV_MAX_BOUNDARIES:-}" ]]; then
  LONGAV_OPTIONAL_ARGS+=(--max-boundaries "${LONGAV_MAX_BOUNDARIES}")
fi

print_longav_eval_env
echo "[LongAV] sample dir : ${SAMPLE_DIR}"
echo "[LongAV] output dir : ${OUTPUT_DIR}"

"${LONGAV_PYTHON}" -m longav_eval run-event-eval \
  --sample-dir "${SAMPLE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --api-keys "${API_KEYS}" \
  "${LONGAV_MODEL_ARGS[@]}" \
  "${LONGAV_OPTIONAL_ARGS[@]}"
