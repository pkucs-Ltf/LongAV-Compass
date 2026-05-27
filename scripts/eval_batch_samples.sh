#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  bash scripts/eval_batch_samples.sh SAMPLE_ROOT [OUTPUT_ROOT] [SAMPLE_GLOB]

Examples:
  bash run_eval_batch.sh samples runs/example '*__*'
  LONGAV_MODEL_NAMES="Kling Seedance2" bash run_eval_batch.sh samples runs/v2av 'v2av__*'
  LONGAV_SKIP_AUDIO=0 bash run_eval_batch.sh samples runs/full '*__*'

Environment:
  LONGAV_API_KEYS       API key YAML. Default: configs/api_keys.yaml
  LONGAV_QA_ROOT        Optional external fixed-QA dataset root
  LONGAV_MODEL_NAMES    Space-separated model aliases to evaluate
  LONGAV_MAX_WORKERS    Concurrent sample workers. Default: 8
  LONGAV_SKIP_AUDIO     1 skips audio diagnostics, 0 enables them. Default: 1
  LONGAV_ONLY_MISSING   1 computes only missing metrics
  LONGAV_REUSE_ROOT     Previous batch output root for reuse/resume
  LONGAV_MAX_EVENTS     Optional smoke-test event limit
  LONGAV_MAX_BOUNDARIES Optional smoke-test boundary limit
USAGE
  exit 0
fi

SAMPLE_ROOT="$(resolve_longav_path "$1")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="$(resolve_longav_path "${2:-${LONGAV_RUNS_ROOT}/longav_eval_${TIMESTAMP}}")"
SAMPLE_GLOB="${3:-${LONGAV_SAMPLE_GLOB:-*__*}}"
MAX_WORKERS="${LONGAV_MAX_WORKERS:-8}"
API_KEYS="$(resolve_longav_path "${LONGAV_API_KEYS}")"

require_longav_dir "${SAMPLE_ROOT}"
require_longav_file "${API_KEYS}"
mkdir -p "${OUTPUT_ROOT}"

append_model_args
append_optional_eval_args
print_longav_eval_env
echo "[LongAV] sample root: ${SAMPLE_ROOT}"
echo "[LongAV] sample glob: ${SAMPLE_GLOB}"
echo "[LongAV] output root: ${OUTPUT_ROOT}"

"${LONGAV_PYTHON}" -m longav_eval run-event-eval-batch \
  --sample-root "${SAMPLE_ROOT}" \
  --sample-glob "${SAMPLE_GLOB}" \
  --output-root "${OUTPUT_ROOT}" \
  --api-keys "${API_KEYS}" \
  --max-workers "${MAX_WORKERS}" \
  "${LONGAV_MODEL_ARGS[@]}" \
  "${LONGAV_OPTIONAL_ARGS[@]}"
