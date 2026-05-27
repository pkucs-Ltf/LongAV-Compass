#!/usr/bin/env bash
set -euo pipefail

LONGAV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LONGAV_ROOT="$(cd "${LONGAV_SCRIPT_DIR}/.." && pwd)"

export LONGAV_CACHE_ROOT="${LONGAV_CACHE_ROOT:-${LONGAV_ROOT}/.cache}"
export LONGAV_RUNS_ROOT="${LONGAV_RUNS_ROOT:-${LONGAV_ROOT}/runs}"
export LONGAV_API_KEYS="${LONGAV_API_KEYS:-${LONGAV_ROOT}/configs/api_keys.yaml}"
export PYTHONPATH="${LONGAV_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${LONGAV_PYTHON:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    export LONGAV_PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    export LONGAV_PYTHON="$(command -v python)"
  else
    echo "ERROR: python not found. Set LONGAV_PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

resolve_longav_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  elif [[ -e "${PWD}/${path}" ]]; then
    printf '%s\n' "${PWD}/${path}"
  else
    printf '%s\n' "${LONGAV_ROOT}/${path}"
  fi
}

require_longav_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required file not found: ${path}" >&2
    exit 1
  fi
}

require_longav_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: required directory not found: ${path}" >&2
    exit 1
  fi
}

append_model_args() {
  local model
  LONGAV_MODEL_ARGS=()
  if [[ -n "${LONGAV_MODEL_NAMES:-}" ]]; then
    for model in ${LONGAV_MODEL_NAMES}; do
      LONGAV_MODEL_ARGS+=(--model "${model}")
    done
  fi
}

append_optional_eval_args() {
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
  if [[ -n "${LONGAV_REUSE_ROOT:-}" ]]; then
    LONGAV_OPTIONAL_ARGS+=(--reuse-root "$(resolve_longav_path "${LONGAV_REUSE_ROOT}")")
  fi
  if [[ -n "${LONGAV_MAX_EVENTS:-}" ]]; then
    LONGAV_OPTIONAL_ARGS+=(--max-events "${LONGAV_MAX_EVENTS}")
  fi
  if [[ -n "${LONGAV_MAX_BOUNDARIES:-}" ]]; then
    LONGAV_OPTIONAL_ARGS+=(--max-boundaries "${LONGAV_MAX_BOUNDARIES}")
  fi
}

print_longav_eval_env() {
  echo "[LongAV] root       : ${LONGAV_ROOT}"
  echo "[LongAV] python     : ${LONGAV_PYTHON}"
  echo "[LongAV] api keys   : ${LONGAV_API_KEYS}"
  echo "[LongAV] skip audio : ${LONGAV_SKIP_AUDIO:-1}"
  echo "[LongAV] workers    : ${LONGAV_MAX_WORKERS:-8}"
  if [[ -n "${LONGAV_MODEL_NAMES:-}" ]]; then
    echo "[LongAV] models     : ${LONGAV_MODEL_NAMES}"
  else
    echo "[LongAV] models     : all models in each sample"
  fi
  if [[ -n "${LONGAV_QA_ROOT:-}" ]]; then
    echo "[LongAV] qa root    : ${LONGAV_QA_ROOT}"
  fi
}
