#!/usr/bin/env bash
set -euo pipefail

LONGAV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LONGAV_RUNS_ROOT="${LONGAV_RUNS_ROOT:-${LONGAV_ROOT}/runs}"
LONGAV_PYTHON="${LONGAV_PYTHON:-python}"

echo "[LongAV] repository : ${LONGAV_ROOT}"
echo "[LongAV] python     : ${LONGAV_PYTHON}"

"${LONGAV_PYTHON}" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("LongAV-Compass requires Python >= 3.11")
print(f"[LongAV] python ver : {sys.version.split()[0]}")
PY

echo "[LongAV] installing LongAV-Compass package..."
"${LONGAV_PYTHON}" -m pip install --upgrade pip
(cd "${LONGAV_ROOT}" && "${LONGAV_PYTHON}" -m pip install -e '.[judge,clip]')

if ! "${LONGAV_PYTHON}" -c "import open_clip" >/dev/null 2>&1
then
  echo "[LongAV] installing OpenCLIP..."
  "${LONGAV_PYTHON}" -m pip install "open_clip_torch>=2.24"
fi

if ! "${LONGAV_PYTHON}" -c "import clip" >/dev/null 2>&1
then
  echo "[LongAV] installing OpenAI CLIP..."
  "${LONGAV_PYTHON}" -m pip install git+https://github.com/openai/CLIP.git
fi

echo "[LongAV] checking CLIP backend..."
"${LONGAV_PYTHON}" - <<'PY'
from longav_eval.clip_backend import _load_clip

_load_clip()
print("[LongAV] CLIP backend is ready")
PY

mkdir -p "${LONGAV_RUNS_ROOT}"

if [[ ! -f "${LONGAV_ROOT}/configs/api_keys.yaml" ]]; then
  cp "${LONGAV_ROOT}/configs/api_keys.example.yaml" "${LONGAV_ROOT}/configs/api_keys.yaml"
  echo "[LongAV] created configs/api_keys.yaml from template"
else
  echo "[LongAV] configs/api_keys.yaml already exists"
fi

if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  echo "[LongAV] ffmpeg    : $(command -v ffmpeg)"
  echo "[LongAV] ffprobe   : $(command -v ffprobe)"
else
  echo "[LongAV] warning: ffmpeg/ffprobe not found. Install them before media clipping or video diagnostics." >&2
fi

echo "[LongAV] setup complete"
echo "[LongAV] CLIP backend was initialized by setup_longav.sh."
echo "[LongAV] next: edit configs/api_keys.yaml, then run:"
echo "         bash run_eval_batch.sh /path/to/test_sample runs/example_eval '*__*'"
