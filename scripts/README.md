# LongAV-Compass Scripts

These scripts are thin wrappers around the Python CLI. They are intended for users who want stable copy-paste commands without memorizing every `python -m longav_eval` option.

## Top-Level Entry Points

Run from the repository root:

```bash
conda env create -f environment.yaml
conda activate longav_compass
bash setup_longav.sh
bash run_eval_batch.sh /path/to/test_sample runs/example '*__*'
bash run_eval_single.sh /path/to/sample runs/single
bash run_compute_scores.sh /path/to/best_sample_model_metrics_wide.csv runs/paper
```

## Environment Variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `LONGAV_PYTHON` | Python interpreter | first `python3` on `PATH` |
| `LONGAV_API_KEYS` | API key YAML | `configs/api_keys.yaml` |
| `LONGAV_QA_ROOT` | External fixed-QA dataset root | unset |
| `LONGAV_MODEL_NAMES` | Space-separated model aliases | all models in each sample |
| `LONGAV_MAX_WORKERS` | Batch worker count | `8` |
| `LONGAV_SKIP_AUDIO` | `1` skips audio diagnostics, `0` enables them | `1` |
| `LONGAV_ONLY_MISSING` | compute only missing metrics | `0` |
| `LONGAV_REUSE_ROOT` | previous batch output root | unset |
| `LONGAV_CACHE_ROOT` | local cache root | `.cache/` |
| `LONGAV_RUNS_ROOT` | default run output root | `runs/` |

## Direct CLI Equivalent

The batch wrapper expands to:

```bash
PYTHONPATH=src python -m longav_eval run-event-eval-batch \
  --sample-root /path/to/test_sample \
  --sample-glob '*__*' \
  --output-root runs/example \
  --api-keys configs/api_keys.yaml \
  --max-workers 8 \
  --skip-audio
```
