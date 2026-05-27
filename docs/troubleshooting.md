# Troubleshooting

This page lists the failure modes that usually prevent LongAV-Compass from running.

## `configs/api_keys.yaml` is missing

Create it from the template:

```bash
cp configs/api_keys.example.yaml configs/api_keys.yaml
```

Then fill either Gemini API-key fields or Vertex fields. Do not commit real credentials.

## `google-genai` import errors

Install judge dependencies:

```bash
python -m pip install -e '.[judge]'
```

or rerun:

```bash
bash setup_longav.sh
```

## `ffmpeg` or `ffprobe` is missing

The evaluator expects these tools for media probing and clip handling. Install them with your system package manager, then verify:

```bash
ffmpeg -version
ffprobe -version
```

`setup_longav.sh` reports whether they are visible, but it does not install system packages automatically.

## No samples are discovered

Check the sample glob. The default wrapper glob is `*__*`, which matches directories such as:

```text
t2av__advertising__2026-05-06-150459
i2av__ecommerce__2026-05-06-152204
v2av__personal_vlog__2026-05-06-171717
```

Use a task-specific glob when needed:

```bash
bash run_eval_batch.sh /path/to/test_sample runs/v2av 'v2av__*'
```

## A model is skipped inside a sample

The evaluator only runs models that have a valid model subdirectory with:

```text
<model_name>/
  events_manifest.json
  full_video.mp4
```

If `LONGAV_MODEL_NAMES` is set, the aliases must match the model directory names exactly.

## Fixed QA files are not used

Pass the external dataset root:

```bash
LONGAV_QA_ROOT=data/LongAV-Compass \
bash run_eval_batch.sh samples/prepared runs/with_fixed_qa '*__*'
```

If `LONGAV_QA_ROOT` is omitted, the evaluator uses sample-local `event_checklists.json` when available, otherwise it generates fallback checklists through the configured judge provider.

## The run is too slow

Use video-only mode and adjust sample-level concurrency:

```bash
LONGAV_SKIP_AUDIO=1 LONGAV_MAX_WORKERS=8 \
bash run_eval_batch.sh /path/to/test_sample runs/video_only '*__*'
```

Increase `LONGAV_MAX_WORKERS` only if your API quota and machine resources can handle the extra requests.

## Resume a failed run

Use a new output directory and point `LONGAV_REUSE_ROOT` to the previous one:

```bash
LONGAV_REUSE_ROOT=runs/previous_eval \
LONGAV_ONLY_MISSING=1 \
bash run_eval_batch.sh /path/to/test_sample runs/resumed_eval '*__*'
```

## Paper scores do not match a table

Confirm that the table was produced from the same wide CSV and that partial rows were not included accidentally. The paper-facing aggregate uses complete six-metric `balanced_score` rows by default:

```bash
bash run_compute_scores.sh /path/to/best_sample_model_metrics_wide.csv runs/paper
```

Set `LONGAV_ALLOW_PARTIAL=1` only for diagnostics, not for paper tables unless the paper explicitly says so.
