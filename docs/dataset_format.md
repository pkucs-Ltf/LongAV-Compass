# Prepared Sample Format

LongAV-Compass evaluates prepared samples. Importers should convert raw model outputs into this format before evaluation.

```text
sample_dir/
  sample_index.json
  canonical_events.json
  event_checklists.json          optional
  <model_name>/
    events_manifest.json
    full_video.mp4
    events/
      event_001.mp4
      event_002.mp4
    boundaries/
      boundary_001.mp4
```

## Building Samples from LongAVBench

The released Hugging Face dataset provides prompts, fixed QA, and I2AV/V2AV references:

Dataset repository: https://huggingface.co/datasets/TengfeiLiuCoder/LongAV-Compass

```text
LongAVBench/
  T2AV/final_json/T2AV_001.json
  I2AV/final_json/I2AV_001.json
  I2AV/images/I2AV_001.jpg
  V2AV/final_json/V2AV_001.json
  V2AV/videos/V2AV_001.mp4
```

Put generated model videos separately. The model directory name must contain the task name, and each sample needs `full_video.mp4`:

```text
outputs/
  MyModel_v2av_run/
    advertising/
      V2AV_001/
        full_video.mp4
```

Convert the dataset JSON plus model outputs into prepared samples:

```bash
PYTHONPATH=src python -m longav_eval prepare-event-testset \
  --source-root data/LongAV-Compass \
  --output-root outputs \
  --test-root samples/prepared \
  --sample V2AV_001
```

The `--sample` argument accepts either `V2AV_001` or `v2av:advertising:V2AV_001`.

## `sample_index.json`

This file identifies the task, category, source sample, and available model outputs.

Required fields:

```json
{
  "sample_id": "t2av__advertising__2026-05-06%20214648",
  "task": "t2av",
  "category": "advertising",
  "source_sample_id": "2026-05-06%20214648",
  "models": [
    {"model": "LongCat-Video"}
  ]
}
```

When built from LongAVBench, I2AV and V2AV samples also include a `reference` object in `sample_index.json`, for example:

```json
{
  "reference": {
    "image_path": "data/LongAV-Compass/I2AV/images/I2AV_001.jpg"
  }
}
```

## `canonical_events.json`

This file defines the global prompt and canonical event sequence used by evaluators.

Required fields:

```json
{
  "global_description": "Create a long-form video ...",
  "events": [
    {
      "event_id": "e1",
      "start_sec": 0.0,
      "end_sec": 8.0,
      "text": "..."
    }
  ]
}
```

## `<model_name>/events_manifest.json`

This file maps the model's generated media to event and boundary clips.

Required fields:

```json
{
  "video_path": "/absolute/or/relative/path/to/full_video.mp4",
  "events": [
    {
      "event_id": "e1",
      "video_path": "/path/to/events/event_001.mp4",
      "adjusted_start_sec": 0.0,
      "adjusted_end_sec": 8.0,
      "text": "..."
    }
  ],
  "boundaries": [
    {
      "boundary_id": "b1",
      "video_path": "/path/to/boundaries/boundary_001.mp4",
      "left_text": "...",
      "right_text": "..."
    }
  ]
}
```

## QA Files

`event_checklists.json` can be stored inside the sample directory. If it is absent, the evaluator can load frozen QA from the configured dataset root or create a fallback checklist.

The full benchmark QA dataset is not stored in this code repository. Download LongAVBench and pass its root with `--qa-root` when you want to use fixed QA files:

```bash
PYTHONPATH=src python -m longav_eval run-event-eval \
  --sample-dir path/to/sample \
  --output-dir runs/example \
  --api-keys configs/api_keys.yaml \
  --qa-root data/LongAV-Compass
```

If `--qa-root` is omitted, the evaluator uses local `event_checklists.json` when present; otherwise it generates fallback checklists with the configured judge provider.
