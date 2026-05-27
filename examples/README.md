# Examples

This directory contains small templates for prepared samples. They are schema examples only; the placeholder media paths are not meant to run.

## Prepared Sample Template

```text
prepared_sample_template/
  sample_index.json
  canonical_events.json
  event_checklists.json
  ExampleModel/
    events_manifest.json
```

To run a real sample, replace the placeholder paths in `events_manifest.json` with actual `full_video.mp4`, event clips, and boundary clips.

## Minimal Batch Command

After preparing real sample directories under `samples/`:

```bash
bash run_eval_batch.sh samples runs/example '*__*'
```

For a smoke test:

```bash
LONGAV_MAX_EVENTS=1 LONGAV_MAX_BOUNDARIES=1 \
bash run_eval_batch.sh samples runs/smoke '*__*'
```
