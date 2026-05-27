# Plugin Guide

LongAV-Compass should keep task preparation, media processing, model judging, metric computation, and paper aggregation separate.

## Plugin Types

### Metric Plugins

Metric plugins compute one automatic metric for one model output.

Expected interface:

```python
class Metric:
    id: str
    display_name: str
    tasks: set[str]
    required_inputs: set[str]

    def compute(self, sample, model_output, context):
        ...
```

Examples:

```text
event_fulfillment
event_realization
long_form_structure
transition_stability
holistic_presentation
text_video_alignment
image_alignment
audio_quality
```

Metric plugins should output raw metric scores only. Paper-facing aggregation is handled separately by balanced-score utilities.

### Provider Plugins

Provider plugins wrap external judge or embedding backends.

Expected interface:

```python
class JudgeProvider:
    name: str

    def score_video_json(self, prompt, video_path, schema=None):
        ...

    def score_text_json(self, prompt, schema=None):
        ...

    def score_image_json(self, prompt, image_paths, schema=None):
        ...
```

Examples:

```text
gemini
gemini_vertex
gemini_api_key
openai
local_embedding
```

Metric code should call the provider interface rather than directly depending on a specific API client.

### Importer Plugins

Importer plugins convert raw model outputs into the prepared sample format documented in `dataset_format.md`.

Expected interface:

```python
class Importer:
    task: str

    def import_samples(self, input_root, output_root):
        ...
```

Examples:

```text
t2av_raw_outputs
i2av_raw_outputs
v2av_raw_outputs
```

## Aggregation Boundary

Aggregation is not a metric plugin. The paper-facing `balanced_score` is computed after sample-model metric rows are available.

This prevents hidden weighting from entering metric computation and keeps scenario, event-count, difficulty, and family analyses reproducible from the same CSV inputs.
