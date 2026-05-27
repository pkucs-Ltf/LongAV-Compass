# Score Definitions

LongAV-Compass separates metric scores, paper-facing balanced scores, and human-alignment win rates.

## Metric Scores

Metric scores are the direct outputs of individual evaluators.

| Metric | Column | Range | Notes |
| --- | --- | --- | --- |
| Event fulfillment | `event_fulfillment` | 0-1 | Event-level QA satisfaction. |
| Event realization / visual quality | `event_realization` | 1-5 | MOS-style event visual quality. |
| Long-form structure | `long_form_structure` | 1-5 | Event order, coverage, pacing, and continuity. |
| Transition stability | `transition_stability` | 1-5 | Boundary stability between adjacent events. |
| Holistic presentation | `holistic_presentation` | 1-5 | Overall presentation and watchability. |
| Text-video alignment | `text_video_alignment_clip` | 0-1 | CLIP-based event-level text-video alignment. |
| First-frame image anchoring | `iv1_clip` | 0-1 | I2AV reference image to first-frame alignment. |
| Image-video alignment | `imgalign_clip` | 0-1 | I2AV chained image-video event alignment. |

Audio diagnostics are reported separately and are not included in the paper-facing balanced score.

## Balanced Score

`balanced_score` is the paper-facing 0-100 score used for scenario, difficulty, event-count, and family-level analyses. It uses the six shared video metrics only.

Each metric is first normalized to 0-100:

```text
VQA_100     = event_fulfillment * 100
VQ_100      = event_realization / 5 * 100
Cont_100    = long_form_structure / 5 * 100
Trans_100   = transition_stability / 5 * 100
Hol_100     = holistic_presentation / 5 * 100
TVAlign_100 = text_video_alignment_clip * 100
```

The balanced score is the unweighted mean:

```text
balanced_score = mean(
  VQA_100,
  VQ_100,
  Cont_100,
  Trans_100,
  Hol_100,
  TVAlign_100
)
```

It does not include audio, I2AV image-alignment metrics, or any weighted total.

## Human Alignment

Human alignment does not use a total score. Human raters score three dimensions on a 1-5 scale:

```text
content_fidelity
visual_quality
long_video_stability
```

The benchmark side maps automatic metrics to the same three dimensions:

```text
benchmark_content_fidelity =
mean(event_fulfillment, text_video_alignment_clip)

benchmark_visual_quality =
mean(event_realization / 5, holistic_presentation / 5)

benchmark_long_video_stability =
mean(long_form_structure / 5, transition_stability / 5)
```

The comparison is converted into pairwise win rates within the same sample and rater. The scatter plot compares human win rate against LongAV-Compass win rate for each model and dimension.
