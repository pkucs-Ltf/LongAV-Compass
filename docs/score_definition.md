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

## Video MOS Rubrics

The 1-5 video MOS metrics are computed from dimension-level fields:

```text
event_realization =
  0.30 * motion_naturalness +
  0.25 * subject_integrity +
  0.25 * artifact_control +
  0.20 * visual_quality

long_form_structure =
  0.30 * event_order_correctness +
  0.25 * coverage_balance +
  0.25 * pacing_consistency +
  0.20 * cross_event_continuity

transition_stability =
  0.70 * algorithm_transition_stability +
  0.30 * llm_transition_stability

holistic_presentation =
  0.25 * style_consistency +
  0.25 * visual_appeal +
  0.25 * commercial_completeness +
  0.25 * overall_watchability
```

### Event Realization

Event realization evaluates the generation quality of a single attempted event. Semantic checklist fulfillment is scored separately; if the target event is absent or unjudgeable, use 1-2 for the event-realization fields.

`motion_naturalness`

| Score | Definition |
| --- | --- |
| 1 | Motion is failed, physically incoherent, or impossible to judge. |
| 2 | Motion is mostly stiff, incorrect, or visibly broken. |
| 3 | Motion is understandable but has visible unnaturalness, jitter, or timing issues. |
| 4 | Motion is natural overall, with only minor local issues. |
| 5 | Motion is fluent, physically plausible, and well matched to the attempted event. |

`subject_integrity`

| Score | Definition |
| --- | --- |
| 1 | Main subjects are missing, unrecognizable, or severely deformed. |
| 2 | Subjects are recognizable but unstable, distorted, or frequently broken. |
| 3 | Subjects are mostly recognizable, with visible but tolerable identity or geometry issues. |
| 4 | Subjects remain stable and intact, with only minor defects. |
| 5 | Subjects are consistently clear, complete, and visually coherent. |

`artifact_control`

| Score | Definition |
| --- | --- |
| 1 | Severe artifacts dominate the clip and make it hard to judge. |
| 2 | Obvious artifacts repeatedly disrupt the clip. |
| 3 | Artifacts are noticeable but the event remains viewable. |
| 4 | Artifacts are mild or infrequent. |
| 5 | The clip is clean, with no obvious generation artifacts. |

`visual_quality`

| Score | Definition |
| --- | --- |
| 1 | Image quality is unusable because of blur, exposure failure, compression, or corruption. |
| 2 | Image quality is poor and frequently distracts from the event. |
| 3 | Image quality is acceptable, with visible but tolerable issues. |
| 4 | Image quality is good, with only minor defects. |
| 5 | Image quality is sharp, clear, well exposed, and visually polished. |

### Long-Form Structure

Long-form structure evaluates the complete generated video. It focuses on event order, coverage balance, pacing, and cross-event continuity. Low-level frame quality is considered only when it breaks the long-form structure.

`event_order_correctness`

| Score | Definition |
| --- | --- |
| 1 | Event order is mostly wrong, missing, or impossible to follow. |
| 2 | Several events are out of order or incorrectly arranged. |
| 3 | The main event order is mostly correct, with noticeable ordering or transition ambiguity. |
| 4 | Event order is correct, with only minor ambiguity. |
| 5 | Event order fully follows the intended sequence. |

`coverage_balance`

| Score | Definition |
| --- | --- |
| 1 | Most events are absent or severely underrepresented. |
| 2 | Some events are covered, but important events are missing or extremely imbalanced. |
| 3 | Major events are covered, but duration or emphasis is visibly uneven. |
| 4 | Event coverage is balanced overall, with only minor imbalance. |
| 5 | All events are covered with appropriate and well balanced emphasis. |

`pacing_consistency`

| Score | Definition |
| --- | --- |
| 1 | Pacing is broken, with severe stalls, jumps, or rushed segments. |
| 2 | Pacing is often too rushed, too slow, or uneven. |
| 3 | Pacing is acceptable, but several segments feel rushed, stretched, or uneven. |
| 4 | Pacing is smooth overall, with minor rhythm issues. |
| 5 | Pacing is natural and supports the long-video structure throughout. |

`cross_event_continuity`

| Score | Definition |
| --- | --- |
| 1 | Cross-event continuity is absent or incoherent. |
| 2 | Continuity between events is often broken or confusing. |
| 3 | Continuity is understandable but has visible gaps or abrupt changes. |
| 4 | Continuity is coherent overall, with only minor discontinuities. |
| 5 | Events connect into a coherent long-form video with natural continuity. |

### Transition Stability

Transition stability evaluates event-boundary clips. Normal shot changes and event changes are allowed; only boundary defects such as black frames, flashes, freezes, repeated frames, stutter, non-story deformation, broken action, or object disappearance are penalized.

`llm_transition_stability`

| Score | Definition |
| --- | --- |
| 1 | Severe boundary failure such as black frames, freezes, broken action, major deformation, or object disappearance. |
| 2 | Obvious boundary instability that disrupts viewing or event understanding. |
| 3 | Noticeable but tolerable boundary defect, such as a small jump, brief stutter, or mild deformation. |
| 4 | Mostly clean boundary, with only slight visual or motion discontinuity. |
| 5 | Clean and stable boundary with no obvious technical defect. |

### Holistic Presentation

Holistic presentation evaluates the complete video as a finished presentation. It focuses on style consistency, visual appeal, commercial or presentation completeness, and overall watchability, without duplicating event-level checklist scoring.

`style_consistency`

| Score | Definition |
| --- | --- |
| 1 | Visual style is chaotic or inconsistent across the video. |
| 2 | Style changes are frequent and distracting. |
| 3 | Style is mostly consistent, but with noticeable inconsistencies. |
| 4 | Style is consistent overall, with minor deviations. |
| 5 | Style is coherent and stable throughout. |

`visual_appeal`

| Score | Definition |
| --- | --- |
| 1 | The video is visually unpleasant or unusable. |
| 2 | Visual appeal is weak because of repeated quality, composition, or aesthetic problems. |
| 3 | Visual appeal is acceptable but ordinary or uneven. |
| 4 | The video is visually pleasing overall, with minor issues. |
| 5 | The video is highly polished, attractive, and engaging. |

`commercial_completeness`

| Score | Definition |
| --- | --- |
| 1 | The video does not work as a complete advertising or presentation piece. |
| 2 | The video has some relevant material but lacks a clear complete presentation. |
| 3 | The video communicates the main content, but feels incomplete or weakly organized. |
| 4 | The video feels mostly complete as an advertising or presentation piece. |
| 5 | The video feels complete, coherent, and effective as a polished advertising or presentation piece. |

`overall_watchability`

| Score | Definition |
| --- | --- |
| 1 | The video is difficult to watch to completion. |
| 2 | The viewing experience is poor, with repeated disruptions or weak coherence. |
| 3 | The video is watchable, but has noticeable issues in rhythm, clarity, or polish. |
| 4 | The video is easy to watch and mostly polished. |
| 5 | The video is smooth, engaging, coherent, and highly watchable. |

## Audio Diagnostics

Audio diagnostics are reported separately and are not included in the paper-facing balanced score. Event-level audio fields use 1-5 MOS scores:

| Field | Range | Notes |
| --- | --- | --- |
| `av_sync` | 1-5 | Whether speech, sound effects, music changes, and audible accents align with visible actions and cuts. |
| `audio_event_match` | 1-5 | Whether the audio matches the event text and audio expectation. |
| `audio_realism` | 1-5 | Whether the audio is natural, clear, and plausible for the scene. |
| `audio_artifact_control` | 1-5 | Whether the clip avoids clipping, buzzing, abrupt silence, glitches, and repetitive loops. |

Full-video audio fields also use 1-5 MOS scores:

| Field | Range | Notes |
| --- | --- | --- |
| `audio_continuity` | 1-5 | Whether the soundtrack avoids unexplained dropouts, hard cuts, or broken segments. |
| `ambience_stability` | 1-5 | Whether background ambience or music remains coherent across events. |
| `source_consistency` | 1-5 | Whether recurring voices, sound sources, and sound effects stay plausible. |
| `volume_stability` | 1-5 | Whether volume, loudness, and mixing avoid abrupt jumps, clipping, or distortion. |

The reported audio aggregates are:

```text
av_sync_mos  = 0.50 * llm_av_sync + 0.50 * algorithm_av_sync
audq_mos     = 0.75 * llm_audq + 0.25 * algorithm_audq
audlong_mos  = 0.75 * llm_audlong + 0.25 * algorithm_audlong

llm_audq =
  0.45 * audio_event_match +
  0.35 * audio_realism +
  0.20 * audio_artifact_control

llm_audlong =
  0.30 * audio_continuity +
  0.25 * ambience_stability +
  0.25 * source_consistency +
  0.20 * volume_stability
```

The final audio diagnostic score is reported on a 0-100 scale:

```text
audio_score = 100 * (
  0.35 * av_sync_norm +
  0.30 * audq_norm +
  0.35 * audlong_norm
)
```

Here each `*_norm` maps a 1-5 MOS score to 0-1 with `(score - 1) / 4`.

### Event-Level Audio Rubric

`av_sync`

| Score | Definition |
| --- | --- |
| 1 | Audio is unrelated to visible actions or cuts, or speech, sound effects, and music changes are severely misaligned. |
| 2 | Some synchronization exists, but there are multiple obvious delays, early sounds, or mismatched sound cues. |
| 3 | Audio is mostly synchronized, with noticeable but tolerable timing errors or missing sound cues. |
| 4 | Audio is well synchronized, with only minor local timing errors. |
| 5 | Audio tightly matches visible actions, cuts, mouth motion when present, and sound-effect trigger points. |

`audio_event_match`

| Score | Definition |
| --- | --- |
| 1 | Audio does not match the event text or audio expectation, or key expected sounds are absent. |
| 2 | Audio has limited relevance, but the main expected sounds or event-specific audio are mostly wrong or missing. |
| 3 | Audio generally matches the event, but some expected sounds, speech, ambience, or music details are missing or inaccurate. |
| 4 | Audio matches the event and audio expectation well, with only minor missing or imprecise details. |
| 5 | Audio precisely covers the event's expected speech, ambience, sound effects, and music behavior. |

`audio_realism`

| Score | Definition |
| --- | --- |
| 1 | Audio is clearly unnatural, distorted, mechanical, or implausible for the scene. |
| 2 | Audio is understandable but has weak realism, poor spatial fit, or obvious synthetic artifacts. |
| 3 | Audio is basically plausible, but has noticeable synthetic quality or imperfect scene fit. |
| 4 | Audio is natural, clear, and scene-appropriate, with only minor realism issues. |
| 5 | Audio is highly natural, clear, spatially plausible, and convincing for the scene. |

`audio_artifact_control`

| Score | Definition |
| --- | --- |
| 1 | Severe clipping, buzzing, dropouts, abrupt silence, glitches, or repetitive loops interfere with understanding. |
| 2 | Obvious artifacts appear repeatedly and hurt the viewing experience. |
| 3 | Artifacts are noticeable but the audio remains usable. |
| 4 | Artifacts are rare or mild. |
| 5 | Audio is clean and stable, with no obvious technical artifacts. |

### Long-Range Audio Rubric

`audio_continuity`

| Score | Definition |
| --- | --- |
| 1 | The soundtrack is fragmented, with frequent unexplained dropouts, hard cuts, or missing segments. |
| 2 | Multiple continuity problems make transitions between events clearly unnatural. |
| 3 | Audio is generally continuous, but has several audible hard cuts, gaps, or abrupt changes. |
| 4 | Audio continuity is good, with only minor cross-event discontinuities. |
| 5 | Audio remains smooth and continuous across the full video. |

`ambience_stability`

| Score | Definition |
| --- | --- |
| 1 | Background ambience or music is chaotic, inconsistent, or changes without scene logic. |
| 2 | Ambience has obvious instability or abrupt cross-event shifts. |
| 3 | Ambience is mostly stable, but contains several noticeable jumps or mismatched background changes. |
| 4 | Ambience and music are stable overall, with only minor fluctuations. |
| 5 | Ambience, music, and acoustic atmosphere remain coherent and scene-appropriate throughout. |

`source_consistency`

| Score | Definition |
| --- | --- |
| 1 | Sound sources are confusing or implausible; voices, objects, or environmental sounds do not match the video. |
| 2 | Several sound sources change identity, direction, or type in inconsistent ways. |
| 3 | Main sound sources are mostly plausible, but some local inconsistencies remain. |
| 4 | Sound sources are consistent, with only minor detail errors. |
| 5 | Voices, object sounds, and environmental sources remain consistent and believable throughout. |

`volume_stability`

| Score | Definition |
| --- | --- |
| 1 | Volume is uncontrolled, with frequent overly loud, too quiet, clipped, or suddenly silent passages. |
| 2 | Multiple loudness jumps or mixing imbalances clearly hurt the experience. |
| 3 | Volume is acceptable overall, but noticeable fluctuations remain. |
| 4 | Loudness and mixing are mostly stable, with only mild fluctuations. |
| 5 | Loudness, dynamic range, and mixing are stable and natural across the full video. |

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
