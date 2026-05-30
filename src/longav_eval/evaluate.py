from __future__ import annotations

from pathlib import Path
from typing import Any

from .clip_backend import (
    ClipBackendError,
    image_semantic_alignment_score,
    image_video_alignment_score,
    text_video_alignment_score,
)
from .gemini import GeminiClient
from .heuristics import (
    audio_coherence_score,
    audio_quality_score,
    average,
    bounded_score,
    cosine_similarity_score,
    frame_motion_score,
    image_quality_score,
    map_correlation_to_score,
    safe_correlation,
)
from .runtime import MetricResult, PreparedSample
from .schemas import ExecutionPlan, SampleManifest


CLIP_FIRST_METRICS = {"TVAlign", "IV_First", "ImgSem", "SubDrift", "StyleConsist"}
GEMINI_METRICS = {"VQ", "Trans", "Narrative", "AudQ", "AudLong", "ContTrans", "StoryCreat"}
LLM_ONLY_METRICS = {"StoryCreat"}


def evaluate_plan(
    manifest: SampleManifest,
    plan: ExecutionPlan,
    prepared: PreparedSample,
    api_keys: dict[str, Any],
) -> dict[str, MetricResult]:
    results: dict[str, MetricResult] = {}
    gemini_client = _build_gemini_client(api_keys)

    for decision in plan.metrics:
        if decision.status != "ready":
            results[decision.metric_id] = MetricResult(
                metric_id=decision.metric_id,
                status="skipped",
                backend="none",
                score=None,
                reason=decision.reason or f"Metric status is {decision.status}",
            )
            continue

        if decision.metric_id in CLIP_FIRST_METRICS:
            local_result = _evaluate_locally(decision.metric_id, manifest, prepared)
            if local_result is not None:
                results[decision.metric_id] = local_result
                continue

        if gemini_client and decision.metric_id in GEMINI_METRICS:
            try:
                results[decision.metric_id] = _evaluate_with_gemini(decision.metric_id, manifest, prepared, gemini_client)
                continue
            except Exception as exc:  # noqa: BLE001
                if decision.metric_id in LLM_ONLY_METRICS:
                    results[decision.metric_id] = MetricResult(
                        metric_id=decision.metric_id,
                        status="error",
                        backend="gemini",
                        score=None,
                        reason=str(exc),
                    )
                    continue

        local_result = _evaluate_locally(decision.metric_id, manifest, prepared)
        if local_result is not None:
            results[decision.metric_id] = local_result
            continue

        results[decision.metric_id] = MetricResult(
            metric_id=decision.metric_id,
            status="skipped",
            backend="none",
            score=None,
            reason="No local evaluator available and no usable Gemini backend configured.",
        )
    return results


def _build_gemini_client(api_keys: dict[str, Any]) -> GeminiClient | None:
    gemini = api_keys.get("gemini") or {}
    provider = gemini.get("provider") or "vertex"
    model = gemini.get("model") or "gemini-2.5-pro"
    api_key = gemini.get("api_key") or ""
    project = gemini.get("project") or ""
    location = gemini.get("location") or "global"
    credentials_json = gemini.get("credentials_json") or ""

    if provider == "vertex" and project and credentials_json:
        return GeminiClient(
            model=model,
            provider="vertex",
            project=project,
            location=location,
            credentials_json=credentials_json,
        )
    if provider == "api_key" and api_key:
        return GeminiClient(
            model=model,
            provider="api_key",
            api_key=api_key,
        )
    if api_key:
        return GeminiClient(
            model=model,
            provider="api_key",
            api_key=api_key,
        )
    if project and credentials_json:
        return GeminiClient(
            model=model,
            provider="vertex",
            project=project,
            location=location,
            credentials_json=credentials_json,
        )
    if provider not in {"vertex", "api_key"}:
        raise RuntimeError(
            f"Unsupported Gemini provider '{provider}'. Use 'vertex' or 'api_key'."
        )
    return None


def _evaluate_locally(metric_id: str, manifest: SampleManifest, prepared: PreparedSample) -> MetricResult | None:
    segments = list(prepared.segments)
    frame_paths = _all_frame_paths(segments)

    if metric_id == "TVAlign":
        text = _alignment_text(manifest)
        try:
            details = text_video_alignment_score(text, frame_paths)
        except ClipBackendError as exc:
            return MetricResult(metric_id, "failed", "clip", None, reason=str(exc))
        score = 100.0 * float(details["text_video_alignment_clip"])
        return MetricResult(metric_id, "computed", "clip", round(score, 4), details=details)

    if metric_id == "ImgSem":
        if not prepared.reference_image_path:
            return MetricResult(metric_id, "skipped", "clip", None, reason="Missing reference image.")
        try:
            details = image_semantic_alignment_score(prepared.reference_image_path, _alignment_text(manifest), frame_paths)
        except ClipBackendError as exc:
            return MetricResult(metric_id, "failed", "clip", None, reason=str(exc))
        score = 100.0 * float(details["image_semantic_alignment_clip"])
        return MetricResult(metric_id, "computed", "clip", round(score, 4), details=details)

    if metric_id == "VQ":
        frame_scores = []
        details = []
        for segment in segments:
            for frame_path in segment.frame_paths:
                score, parts = image_quality_score(frame_path)
                frame_scores.append(score)
                details.append(parts)
        return MetricResult(metric_id, "computed", "heuristic", round(average(frame_scores), 4), details={"frame_count": len(frame_scores)})

    if metric_id == "ID_seg":
        scores = []
        for segment in segments:
            if len(segment.frame_paths) < 2:
                continue
            segment_scores = []
            for first, second in zip(segment.frame_paths, segment.frame_paths[1:]):
                segment_scores.append(cosine_similarity_score(first, second))
            scores.append(average(segment_scores))
        return MetricResult(metric_id, "computed", "heuristic", round(average(scores), 4), details={"segment_count": len(scores)})

    if metric_id == "ID_cross":
        scores = []
        for first, second in zip(segments, segments[1:]):
            if not first.frame_paths or not second.frame_paths:
                continue
            scores.append(cosine_similarity_score(first.frame_paths[-1], second.frame_paths[0]))
        return MetricResult(metric_id, "computed", "heuristic", round(average(scores), 4), details={"transition_count": len(scores)})

    if metric_id == "Trans":
        boundary_scores = []
        for first, second in zip(segments, segments[1:]):
            if not first.frame_paths or not second.frame_paths:
                continue
            visual = cosine_similarity_score(first.frame_paths[-1], second.frame_paths[0])
            motion_delta = abs(frame_motion_score(first.frame_paths) - frame_motion_score(second.frame_paths))
            boundary_scores.append(bounded_score(0.75 * visual + 0.25 * max(0.0, 100.0 - motion_delta)))
        return MetricResult(metric_id, "computed", "heuristic", round(average(boundary_scores), 4), details={"transition_count": len(boundary_scores)})

    if metric_id == "Narrative":
        cross_scores = []
        motion_scores = []
        for first, second in zip(segments, segments[1:]):
            if first.frame_paths and second.frame_paths:
                cross_scores.append(cosine_similarity_score(first.frame_paths[-1], second.frame_paths[0]))
                motion_scores.append(100.0 - abs(frame_motion_score(first.frame_paths) - frame_motion_score(second.frame_paths)))
        coverage = 100.0 if segments else 0.0
        score = bounded_score(0.5 * average(cross_scores) + 0.35 * average(motion_scores) + 0.15 * coverage)
        return MetricResult(metric_id, "computed", "heuristic", round(score, 4), details={"segment_count": len(segments)})

    if metric_id == "AVSync":
        motion_values = []
        audio_values = []
        for segment in segments:
            if segment.frame_paths:
                motion_values.append(frame_motion_score(segment.frame_paths))
            if segment.audio_path:
                audio_score, audio_details = audio_quality_score(segment.audio_path)
                audio_values.append(audio_details["rms"])
            if len(audio_values) < len(motion_values):
                audio_values.append(0.0)
        score = map_correlation_to_score(safe_correlation(motion_values, audio_values))
        return MetricResult(metric_id, "computed", "heuristic", round(score, 4), details={"segment_count": len(segments)})

    if metric_id == "AudQ":
        scores = []
        for segment in segments:
            if segment.audio_path:
                score, _ = audio_quality_score(segment.audio_path)
                scores.append(score)
        return MetricResult(metric_id, "computed", "heuristic", round(average(scores), 4), details={"segment_count": len(scores)})

    if metric_id == "AudLong":
        if not prepared.normalized_audio_path:
            return MetricResult(metric_id, "skipped", "heuristic", None, reason="No normalized audio available.")
        score, details = audio_coherence_score(prepared.normalized_audio_path)
        return MetricResult(metric_id, "computed", "heuristic", round(score, 4), details=details)

    if metric_id == "IV_First":
        if not prepared.reference_image_path or not segments or not segments[0].frame_paths:
            return MetricResult(metric_id, "skipped", "clip", None, reason="Missing reference image or first segment frame.")
        try:
            details = image_video_alignment_score(prepared.reference_image_path, [segments[0].frame_paths[0]])
        except ClipBackendError as exc:
            return MetricResult(metric_id, "failed", "clip", None, reason=str(exc))
        score = 100.0 * float(details["image_video_alignment_clip"])
        return MetricResult(metric_id, "computed", "clip", round(score, 4), details=details)

    if metric_id == "SubDrift":
        if not prepared.reference_image_path:
            return MetricResult(metric_id, "skipped", "clip", None, reason="Missing reference image.")
        try:
            details = image_video_alignment_score(prepared.reference_image_path, frame_paths)
        except ClipBackendError as exc:
            return MetricResult(metric_id, "failed", "clip", None, reason=str(exc))
        score = 100.0 * float(details["image_video_alignment_clip"])
        return MetricResult(metric_id, "computed", "clip", round(score, 4), details=details)

    if metric_id == "ContTrans":
        if not prepared.reference_video_frame_paths or not segments or not segments[0].frame_paths:
            return MetricResult(metric_id, "skipped", "clip", None, reason="Missing reference video frames or continuation frames.")
        try:
            details = image_video_alignment_score(prepared.reference_video_frame_paths[-1], [segments[0].frame_paths[0]])
        except ClipBackendError as exc:
            return MetricResult(metric_id, "failed", "clip", None, reason=str(exc))
        score = 100.0 * float(details["image_video_alignment_clip"])
        return MetricResult(metric_id, "computed", "clip", round(score, 4), details=details)

    if metric_id == "StyleConsist":
        if not prepared.reference_video_frame_paths:
            return MetricResult(metric_id, "skipped", "clip", None, reason="Missing reference video frames.")
        continuation_frames = [frame for segment in segments for frame in segment.frame_paths[:1]]
        if not continuation_frames:
            return MetricResult(metric_id, "skipped", "clip", None, reason="Missing continuation frames.")
        try:
            payloads = [
                image_video_alignment_score(reference_frame, continuation_frames)
                for reference_frame in prepared.reference_video_frame_paths
            ]
        except ClipBackendError as exc:
            return MetricResult(metric_id, "failed", "clip", None, reason=str(exc))
        scores = [float(payload["image_video_alignment_clip"]) for payload in payloads]
        score = 100.0 * average(scores)
        return MetricResult(
            metric_id,
            "computed",
            "clip",
            round(score, 4),
            details={
                "backend": "clip",
                "reference_frame_count": len(payloads),
                "continuation_frame_count": len(continuation_frames),
                "style_consistency_clip": round(average(scores), 6),
                "reference_scores": scores,
            },
        )

    return None


def _all_frame_paths(segments: list[Any]) -> list[str]:
    return [frame_path for segment in segments for frame_path in segment.frame_paths]


def _alignment_text(manifest: SampleManifest) -> str:
    event_lines = [
        f"{event.event_id}: {event.text}"
        for event in manifest.reference.events
        if event.text
    ]
    parts = [manifest.reference.global_description.strip()]
    if event_lines:
        parts.append("\n".join(event_lines))
    return "\n".join(part for part in parts if part)


def _evaluate_with_gemini(
    metric_id: str,
    manifest: SampleManifest,
    prepared: PreparedSample,
    client: GeminiClient,
) -> MetricResult:
    prompt, media_paths, mime_types = _build_gemini_request(metric_id, manifest, prepared)
    payload = client.score_media(prompt, media_paths, mime_types)
    score = float(payload["score"])
    return MetricResult(
        metric_id=metric_id,
        status="computed",
        backend="gemini",
        score=round(score, 4),
        details={"reason": payload.get("reason", "")},
    )


def _build_gemini_request(metric_id: str, manifest: SampleManifest, prepared: PreparedSample) -> tuple[str, list[Path], list[str]]:
    event_lines = []
    for event in manifest.reference.events:
        event_lines.append(f"- {event.event_id}: {event.start_sec:.1f}-{event.end_sec:.1f}s | {event.text}")
    event_block = "\n".join(event_lines)
    media_paths: list[Path] = []
    mime_types: list[str] = []

    if metric_id in {"VQ", "Trans", "Narrative", "TVAlign", "ImgSem", "ContTrans", "StoryCreat"}:
        for segment in prepared.segments[:6]:
            if segment.frame_paths:
                media_paths.append(Path(segment.frame_paths[1 if len(segment.frame_paths) > 1 else 0]))
                mime_types.append("image/jpeg")
    if metric_id == "ImgSem" and prepared.reference_image_path:
        media_paths.insert(0, Path(prepared.reference_image_path))
        mime_types.insert(0, "image/jpeg")
    if metric_id == "ContTrans" and prepared.reference_video_frame_paths:
        media_paths = [Path(prepared.reference_video_frame_paths[-1])] + media_paths[:2]
        mime_types = ["image/jpeg"] + mime_types[:2]
    if metric_id in {"AudQ", "AudLong"} and prepared.normalized_audio_path:
        media_paths = [Path(prepared.normalized_audio_path)]
        mime_types = ["audio/wav"]

    prompts = {
        "VQ": "Score the visual quality of the provided video frames from 0 to 100. Focus on sharpness, exposure, artifacts, and overall fidelity.",
        "Trans": "Score transition naturalness from 0 to 100. Judge whether adjacent events connect smoothly without jarring visual discontinuities.",
        "Narrative": "Score narrative consistency from 0 to 100. Judge whether the sampled frames follow the global description and remain coherent over time.",
        "TVAlign": "Score text-video alignment from 0 to 100. Compare the sampled frames against the global description and event list.",
        "AudQ": "Score audio realism and event-level correspondence from 0 to 100. Use the event list and overall description as reference.",
        "AudLong": "Score long-range audio coherence from 0 to 100. Focus on whether the audio remains stable and plausible across the full clip.",
        "ImgSem": "Score image semantic faithfulness from 0 to 100. The first image is the reference image and the rest are generated video frames.",
        "ContTrans": "Score continuation naturalness from 0 to 100. The first frame comes from the reference video and the remaining frames come from the continuation.",
        "StoryCreat": "Score story creativity from 0 to 100. Consider whether the sampled frames expand the low-spec prompt in a coherent but inventive way.",
    }
    prompt = (
        f"{prompts[metric_id]}\n"
        f"Global description:\n{manifest.reference.global_description}\n\n"
        f"Events:\n{event_block}\n\n"
        'Return strict JSON: {"score": <0-100 number>, "reason": "<short explanation>"}'
    )
    return prompt, media_paths, mime_types
