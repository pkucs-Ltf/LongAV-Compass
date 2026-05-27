from __future__ import annotations

import json
from pathlib import Path

from .media import (
    ensure_exists,
    extract_frame,
    extract_segment_audio,
    extract_segment_video,
    normalize_audio,
    normalize_video,
    probe_audio,
    probe_video,
)
from .runtime import PreparedSample, SegmentArtifacts
from .schemas import SampleManifest


def prepare_sample(manifest: SampleManifest, run_dir: Path) -> PreparedSample:
    normalized_dir = run_dir / "normalized_media"
    segments_dir = run_dir / "segments"
    frames_dir = run_dir / "frames"
    metadata_dir = run_dir / "metadata"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    normalized_video_path: Path | None = None
    normalized_audio_path: Path | None = None
    video_info = None
    audio_info = None

    if "video" in manifest.media.modalities:
        raw_video_source = manifest.resolve_path(manifest.media.video_path) or manifest.resolve_path(manifest.media.container_path)
        video_source = ensure_exists(raw_video_source, "video source")
        normalized_video_path = normalized_dir / "video.mp4"
        normalize_video(video_source, normalized_video_path)
        video_info = probe_video(normalized_video_path)

    if "audio" in manifest.media.modalities:
        raw_audio_source = manifest.resolve_path(manifest.media.audio_path) or manifest.resolve_path(manifest.media.container_path)
        audio_source = ensure_exists(raw_audio_source, "audio source")
        normalized_audio_path = normalized_dir / "audio.wav"
        normalize_audio(audio_source, normalized_audio_path)
        audio_info = probe_audio(normalized_audio_path)

    reference_video_frames: list[str] = []
    reference_video_path = manifest.resolve_path(manifest.reference.ref_video_path)
    if reference_video_path:
        ensure_exists(reference_video_path, "reference video")
        ref_frames_dir = frames_dir / "reference_video"
        timestamps = _sample_reference_timestamps(reference_video_path)
        for index, timestamp in enumerate(timestamps, start=1):
            frame_path = ref_frames_dir / f"ref_{index:02d}.jpg"
            extract_frame(reference_video_path, frame_path, timestamp)
            reference_video_frames.append(frame_path.as_posix())

    segment_bundles: list[SegmentArtifacts] = []
    for index, event in enumerate(manifest.reference.events, start=1):
        overlap = event.overlap_sec or 0.0
        start_sec = max(0.0, event.start_sec - overlap)
        end_sec = event.end_sec + overlap
        segment_slug = f"{index:02d}_{event.event_id}"

        video_segment_path: Path | None = None
        audio_segment_path: Path | None = None

        if normalized_video_path is not None:
            video_segment_path = segments_dir / f"{segment_slug}.mp4"
            extract_segment_video(normalized_video_path, video_segment_path, start_sec, end_sec)
        if normalized_audio_path is not None:
            audio_segment_path = segments_dir / f"{segment_slug}.wav"
            extract_segment_audio(normalized_audio_path, audio_segment_path, start_sec, end_sec)

        frame_paths: list[str] = []
        if normalized_video_path is not None:
            event_frames_dir = frames_dir / segment_slug
            timestamps = _sample_event_timestamps(start_sec, end_sec)
            for frame_index, timestamp in enumerate(timestamps, start=1):
                frame_path = event_frames_dir / f"frame_{frame_index:02d}.jpg"
                extract_frame(normalized_video_path, frame_path, timestamp)
                frame_paths.append(frame_path.as_posix())

        segment_bundles.append(
            SegmentArtifacts(
                event_id=event.event_id,
                start_sec=start_sec,
                end_sec=end_sec,
                video_path=video_segment_path.as_posix() if video_segment_path else None,
                audio_path=audio_segment_path.as_posix() if audio_segment_path else None,
                frame_paths=tuple(frame_paths),
            )
        )

    prepared = PreparedSample(
        sample_id=manifest.sample_id,
        task=manifest.task,
        run_dir=run_dir.as_posix(),
        normalized_video_path=normalized_video_path.as_posix() if normalized_video_path else None,
        normalized_audio_path=normalized_audio_path.as_posix() if normalized_audio_path else None,
        reference_image_path=str(manifest.resolve_path(manifest.reference.ref_image_path)) if manifest.reference.ref_image_path else None,
        reference_video_path=str(reference_video_path) if reference_video_path else None,
        reference_video_frame_paths=tuple(reference_video_frames),
        video_info=video_info,
        audio_info=audio_info,
        segments=tuple(segment_bundles),
    )

    with (metadata_dir / "prepared_sample.json").open("w", encoding="utf-8") as handle:
        json.dump(prepared.to_dict(), handle, indent=2)
    return prepared


def _sample_event_timestamps(start_sec: float, end_sec: float) -> tuple[float, float, float]:
    midpoint = (start_sec + end_sec) / 2.0
    duration = max(end_sec - start_sec, 0.1)
    margin = min(0.5, duration / 4.0)
    return (start_sec + margin, midpoint, end_sec - margin)


def _sample_reference_timestamps(reference_video_path: Path) -> tuple[float, float, float]:
    info = probe_video(reference_video_path)
    duration = info.duration_sec if info and info.duration_sec else 1.0
    return (0.0, duration / 2.0, max(0.0, duration - 0.1))

