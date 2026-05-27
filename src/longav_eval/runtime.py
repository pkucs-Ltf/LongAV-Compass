from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class StreamInfo:
    codec: str | None
    duration_sec: float | None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None
    channels: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SegmentArtifacts:
    event_id: str
    start_sec: float
    end_sec: float
    video_path: str | None
    audio_path: str | None
    frame_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreparedSample:
    sample_id: str
    task: str
    run_dir: str
    normalized_video_path: str | None
    normalized_audio_path: str | None
    reference_image_path: str | None
    reference_video_path: str | None
    reference_video_frame_paths: tuple[str, ...]
    video_info: StreamInfo | None
    audio_info: StreamInfo | None
    segments: tuple[SegmentArtifacts, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(slots=True)
class MetricResult:
    metric_id: str
    status: str
    backend: str
    score: float | None
    reason: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunSummary:
    sample_id: str
    profile_name: str
    run_dir: str
    metrics: dict[str, dict[str, Any]]
    scores: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
