from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import EVALUATOR_TYPES, MEDIA_MODALITIES, PROFILE_MODES, TASKS


class ConfigError(ValueError):
    """Raised when a manifest or profile is invalid."""


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Expected non-empty string for '{key}'")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Expected string or null for '{key}'")
    value = value.strip()
    return value or None


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ConfigError(f"Expected number or null for '{key}'")


@dataclass(slots=True)
class EventSpec:
    event_id: str
    start_sec: float
    end_sec: float
    text: str
    audio_expectation: str | None = None
    overlap_sec: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventSpec":
        event_id = _require_str(data, "event_id")
        text = _require_str(data, "text")
        start_sec = _optional_float(data, "start_sec")
        end_sec = _optional_float(data, "end_sec")
        if start_sec is None or end_sec is None:
            raise ConfigError("Each event must include numeric start_sec and end_sec")
        if end_sec <= start_sec:
            raise ConfigError(f"Event '{event_id}' must satisfy end_sec > start_sec")
        return cls(
            event_id=event_id,
            start_sec=start_sec,
            end_sec=end_sec,
            text=text,
            audio_expectation=_optional_str(data, "audio_expectation"),
            overlap_sec=_optional_float(data, "overlap_sec"),
        )


@dataclass(slots=True)
class MediaSpec:
    container_path: str | None
    video_path: str | None
    audio_path: str | None
    modalities: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaSpec":
        raw_modalities = data.get("modalities")
        if not isinstance(raw_modalities, list) or not raw_modalities:
            raise ConfigError("media.modalities must be a non-empty list")
        modalities: list[str] = []
        for modality in raw_modalities:
            if modality not in MEDIA_MODALITIES:
                raise ConfigError(f"Unsupported media modality '{modality}'")
            modalities.append(modality)
        return cls(
            container_path=_optional_str(data, "container_path"),
            video_path=_optional_str(data, "video_path"),
            audio_path=_optional_str(data, "audio_path"),
            modalities=tuple(dict.fromkeys(modalities)),
        )


@dataclass(slots=True)
class ReferenceSpec:
    global_description: str
    events: tuple[EventSpec, ...]
    ref_image_path: str | None = None
    ref_video_path: str | None = None
    continuation_boundary_sec: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceSpec":
        global_description = _require_str(data, "global_description")
        raw_events = data.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ConfigError("reference.events must be a non-empty list")
        events = tuple(EventSpec.from_dict(item) for item in raw_events)
        return cls(
            global_description=global_description,
            events=events,
            ref_image_path=_optional_str(data, "ref_image_path"),
            ref_video_path=_optional_str(data, "ref_video_path"),
            continuation_boundary_sec=_optional_float(data, "continuation_boundary_sec"),
        )


@dataclass(slots=True)
class MetaSpec:
    scenario: str | None = None
    complexity: str | None = None
    prompt_detail: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetaSpec":
        return cls(
            scenario=_optional_str(data, "scenario"),
            complexity=_optional_str(data, "complexity"),
            prompt_detail=_optional_str(data, "prompt_detail"),
        )


@dataclass(slots=True)
class SampleManifest:
    sample_id: str
    task: str
    media: MediaSpec
    reference: ReferenceSpec
    meta: MetaSpec = field(default_factory=MetaSpec)
    manifest_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleManifest":
        sample_id = _require_str(data, "sample_id")
        task = _require_str(data, "task")
        if task not in TASKS:
            raise ConfigError(f"Unsupported task '{task}'")
        media = MediaSpec.from_dict(data.get("media") or {})
        reference = ReferenceSpec.from_dict(data.get("reference") or {})
        meta = MetaSpec.from_dict(data.get("meta") or {})
        manifest = cls(sample_id=sample_id, task=task, media=media, reference=reference, meta=meta)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.task == "i2av" and not self.reference.ref_image_path:
            raise ConfigError("i2av samples must include reference.ref_image_path")
        if self.task == "v2av":
            if not self.reference.ref_video_path:
                raise ConfigError("v2av samples must include reference.ref_video_path")
            if self.reference.continuation_boundary_sec is None:
                raise ConfigError("v2av samples must include continuation_boundary_sec")

    def available_inputs(self) -> set[str]:
        available = set(self.media.modalities)
        available.add("global_text")
        available.add("events")
        if self.reference.ref_image_path:
            available.add("reference_image")
        if self.reference.ref_video_path:
            available.add("reference_video")
        if self.reference.continuation_boundary_sec is not None:
            available.add("continuation_boundary")
        return available

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def base_dir(self) -> Path:
        if self.manifest_path:
            return Path(self.manifest_path).resolve().parent
        return Path.cwd()

    def resolve_path(self, raw_path: str | None) -> Path | None:
        if not raw_path:
            return None
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        manifest_relative = (self.base_dir() / candidate).resolve()
        if manifest_relative.exists():
            return manifest_relative
        cwd_relative = (Path.cwd() / candidate).resolve()
        if cwd_relative.exists():
            return cwd_relative
        return manifest_relative


@dataclass(slots=True)
class AggregationSpec:
    enable: bool
    scheme: str | None
    require_complete_metrics: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AggregationSpec":
        enable = bool(data.get("enable", False))
        scheme = data.get("scheme")
        if scheme is not None and not isinstance(scheme, str):
            raise ConfigError("aggregation.scheme must be a string or null")
        require_complete = bool(data.get("require_complete_metrics", True))
        return cls(enable=enable, scheme=scheme, require_complete_metrics=require_complete)


@dataclass(slots=True)
class Profile:
    profile_name: str
    mode: str
    description: str | None
    allow_auto_skip: bool
    metrics: dict[str, bool]
    aggregation: AggregationSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Profile":
        profile_name = _require_str(data, "profile_name")
        mode = _require_str(data, "mode")
        if mode not in PROFILE_MODES:
            raise ConfigError(f"Unsupported profile mode '{mode}'")
        raw_metrics = data.get("metrics")
        if not isinstance(raw_metrics, dict):
            raise ConfigError("profile.metrics must be a mapping of metric_id -> bool")
        metrics: dict[str, bool] = {}
        for metric_id, enabled in raw_metrics.items():
            if not isinstance(enabled, bool):
                raise ConfigError(f"Metric toggle for '{metric_id}' must be true/false")
            metrics[metric_id] = enabled
        return cls(
            profile_name=profile_name,
            mode=mode,
            description=_optional_str(data, "description"),
            allow_auto_skip=bool(data.get("allow_auto_skip", True)),
            metrics=metrics,
            aggregation=AggregationSpec.from_dict(data.get("aggregation") or {}),
        )

    def enabled_metric_ids(self) -> set[str]:
        return {metric_id for metric_id, enabled in self.metrics.items() if enabled}

    def metric_enabled(self, metric_id: str) -> bool:
        return self.metrics.get(metric_id, False)


@dataclass(slots=True)
class MetricDefinition:
    metric_id: str
    display_name: str
    evaluator_type: str
    required_inputs: frozenset[str]
    tasks: frozenset[str]
    providers: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if self.evaluator_type not in EVALUATOR_TYPES:
            raise ConfigError(f"Unsupported evaluator type '{self.evaluator_type}'")


@dataclass(slots=True)
class MetricDecision:
    metric_id: str
    display_name: str
    enabled_in_profile: bool
    status: str
    evaluator_type: str
    providers: tuple[str, ...]
    missing_inputs: tuple[str, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScoreAvailability:
    score_id: str
    status: str
    required_metrics: tuple[str, ...]
    missing_metrics: tuple[str, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionPlan:
    sample_id: str
    task: str
    profile_name: str
    mode: str
    available_inputs: tuple[str, ...]
    metrics: tuple[MetricDecision, ...]
    scores: tuple[ScoreAvailability, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task": self.task,
            "profile_name": self.profile_name,
            "mode": self.mode,
            "available_inputs": list(self.available_inputs),
            "metrics": [item.to_dict() for item in self.metrics],
            "scores": [item.to_dict() for item in self.scores],
        }
