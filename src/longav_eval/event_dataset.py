from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .media import extract_segment_video, probe_video

HF_TASK_DIRS = {
    "t2av": "T2AV",
    "i2av": "I2AV",
    "v2av": "V2AV",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")
FULL_VIDEO_FILENAMES = ("full_video.mp4", "full.mp4")


@dataclass(slots=True)
class CanonicalEvent:
    event_id: str
    start_sec: float
    end_sec: float
    text: str
    audio_expectation: str | None = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_event_testset(
    output_root: str | Path,
    source_root: str | Path,
    test_root: str | Path,
    samples: list[str],
    boundary_margin_sec: float = 2.0,
    overwrite: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_root)
    source_root = Path(source_root)
    test_root = Path(test_root)
    prepared: list[dict[str, Any]] = []
    expected_models_by_task = _expected_models_by_task(output_root)

    for raw_sample in samples:
        task, category, sample_id = _parse_sample_arg(raw_sample)
        source_prompt_path = _source_prompt_path(source_root, task, category, sample_id)
        source_prompt = _load_json(source_prompt_path)
        task = str(source_prompt.get("task") or task).lower()
        category = category or str(source_prompt.get("category") or "")
        sample_id = str(source_prompt.get("sample_id") or sample_id)
        if not category:
            raise ValueError(f"Missing category for sample '{raw_sample}'. Use task:category:sample_id or include category in JSON.")
        canonical = _canonical_from_source(task, category, sample_id, source_prompt)
        reference = _reference_paths(source_root, task, sample_id, canonical)
        if reference:
            canonical["reference"] = reference
        sample_key = f"{task}__{category}__{sample_id}"
        sample_dir = test_root / sample_key
        sample_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_prompt_path, sample_dir / "source_prompt.json")
        _write_json(sample_dir / "canonical_events.json", canonical)

        model_entries = []
        model_errors = []
        for alias, model_dir_name, model_video in _latest_model_videos(output_root, task, category, sample_id):
            model_dir = sample_dir / alias
            model_dir.mkdir(parents=True, exist_ok=True)
            dest_video = model_dir / "full_video.mp4"
            if overwrite or not dest_video.exists():
                shutil.copy2(model_video, dest_video)

            metadata_path = model_video.parent / "metadata.json"
            if metadata_path.exists() and (overwrite or not (model_dir / "metadata.json").exists()):
                shutil.copy2(metadata_path, model_dir / "metadata.json")

            try:
                event_entries = split_model_video(
                    video_path=dest_video,
                    canonical_events=[CanonicalEvent(**item) for item in canonical["events"]],
                    output_dir=model_dir,
                    boundary_margin_sec=boundary_margin_sec,
                    overwrite=overwrite,
                )
            except Exception as exc:
                model_errors.append(
                    {
                        "model": alias,
                        "source_model_dir": model_dir_name,
                        "video_path": dest_video.as_posix(),
                        "error": repr(exc),
                    }
                )
                continue
            model_entries.append(
                {
                    "model": alias,
                    "source_model_dir": model_dir_name,
                    "video_path": dest_video.as_posix(),
                    "metadata_path": (model_dir / "metadata.json").as_posix(),
                    "events": event_entries["events"],
                    "boundaries": event_entries["boundaries"],
                }
            )

        available_models = [entry["model"] for entry in model_entries]
        expected_models = expected_models_by_task.get(task, available_models)
        index = {
            "sample_id": sample_key,
            "task": task,
            "category": category,
            "source_sample_id": sample_id,
            "canonical_events_path": (sample_dir / "canonical_events.json").as_posix(),
            "source_prompt_path": (sample_dir / "source_prompt.json").as_posix(),
            "reference": reference,
            "expected_models": expected_models,
            "available_models": available_models,
            "missing_models": [model for model in expected_models if model not in set(available_models)],
            "model_errors": model_errors,
            "models": model_entries,
        }
        _write_json(sample_dir / "sample_index.json", index)
        prepared.append(index)

    summary = {"test_root": Path(test_root).as_posix(), "sample_count": len(prepared), "samples": prepared}
    _write_json(Path(test_root) / "index.json", summary)
    return summary


def split_model_video(
    video_path: Path,
    canonical_events: list[CanonicalEvent],
    output_dir: Path,
    boundary_margin_sec: float = 2.0,
    overwrite: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    video_info = probe_video(video_path)
    video_duration = video_info.duration_sec if video_info and video_info.duration_sec else canonical_events[-1].end_sec
    canonical_duration = canonical_events[-1].end_sec
    scale = 1.0
    if canonical_duration > 0:
        delta = abs(video_duration - canonical_duration)
        if delta > max(2.0, canonical_duration * 0.03):
            scale = video_duration / canonical_duration

    events_dir = output_dir / "events"
    boundaries_dir = output_dir / "boundaries"
    events_dir.mkdir(parents=True, exist_ok=True)
    boundaries_dir.mkdir(parents=True, exist_ok=True)

    event_entries: list[dict[str, Any]] = []
    adjusted: list[tuple[CanonicalEvent, float, float]] = []
    for index, event in enumerate(canonical_events, start=1):
        start = _clamp(event.start_sec * scale, 0.0, video_duration)
        end = _clamp(event.end_sec * scale, start + 0.05, video_duration)
        output = events_dir / f"event_{index:03d}.mp4"
        if overwrite or not output.exists():
            extract_segment_video(video_path, output, start, end)
        adjusted.append((event, start, end))
        event_entries.append(
            {
                "event_id": event.event_id,
                "index": index,
                "canonical_start_sec": event.start_sec,
                "canonical_end_sec": event.end_sec,
                "adjusted_start_sec": round(start, 4),
                "adjusted_end_sec": round(end, 4),
                "video_path": output.as_posix(),
                "text": event.text,
                "audio_expectation": event.audio_expectation,
            }
        )

    boundary_entries: list[dict[str, Any]] = []
    for index in range(len(adjusted) - 1):
        left, _, left_end = adjusted[index]
        right, right_start, _ = adjusted[index + 1]
        boundary_center = (left_end + right_start) / 2.0
        start = _clamp(boundary_center - boundary_margin_sec, 0.0, video_duration)
        end = _clamp(boundary_center + boundary_margin_sec, start + 0.05, video_duration)
        output = boundaries_dir / f"boundary_{index + 1:03d}_{left.event_id}_to_{right.event_id}.mp4"
        if overwrite or not output.exists():
            extract_segment_video(video_path, output, start, end)
        boundary_entries.append(
            {
                "boundary_id": f"{left.event_id}_to_{right.event_id}",
                "left_event_id": left.event_id,
                "right_event_id": right.event_id,
                "adjusted_start_sec": round(start, 4),
                "adjusted_end_sec": round(end, 4),
                "video_path": output.as_posix(),
                "left_text": left.text,
                "right_text": right.text,
            }
        )

    manifest = {
        "video_path": video_path.as_posix(),
        "video_duration_sec": video_duration,
        "canonical_duration_sec": canonical_duration,
        "time_scale": scale,
        "events": event_entries,
        "boundaries": boundary_entries,
    }
    _write_json(output_dir / "events_manifest.json", manifest)
    return {"events": event_entries, "boundaries": boundary_entries}


def model_alias(model_dir_name: str) -> str:
    markers = [
        "_cfg_distilled_no_sr_shot_t2av_",
        "_cfg_distilled_no_sr_shot_i2av_",
        "_event_t2av_",
        "_event_i2av_",
        "_event_v2av_",
        "_shot_t2av_",
        "_shot_i2av_",
        "_shot_v2av_",
        "_unify_t2av_",
        "_unify_i2av_",
        "_unify_v2av_",
    ]
    for marker in markers:
        if marker in model_dir_name:
            return model_dir_name.split(marker, 1)[0]
    return re.sub(r"_(?:t2av|i2av|v2av).*", "", model_dir_name)


def _latest_model_videos(output_root: Path, task: str, category: str, sample_id: str) -> list[tuple[str, str, Path]]:
    latest_by_alias: dict[str, tuple[str, str, Path]] = {}
    model_videos: list[Path] = []
    for filename in FULL_VIDEO_FILENAMES:
        model_videos.extend(output_root.glob(f"*/{category}/{sample_id}/{filename}"))
    for model_video in sorted(model_videos):
        model_dir_name = model_video.relative_to(output_root).parts[0]
        if f"_{task}_" not in model_dir_name and f"_{task}" not in model_dir_name:
            continue
        alias = model_alias(model_dir_name)
        candidate = (_run_timestamp(model_dir_name), model_dir_name, model_video)
        current = latest_by_alias.get(alias)
        if current is None or candidate[0] >= current[0]:
            latest_by_alias[alias] = candidate
    return [(alias, model_dir_name, model_video) for alias, (_, model_dir_name, model_video) in sorted(latest_by_alias.items())]


def _expected_models_by_task(output_root: Path) -> dict[str, list[str]]:
    models: dict[str, set[str]] = {}
    if not output_root.exists():
        return {}
    for model_dir in output_root.iterdir():
        if not model_dir.is_dir():
            continue
        name = model_dir.name
        task = None
        if "_t2av_" in name or "_t2av" in name:
            task = "t2av"
        elif "_i2av_" in name or "_i2av" in name:
            task = "i2av"
        elif "_v2av_" in name or "_v2av" in name:
            task = "v2av"
        if not task:
            continue
        if any(path for filename in FULL_VIDEO_FILENAMES for path in model_dir.glob(f"*/*/{filename}")):
            models.setdefault(task, set()).add(model_alias(name))
    return {task: sorted(aliases) for task, aliases in models.items()}


def _run_timestamp(model_dir_name: str) -> str:
    match = re.search(r"_(20\d{6}_\d{6})$", model_dir_name)
    return match.group(1) if match else ""


def _canonical_from_source(task: str, category: str, sample_id: str, source_prompt: dict[str, Any]) -> dict[str, Any]:
    stage = source_prompt.get("stage1") or source_prompt
    raw_events = stage.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"Missing stage1.events for {category}/{sample_id}")

    events: list[CanonicalEvent] = []
    for index, item in enumerate(raw_events, start=1):
        start_sec, end_sec = _parse_time_range(item.get("time_range") or item.get("timeRange") or "")
        if end_sec <= start_sec:
            raise ValueError(f"Invalid event time_range for {category}/{sample_id}: {item.get('time_range')}")
        event_id = str(item.get("event_id") or f"e{index}").strip() or f"e{index}"
        events.append(
            CanonicalEvent(
                event_id=event_id,
                start_sec=start_sec,
                end_sec=end_sec,
                text=item.get("visual_description") or item.get("action") or item.get("text") or "",
                audio_expectation=item.get("audio_expectation"),
            )
        )

    return {
        "task": task,
        "category": category,
        "source_sample_id": sample_id,
        "global_description": stage.get("global_description") or stage.get("video_prompt") or "",
        "video_prompt": stage.get("video_prompt"),
        "level": stage.get("Level") or stage.get("level"),
        "events": [event.to_dict() for event in events],
    }


def _parse_sample_arg(raw_sample: str) -> tuple[str, str, str]:
    raw_sample = raw_sample.strip()
    parts = raw_sample.split(":", 2)
    if len(parts) == 3:
        return parts[0].lower(), parts[1], parts[2]
    parts = raw_sample.split("/", 2)
    if len(parts) == 3:
        return parts[0].lower(), parts[1], parts[2]
    match = re.match(r"^(T2AV|I2AV|V2AV)_", raw_sample, flags=re.IGNORECASE)
    if match:
        task = match.group(1).lower()
        return task, "", _hf_sample_id(task, raw_sample)
    raise ValueError("Sample must be formatted as task:category:sample_id or as a Hugging Face sample id like V2AV_001")


def _source_prompt_path(source_root: Path, task: str, category: str, sample_id: str) -> Path:
    candidates: list[Path] = []
    for root in _source_roots(source_root):
        task_dir = HF_TASK_DIRS.get(task)
        if task_dir:
            hf_sample_id = _hf_sample_id(task, sample_id)
            for task_root in _hf_task_roots(root, task):
                candidates.append(task_root / "final_json" / f"{hf_sample_id}.json")
        if task == "t2av":
            candidates.extend(
                [
                    root / "T2AVdataset" / "Real_video" / "final" / category / f"{sample_id}.json",
                    root / "T2AVdataset" / "LLM_template" / "risk_final" / category / f"{sample_id}.json",
                ]
            )
        if task == "i2av":
            candidates.extend(
                [
                    root / "I2AV_dataset" / "Real_video" / "I2AV_Real" / "final" / category / f"{sample_id}.json",
                    root / "I2AV_dataset" / "LLM_template" / "final" / category / f"{sample_id}.json",
                ]
            )
        candidates.append(root / category / f"{sample_id}.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Source prompt not found for {task}:{category}:{sample_id} under {source_root}")


def _reference_paths(source_root: Path, task: str, sample_id: str, canonical: dict[str, Any]) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    hf_sample_id = _hf_sample_id(task, sample_id)
    for root in _source_roots(source_root):
        for task_root in _hf_task_roots(root, task):
            if task == "i2av":
                image_path = _first_existing(task_root / "images" / f"{hf_sample_id}{ext}" for ext in IMAGE_EXTENSIONS)
                if image_path:
                    reference["image_path"] = image_path.as_posix()
                    return reference
            if task == "v2av":
                video_path = _first_existing(task_root / "videos" / f"{hf_sample_id}{ext}" for ext in VIDEO_EXTENSIONS)
                if video_path:
                    reference["video_path"] = video_path.as_posix()
                    events = canonical.get("events") or []
                    if events:
                        reference["continuation_boundary_sec"] = events[0].get("start_sec")
                    return reference
    return reference


def _first_existing(candidates: Any) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _hf_task_roots(root: Path, task: str) -> list[Path]:
    task_dir = HF_TASK_DIRS.get(task)
    if not task_dir:
        return []
    if root.name == "final_json" and root.parent.name.upper() == task_dir:
        return [root.parent]
    if root.name.upper() == task_dir:
        return [root]
    return [root / task_dir]


def _hf_sample_id(task: str, sample_id: str) -> str:
    prefix = HF_TASK_DIRS.get(task.lower(), task.upper())
    sample_id = str(sample_id)
    if re.match(fr"^{prefix}_", sample_id, flags=re.IGNORECASE):
        suffix = sample_id.split("_", 1)[1]
        return f"{prefix}_{suffix}"
    if re.fullmatch(r"\d+", sample_id):
        return f"{prefix}_{int(sample_id):03d}"
    return sample_id


def _source_roots(source_root: Path) -> list[Path]:
    roots = [source_root]
    dataset_root = next((parent for parent in source_root.parents if parent.name == "Dataset_TI2AV3.0"), None)
    if dataset_root:
        roots.extend(
            [
                dataset_root / "Dataset_2.0_fine_grain_shots" / "Dataset_addQA" / "Dataset4.0",
                dataset_root / "Dataset_2.0_fine_grain_shots" / "Dataset4.0",
                dataset_root / "fine_grain_need_review" / "Dataset4.0",
                dataset_root / "未命名文件夹" / "Dataset_2.0",
            ]
        )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def _parse_time_range(raw_value: str) -> tuple[float, float]:
    text = str(raw_value).strip().replace("–", "-").replace("—", "-")
    parts = [part for part in text.split("-") if part.strip()]
    if len(parts) >= 2:
        return _time_token_to_sec(parts[0]), _time_token_to_sec(parts[1])
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    raise ValueError(f"Invalid time range: {raw_value}")


def _time_token_to_sec(raw_token: str) -> float:
    token = raw_token.strip().lower().rstrip("s").rstrip("秒")
    if ":" in token:
        minutes, seconds = token.split(":", 1)
        return float(minutes) * 60.0 + float(seconds)
    return float(token)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
