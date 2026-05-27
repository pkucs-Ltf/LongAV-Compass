from __future__ import annotations

import csv
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_api_keys
from .evaluate import _build_gemini_client
from .heuristics import (
    audio_coherence_score,
    audio_quality_score,
    image_quality_score,
    load_wav_mono,
    map_correlation_to_score,
    safe_correlation,
)
from .media_scoring import analyze_boundary_technical, _make_preview, _score_video_json
from .media import extract_segment_audio, normalize_audio, probe_audio, run_command, video_encoder_args


EVENT_REALIZATION_WEIGHTS = {
    "motion_naturalness": 0.30,
    "subject_integrity": 0.25,
    "artifact_control": 0.25,
    "visual_quality": 0.20,
}

LONG_FORM_WEIGHTS = {
    "event_order_correctness": 0.30,
    "coverage_balance": 0.25,
    "pacing_consistency": 0.25,
    "cross_event_continuity": 0.20,
}

HOLISTIC_WEIGHTS = {
    "style_consistency": 0.25,
    "visual_appeal": 0.25,
    "commercial_completeness": 0.25,
    "overall_watchability": 0.25,
}

AUDIO_WEIGHTS = {
    "av_sync": 0.35,
    "audq": 0.30,
    "audlong": 0.35,
}

AUDQ_WEIGHTS = {
    "audio_event_match": 0.45,
    "audio_realism": 0.35,
    "audio_artifact_control": 0.20,
}

AUDLONG_WEIGHTS = {
    "audio_continuity": 0.30,
    "ambience_stability": 0.25,
    "source_consistency": 0.25,
    "volume_stability": 0.20,
}

AUDIO_BLEND_WEIGHTS = {
    "av_sync_llm": 0.50,
    "audq_llm": 0.75,
    "audlong_llm": 0.75,
}

HF_TASK_DIRS = {
    "t2av": "T2AV",
    "i2av": "I2AV",
    "v2av": "V2AV",
}

def run_event_evaluation(
    sample_dir: str | Path,
    output_dir: str | Path,
    api_keys_path: str | Path = "configs/api_keys.yaml",
    models: list[str] | None = None,
    max_events: int | None = None,
    max_boundaries: int | None = None,
    skip_qa: bool = False,
    qa_root: str | Path | None = None,
    reuse_run: str | Path | None = None,
    only_missing: bool = False,
    skip_audio: bool = False,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_index = _load_json(sample_dir / "sample_index.json")
    canonical = _load_json(sample_dir / "canonical_events.json")
    client = _build_gemini_client(load_api_keys(api_keys_path))
    if client is None:
        raise RuntimeError("No usable Gemini client configured.")

    checklists = None
    qa_source = None
    if not skip_qa:
        qa_root_path = Path(qa_root) if qa_root else None
        checklists, qa_source = _load_event_checklists(
            client=client,
            sample_dir=sample_dir,
            sample_index=sample_index,
            canonical=canonical,
            output_dir=output_dir,
            qa_root=qa_root_path,
        )
    selected_models = set(models or [])
    reuse_run_path = Path(reuse_run) if reuse_run else None

    model_rows: list[dict[str, Any]] = []
    fulfillment_rows: list[dict[str, Any]] = []
    realization_rows: list[dict[str, Any]] = []
    longform_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    holistic_rows: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    audio_rows: list[dict[str, Any]] = []
    audio_event_rows: list[dict[str, Any]] = []

    for model_entry in sample_index["models"]:
        model = model_entry["model"]
        if selected_models and model not in selected_models:
            continue

        model_dir = sample_dir / model
        model_output_dir = output_dir / model
        model_output_dir.mkdir(parents=True, exist_ok=True)

        events_manifest = _load_json(model_dir / "events_manifest.json")
        events = events_manifest["events"][:max_events]
        boundaries = events_manifest["boundaries"][:max_boundaries]
        video_path = Path(events_manifest["video_path"])
        cached_results = _load_cached_model_results(model, output_dir, reuse_run_path) if only_missing else []
        metric_sources: dict[str, str] = {}

        if skip_qa:
            fulfillment = {"event_scores": [], "question_scores": []}
            metric_sources["event_fulfillment"] = "skipped"
        else:
            cached = _cached_event_fulfillment(cached_results, len(events))
            if cached is not None:
                fulfillment = cached
                metric_sources["event_fulfillment"] = "reused"
            else:
                fulfillment = _score_event_fulfillment(client, model, checklists or {"events": []}, events, model_output_dir)
                metric_sources["event_fulfillment"] = "computed"

        cached_realization = _cached_list_metric(cached_results, "event_realization", len(events))
        if cached_realization is not None:
            realization = cached_realization
            metric_sources["event_realization"] = "reused"
        else:
            realization = _score_event_realization(client, model, canonical, events, model_output_dir)
            metric_sources["event_realization"] = "computed"

        cached_long_form = _cached_dict_metric(cached_results, "long_form_structure", "long_form_structure")
        if cached_long_form is not None:
            long_form = cached_long_form
            metric_sources["long_form_structure"] = "reused"
        else:
            long_form = _score_long_form_structure(client, model, canonical, video_path, model_output_dir)
            metric_sources["long_form_structure"] = "computed"

        cached_transitions = _cached_list_metric(cached_results, "transition_stability", len(boundaries))
        if cached_transitions is not None:
            transitions = cached_transitions
            metric_sources["transition_stability"] = "reused"
        else:
            transitions = _score_transition_stability(client, model, boundaries, model_output_dir)
            metric_sources["transition_stability"] = "computed"

        cached_holistic = _cached_dict_metric(cached_results, "holistic_presentation", "holistic_presentation")
        if cached_holistic is not None:
            holistic = cached_holistic
            metric_sources["holistic_presentation"] = "reused"
        else:
            holistic = _score_holistic_presentation(client, model, canonical, video_path, model_output_dir)
            metric_sources["holistic_presentation"] = "computed"

        cached_objective = _cached_dict_metric(cached_results, "objective_video_quality", "objective_video_quality_norm")
        if cached_objective is not None:
            objective = cached_objective
            metric_sources["objective_video_quality"] = "reused"
        else:
            objective = _score_objective_video_quality(video_path, model_output_dir)
            metric_sources["objective_video_quality"] = "computed"

        cached_alignment = _cached_dict_metric(cached_results, "text_video_alignment", "text_video_alignment_norm")
        if cached_alignment is not None:
            alignment = cached_alignment
            metric_sources["text_video_alignment"] = "reused"
        else:
            alignment = _score_text_video_alignment(client, model, canonical, video_path, model_output_dir)
            metric_sources["text_video_alignment"] = "computed"

        if skip_audio:
            audio_eval = _skipped_audio_evaluation("Audio evaluation disabled for this run.")
            metric_sources["audio_evaluation"] = "skipped"
        else:
            cached_audio = _cached_dict_metric(cached_results, "audio_evaluation", "status")
            if cached_audio is not None:
                audio_eval = cached_audio
                metric_sources["audio_evaluation"] = "reused"
            else:
                audio_eval = _score_audio_evaluation(client, model, canonical, events, video_path, model_output_dir)
                metric_sources["audio_evaluation"] = "computed"

        event_fulfillment = None if skip_qa else _weighted_event_average(fulfillment["event_scores"], "event_fulfillment")
        event_realization = _weighted_event_average(realization, "event_realization")
        transition_stability = _average([item["transition_stability"] for item in transitions]) if transitions else 5.0
        long_form_structure = float(long_form["long_form_structure"])
        holistic_presentation = float(holistic["holistic_presentation"])
        objective_video_quality = float(objective["objective_video_quality_norm"])
        text_video_alignment = float(alignment["text_video_alignment_norm"])

        components = {
            "event_fulfillment": 0.0 if event_fulfillment is None else event_fulfillment,
            "event_realization": _norm_mos(event_realization),
            "long_form_structure": _norm_mos(long_form_structure),
            "transition_stability": _norm_mos(transition_stability),
            "holistic_presentation": _norm_mos(holistic_presentation),
            "objective_video_quality": objective_video_quality,
            "text_video_alignment": text_video_alignment,
        }
        audio_score = _optional_float(audio_eval.get("audio_score"))

        model_summary = {
            "sample_id": sample_index["sample_id"],
            "model": model,
            "audio_available": bool(audio_eval.get("audio_available")),
            "audio_score": None if audio_score is None else round(audio_score, 4),
            "av_sync_mos": _round_optional(audio_eval.get("av_sync_mos")),
            "audq_mos": _round_optional(audio_eval.get("audq_mos")),
            "audlong_mos": _round_optional(audio_eval.get("audlong_mos")),
            "av_sync_norm": _round_optional(audio_eval.get("av_sync_norm")),
            "audq_norm": _round_optional(audio_eval.get("audq_norm")),
            "audlong_norm": _round_optional(audio_eval.get("audlong_norm")),
            "event_fulfillment": None if event_fulfillment is None else round(event_fulfillment, 4),
            "event_realization": round(event_realization, 4),
            "long_form_structure": round(long_form_structure, 4),
            "transition_stability": round(transition_stability, 4),
            "holistic_presentation": round(holistic_presentation, 4),
            "objective_video_quality": round(objective_video_quality, 4),
            "text_video_alignment": round(text_video_alignment, 4),
            "event_count": len(events),
            "boundary_count": len(transitions),
            "qa_included": not skip_qa,
            "qa_source": qa_source,
            "metric_sources": json.dumps(metric_sources, ensure_ascii=False, sort_keys=True),
            "objective_backend": objective["backend"],
            "alignment_backend": alignment["backend"],
        }
        model_rows.append(model_summary)

        fulfillment_rows.extend({"model": model, **row} for row in fulfillment["question_scores"])
        realization_rows.extend({"model": model, **row} for row in realization)
        longform_rows.append({"model": model, **long_form})
        transition_rows.extend({"model": model, **row} for row in transitions)
        holistic_rows.append({"model": model, **holistic})
        objective_rows.append({"model": model, **objective})
        alignment_rows.append({"model": model, **alignment})
        audio_rows.append({"model": model, **_audio_summary_row(audio_eval)})
        audio_event_rows.extend({"model": model, **row} for row in audio_eval.get("audio_event_scores", []))

        _write_json(
            model_output_dir / "model_summary.json",
            {
                "summary": model_summary,
                "normalized_components": components,
                "metric_sources": metric_sources,
                "event_fulfillment": fulfillment,
                "event_realization": realization,
                "long_form_structure": long_form,
                "transition_stability": transitions,
                "holistic_presentation": holistic,
                "objective_video_quality": objective,
                "text_video_alignment": alignment,
                "audio_evaluation": audio_eval,
            },
        )

    model_rows.sort(key=lambda item: item["model"])
    summary = {
        "sample_id": sample_index["sample_id"],
        "sample_dir": sample_dir.as_posix(),
        "qa_included": not skip_qa,
        "qa_source": qa_source,
        "reuse_run": reuse_run_path.as_posix() if reuse_run_path else None,
        "only_missing": only_missing,
        "audio_included": not skip_audio,
        "audio_weights": AUDIO_WEIGHTS,
        "audio_score_formula": "100 * (0.35 * av_sync_norm + 0.30 * audq_norm + 0.35 * audlong_norm)",
        "balanced_score_note": "Paper-facing balanced_score is computed as a post-processing step from shared video metrics with equal weights.",
        "models": model_rows,
    }
    _write_json(output_dir / "event_eval_summary.json", summary)
    _write_csv(output_dir / "model_scores.csv", model_rows)
    _write_csv(output_dir / "event_fulfillment_scores.csv", fulfillment_rows)
    _write_csv(output_dir / "event_realization_scores.csv", realization_rows)
    _write_csv(output_dir / "longform_scores.csv", longform_rows)
    _write_csv(output_dir / "transition_scores.csv", transition_rows)
    _write_csv(output_dir / "holistic_scores.csv", holistic_rows)
    _write_csv(output_dir / "objective_video_metrics.csv", objective_rows)
    _write_csv(output_dir / "text_video_alignment.csv", alignment_rows)
    _write_csv(output_dir / "audio_scores.csv", audio_rows)
    _write_csv(output_dir / "audio_event_scores.csv", audio_event_rows)
    return summary


def _load_cached_model_results(model: str, output_dir: Path, reuse_run: Path | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in (output_dir, reuse_run):
        if root is None:
            continue
        path = root / model / "model_summary.json"
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        try:
            candidates.append(_load_json(path))
        except (OSError, json.JSONDecodeError):
            continue
    return candidates


def _cached_event_fulfillment(candidates: list[dict[str, Any]], expected_events: int) -> dict[str, Any] | None:
    for candidate in candidates:
        value = candidate.get("event_fulfillment")
        if not isinstance(value, dict):
            continue
        event_scores = value.get("event_scores")
        question_scores = value.get("question_scores")
        if not isinstance(event_scores, list) or len(event_scores) != expected_events:
            continue
        if expected_events > 0 and not question_scores:
            continue
        return value
    return None


def _cached_list_metric(candidates: list[dict[str, Any]], metric_name: str, expected_count: int) -> list[dict[str, Any]] | None:
    for candidate in candidates:
        value = candidate.get(metric_name)
        if isinstance(value, list) and len(value) == expected_count:
            return value
    return None


def _cached_dict_metric(candidates: list[dict[str, Any]], metric_name: str, required_key: str) -> dict[str, Any] | None:
    for candidate in candidates:
        value = candidate.get(metric_name)
        if isinstance(value, dict) and value.get(required_key) is not None:
            return value
    return None


def _load_event_checklists(
    client,
    sample_dir: Path,
    sample_index: dict[str, Any],
    canonical: dict[str, Any],
    output_dir: Path,
    qa_root: Path | None,
) -> tuple[dict[str, Any], str]:
    local_path = sample_dir / "event_checklists.json"
    if local_path.exists():
        checklists = _normalize_checklists(_load_json(local_path), canonical)
        _write_json(output_dir / "event_checklists.json", checklists)
        return checklists, local_path.as_posix()

    dataset4_path = _dataset4_qa_path(sample_index, qa_root)
    if dataset4_path and dataset4_path.exists():
        checklists = _checklists_from_dataset4(_load_json(dataset4_path), sample_index)
        _write_json(output_dir / "event_checklists.json", checklists)
        return checklists, dataset4_path.as_posix()

    cache_path = output_dir / "event_checklists.json"
    checklists = _load_or_create_event_checklists(client, canonical, cache_path)
    return checklists, "generated_fallback"


def _dataset4_qa_path(sample_index: dict[str, Any], qa_root: Path | None) -> Path | None:
    if qa_root is None:
        return None
    task = str(sample_index.get("task") or "").lower()
    category = str(sample_index.get("category") or "")
    sample_id = str(sample_index.get("source_sample_id") or "")
    if not task or not category or not sample_id:
        return None
    candidates: list[Path] = []
    task_dir = HF_TASK_DIRS.get(task)
    if task_dir:
        hf_sample_id = _hf_sample_id(task, sample_id)
        for task_root in _hf_task_roots(qa_root, task):
            candidates.append(task_root / "final_json" / f"{hf_sample_id}.json")
    if task == "t2av":
        candidates.append(qa_root / "T2AVdataset" / "Real_video" / "final" / category / f"{sample_id}.json")
    if task == "i2av":
        candidates.append(qa_root / "I2AV_dataset" / "Real_video" / "I2AV_Real" / "final" / category / f"{sample_id}.json")
    return next((candidate for candidate in candidates if candidate.exists()), None)


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


def _checklists_from_dataset4(data: dict[str, Any], sample_index: dict[str, Any]) -> dict[str, Any]:
    stage = data.get("stage1") or data
    events = stage.get("events") or []
    normalized_events = []
    for index, event in enumerate(events, start=1):
        event_id = _event_id(event.get("event_id"), index)
        normalized_events.append(
            {
                "event_id": event_id,
                "time_range": event.get("time_range"),
                "event_text": event.get("visual_description") or event.get("action") or "",
                "questions": _normalize_questions(event_id, event.get("questions") or []),
            }
        )
    return {
        "schema": "qa",
        "sample_id": sample_index.get("source_sample_id"),
        "task": sample_index.get("task"),
        "category": sample_index.get("category"),
        "status": "frozen",
        "source": "external_qa_dataset",
        "events": normalized_events,
    }


def _normalize_checklists(checklists: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    events = []
    for index, item in enumerate(checklists.get("events", []), start=1):
        event_id = _event_id(item.get("event_id"), index)
        events.append(
            {
                **item,
                "event_id": event_id,
                "questions": _normalize_questions(event_id, item.get("questions") or []),
            }
        )
    return {**checklists, "events": events}


def _event_id(raw_value: Any, fallback_index: int) -> str:
    if raw_value is None or raw_value == "":
        return f"e{fallback_index}"
    text = str(raw_value)
    return text if text.startswith("e") else f"e{text}"


def _load_or_create_event_checklists(client, canonical: dict[str, Any], cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        return _load_json(cache_path)

    checklists: dict[str, Any] = {"events": []}
    for event in canonical.get("events", []):
        prompt = (
            "Create a compact visual QA checklist for evaluating whether one generated event clip fulfills the prompt.\n"
            "Use 3 to 6 atomic yes/partial/no questions. Each question must be visually verifiable from the clip.\n"
            "Do not include audio questions. Do not ask subjective quality questions.\n\n"
            f"Global description:\n{canonical.get('global_description', '')}\n\n"
            f"Event id: {event.get('event_id')}\n"
            f"Event text:\n{event.get('text', '')}\n\n"
            "Return strict JSON with this schema: "
            '{"event_id": "e1", "questions": [{"id": "e1_q1", "question": "...", "weight": 1.0}]}'
        )
        try:
            payload = client.score_media(prompt, [], [])
            questions = _normalize_questions(event["event_id"], payload.get("questions", []))
            if not questions:
                questions = _fallback_questions(event)
        except Exception:  # noqa: BLE001
            questions = _fallback_questions(event)
        checklists["events"].append({"event_id": event["event_id"], "questions": questions})

    _write_json(cache_path, checklists)
    return checklists


def _score_event_fulfillment(
    client,
    model: str,
    checklists: dict[str, Any],
    events: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    checklist_by_event = {item["event_id"]: item["questions"] for item in checklists.get("events", [])}
    question_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for event in events:
        questions = checklist_by_event.get(event["event_id"]) or _fallback_questions(event)
        preview = _make_preview(Path(event["video_path"]), output_dir / "previews" / f"{event['event_id']}_fulfillment.mp4")
        prompt = (
            "Evaluate event fulfillment for this single event clip.\n"
            "Answer each checklist question based only on visible content in the video.\n"
            "Use answer exactly one of: yes, partial, no. yes=fully satisfied, partial=partly/uncertain, no=not satisfied.\n\n"
            f"Event text:\n{event.get('text', '')}\n\n"
            f"Checklist:\n{json.dumps(questions, ensure_ascii=False)}\n\n"
            "Return strict JSON: "
            '{"answers": [{"id": "e1_q1", "answer": "yes|partial|no", "reason": "short"}]}'
        )
        payload = _score_video_json(client, prompt, preview)
        answer_by_id = {str(item.get("id")): item for item in payload.get("answers", []) if isinstance(item, dict)}
        scores = []
        weighted_scores = []
        weights = []
        for question in questions:
            answer = answer_by_id.get(str(question["id"]), {})
            answer_text = str(answer.get("answer", "partial")).strip().lower()
            score = _answer_score(answer_text)
            weight = _safe_float(question.get("weight"), 1.0)
            scores.append(score)
            weighted_scores.append(score * weight)
            weights.append(weight)
            question_rows.append(
                {
                    "event_id": event["event_id"],
                    "question_id": question["id"],
                    "dimension": question.get("dimension", ""),
                    "question": question["question"],
                    "expected_answer": question.get("expected_answer", "yes"),
                    "answer": answer_text,
                    "score": round(score, 4),
                    "weight": round(weight, 4),
                    "duration_sec": _event_duration(event),
                    "reason": answer.get("reason", ""),
                }
            )
        event_score = sum(weighted_scores) / sum(weights) if sum(weights) > 0 else _average(scores)
        event_rows.append(
            {
                "event_id": event["event_id"],
                "event_fulfillment": round(event_score, 4),
                "question_count": len(questions),
                "duration_sec": _event_duration(event),
            }
        )

    return {"event_scores": event_rows, "question_scores": question_rows}


def _score_event_realization(
    client,
    model: str,
    canonical: dict[str, Any],
    events: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        preview = _make_preview(Path(event["video_path"]), output_dir / "previews" / f"{event['event_id']}_realization.mp4")
        prompt = (
            "Evaluate event realization quality for this single generated event clip.\n"
            "Use 1-5 MOS scores. 1=failed/severe defects, 3=acceptable with visible issues, 5=excellent.\n"
            "Focus on generation quality of the attempted event: motion, subject integrity, artifact control, and visual quality.\n"
            "Semantic fulfillment is scored separately, but if the target event is absent or unjudgeable, use 1-2.\n\n"
            f"Global description:\n{canonical.get('global_description', '')}\n\n"
            f"Event text:\n{event.get('text', '')}\n\n"
            "Return strict JSON: "
            '{"motion_naturalness": number, "subject_integrity": number, "artifact_control": number, '
            '"visual_quality": number, "reason": "short"}'
        )
        payload = _score_video_json(client, prompt, preview)
        row = {
            "event_id": event["event_id"],
            "motion_naturalness": _num_range(payload, "motion_naturalness", 1.0, 5.0),
            "subject_integrity": _num_range(payload, "subject_integrity", 1.0, 5.0),
            "artifact_control": _num_range(payload, "artifact_control", 1.0, 5.0),
            "visual_quality": _num_range(payload, "visual_quality", 1.0, 5.0),
            "duration_sec": _event_duration(event),
            "reason": payload.get("reason", ""),
        }
        row["event_realization"] = round(sum(EVENT_REALIZATION_WEIGHTS[key] * row[key] for key in EVENT_REALIZATION_WEIGHTS), 4)
        rows.append(row)
    return rows


def _score_long_form_structure(client, model: str, canonical: dict[str, Any], video_path: Path, output_dir: Path) -> dict[str, Any]:
    preview = _make_preview(video_path, output_dir / "previews" / "long_form_structure.mp4", fps=3, max_width=480)
    prompt = (
        "Evaluate long-form structure for this complete generated video.\n"
        "Use 1-5 MOS scores. Focus on event order, coverage balance, pacing, and cross-event continuity.\n"
        "Do not judge low-level frame quality here unless it breaks the long-form structure.\n\n"
        f"Global description:\n{canonical.get('global_description', '')}\n\n"
        f"Event list:\n{_event_lines(canonical)}\n\n"
        "Return strict JSON: "
        '{"event_order_correctness": number, "coverage_balance": number, "pacing_consistency": number, '
        '"cross_event_continuity": number, "reason": "short"}'
    )
    payload = _score_video_json(client, prompt, preview)
    row = {
        "event_order_correctness": _num_range(payload, "event_order_correctness", 1.0, 5.0),
        "coverage_balance": _num_range(payload, "coverage_balance", 1.0, 5.0),
        "pacing_consistency": _num_range(payload, "pacing_consistency", 1.0, 5.0),
        "cross_event_continuity": _num_range(payload, "cross_event_continuity", 1.0, 5.0),
        "reason": payload.get("reason", ""),
    }
    row["long_form_structure"] = round(sum(LONG_FORM_WEIGHTS[key] * row[key] for key in LONG_FORM_WEIGHTS), 4)
    return row


def _score_transition_stability(client, model: str, boundaries: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for boundary in boundaries:
        preview = _make_preview(Path(boundary["video_path"]), output_dir / "previews" / f"{boundary['boundary_id']}_transition.mp4", fps=6, max_width=480)
        technical = analyze_boundary_technical(preview)
        prompt = (
            "Evaluate transition stability for this event-boundary clip.\n"
            "Normal shot changes and event changes are allowed. Penalize only boundary defects: black frames, flashes, freezes, repeated frames, stutter, non-story deformation, broken action, or object disappearance.\n"
            "Use a 1-5 quality score. 1=severe boundary failure, 3=noticeable but tolerable defect, 5=clean/stable boundary.\n\n"
            f"Left event:\n{boundary.get('left_text', '')}\n\n"
            f"Right event:\n{boundary.get('right_text', '')}\n\n"
            "Return strict JSON: "
            '{"llm_transition_stability": number, "generation_break": boolean, "deformation": boolean, '
            '"object_disappear": boolean, "reason": "short"}'
        )
        payload = _score_video_json(client, prompt, preview)
        algorithm_score = _norm_0_100_to_mos(float(technical["technical_boundary_score"]))
        llm_score = _num_range(payload, "llm_transition_stability", 1.0, 5.0)
        transition_score = round(0.70 * algorithm_score + 0.30 * llm_score, 4)
        rows.append(
            {
                "boundary_id": boundary["boundary_id"],
                "algorithm_transition_stability": round(algorithm_score, 4),
                "llm_transition_stability": round(llm_score, 4),
                "transition_stability": transition_score,
                "technical_boundary_score_0_100": technical["technical_boundary_score"],
                "black_frame_ratio": technical["black_frame_ratio"],
                "flash_count": technical["flash_count"],
                "duplicate_frame_ratio": technical["duplicate_frame_ratio"],
                "freeze_max_sec": technical["freeze_max_sec"],
                "generation_break": payload.get("generation_break"),
                "deformation": payload.get("deformation"),
                "object_disappear": payload.get("object_disappear"),
                "reason": payload.get("reason", ""),
            }
        )
    return rows


def _score_holistic_presentation(client, model: str, canonical: dict[str, Any], video_path: Path, output_dir: Path) -> dict[str, Any]:
    preview = _make_preview(video_path, output_dir / "previews" / "holistic_presentation.mp4", fps=3, max_width=480)
    prompt = (
        "Evaluate holistic presentation for this complete generated advertising video.\n"
        "Use 1-5 MOS scores. Focus on overall presentation, style consistency, visual appeal, commercial completeness, and watchability.\n"
        "Do not duplicate event-level checklist scoring.\n\n"
        f"Global description:\n{canonical.get('global_description', '')}\n\n"
        f"Event list:\n{_event_lines(canonical)}\n\n"
        "Return strict JSON: "
        '{"style_consistency": number, "visual_appeal": number, "commercial_completeness": number, '
        '"overall_watchability": number, "reason": "short"}'
    )
    payload = _score_video_json(client, prompt, preview)
    row = {
        "style_consistency": _num_range(payload, "style_consistency", 1.0, 5.0),
        "visual_appeal": _num_range(payload, "visual_appeal", 1.0, 5.0),
        "commercial_completeness": _num_range(payload, "commercial_completeness", 1.0, 5.0),
        "overall_watchability": _num_range(payload, "overall_watchability", 1.0, 5.0),
        "reason": payload.get("reason", ""),
    }
    row["holistic_presentation"] = round(sum(HOLISTIC_WEIGHTS[key] * row[key] for key in HOLISTIC_WEIGHTS), 4)
    return row


def _score_objective_video_quality(video_path: Path, output_dir: Path) -> dict[str, Any]:
    frame_paths = _extract_sampled_frames(video_path, output_dir / "objective_frames", fps=1, max_frames=12, max_width=256)
    technical_scores = []
    aesthetic_scores = []
    for frame_path in frame_paths:
        technical, _ = image_quality_score(frame_path)
        technical_scores.append(technical)
        aesthetic_scores.append(_aesthetic_proxy_score(frame_path))

    technical_raw = _average(technical_scores)
    aesthetic_raw = _average(aesthetic_scores)
    normalized = (0.60 * technical_raw + 0.40 * aesthetic_raw) / 100.0
    return {
        "backend": "heuristic_proxy",
        "expert_backend": "not_configured",
        "video_technical_raw": round(technical_raw, 4),
        "video_aesthetic_raw": round(aesthetic_raw, 4),
        "objective_video_quality_norm": round(_clip(normalized, 0.0, 1.0), 4),
        "frame_count": len(frame_paths),
        "note": "Proxy metric used until DOVER++/aesthetic predictor is configured.",
    }


def _score_text_video_alignment(client, model: str, canonical: dict[str, Any], video_path: Path, output_dir: Path) -> dict[str, Any]:
    preview = _make_preview(video_path, output_dir / "previews" / "text_video_alignment.mp4", fps=2, max_width=480)
    prompt = (
        "Estimate text-video semantic alignment for this complete video.\n"
        "Return a score from 0.0 to 1.0. 1.0 means the video strongly matches the global prompt and event list; 0.0 means unrelated.\n"
        "This is a coarse alignment metric, not event-level QA.\n\n"
        f"Global description:\n{canonical.get('global_description', '')}\n\n"
        f"Event list:\n{_event_lines(canonical)}\n\n"
        "Return strict JSON: "
        '{"text_video_alignment": number, "reason": "short"}'
    )
    payload = _score_video_json(client, prompt, preview)
    return {
        "backend": "gemini_surrogate",
        "expert_backend": "not_configured",
        "text_video_alignment_norm": round(_num_range(payload, "text_video_alignment", 0.0, 1.0), 4),
        "reason": payload.get("reason", ""),
        "note": "Surrogate metric used until ViCLIP/VideoCLIP embedding backend is configured.",
    }


def _score_audio_evaluation(
    client,
    model: str,
    canonical: dict[str, Any],
    events: list[dict[str, Any]],
    video_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    diagnostics = _audio_diagnostics(video_path, output_dir)
    if not diagnostics.get("audio_available"):
        return _empty_audio_evaluation("no_audio", "No audio stream found in the generated video.", diagnostics)

    try:
        event_scores = _score_audio_events(client, model, events, video_path, output_dir)
        audlong = _score_audlong(client, model, canonical, events, video_path, diagnostics, output_dir)
    except Exception as exc:  # noqa: BLE001
        return _empty_audio_evaluation(f"error:{type(exc).__name__}", str(exc), diagnostics)

    av_sync_mos = _weighted_optional_event_average(event_scores, "av_sync_mos")
    audq_mos = _weighted_optional_event_average(event_scores, "audq_mos")
    audlong_mos = _optional_float(audlong.get("audlong_mos"))

    av_sync_norm = None if av_sync_mos is None else _norm_mos(av_sync_mos)
    audq_norm = None if audq_mos is None else _norm_mos(audq_mos)
    audlong_norm = None if audlong_mos is None else _norm_mos(audlong_mos)

    audio_score = None
    if av_sync_norm is not None and audq_norm is not None and audlong_norm is not None:
        audio_score = round(
            100.0
            * (
                AUDIO_WEIGHTS["av_sync"] * av_sync_norm
                + AUDIO_WEIGHTS["audq"] * audq_norm
                + AUDIO_WEIGHTS["audlong"] * audlong_norm
            ),
            4,
        )

    return {
        "status": "ok",
        "audio_available": True,
        "audio_score": audio_score,
        "av_sync_mos": None if av_sync_mos is None else round(av_sync_mos, 4),
        "audq_mos": None if audq_mos is None else round(audq_mos, 4),
        "audlong_mos": None if audlong_mos is None else round(audlong_mos, 4),
        "av_sync_norm": None if av_sync_norm is None else round(av_sync_norm, 4),
        "audq_norm": None if audq_norm is None else round(audq_norm, 4),
        "audlong_norm": None if audlong_norm is None else round(audlong_norm, 4),
        "audio_diagnostics": diagnostics,
        "audio_event_scores": event_scores,
        "audlong": audlong,
        "weights": AUDIO_WEIGHTS,
        "note": "audio_score is reported separately from paper-facing balanced_score.",
    }


def _score_audio_events(
    client,
    model: str,
    events: list[dict[str, Any]],
    video_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        event_id = event["event_id"]
        start = _safe_float(event.get("adjusted_start_sec"), 0.0)
        end = _safe_float(event.get("adjusted_end_sec"), start)
        av_clip = _make_av_segment(video_path, output_dir / "audio_event_clips" / f"{event_id}_av.mp4", start, end)
        wav_path = _make_audio_segment(video_path, output_dir / "audio_segments" / f"{event_id}.wav", start, end)

        audio_quality_raw, audio_quality_details = audio_quality_score(wav_path)
        algorithm_sync_raw, sync_details = _algorithm_av_sync_score(av_clip, wav_path, output_dir / "audio_sync_frames" / event_id)
        algorithm_av_sync_mos = _norm_0_100_to_mos(algorithm_sync_raw)
        algorithm_audq_mos = _norm_0_100_to_mos(audio_quality_raw)

        prompt = (
            "Evaluate the audio of this generated event clip.\n"
            "Use 1-5 MOS scores. 1=failed or absent audio, 3=partly acceptable, 5=excellent.\n"
            "Score only audio-related behavior, while using the visible video to judge synchronization and correspondence.\n"
            "- av_sync: whether speech, sound effects, music changes, and audible accents align with visible actions/cuts.\n"
            "- audio_event_match: whether the audio matches the event text and audio expectation.\n"
            "- audio_realism: whether the audio is natural, clear, and plausible for the scene.\n"
            "- audio_artifact_control: whether the clip avoids clipping, buzzing, abrupt silence, glitches, or repetitive loops.\n"
            "If the clip is silent when silence is not explicitly expected, score the audio fields low.\n\n"
            f"Model: {model}\n\n"
            f"Event text:\n{event.get('text', '')}\n\n"
            f"Audio expectation:\n{event.get('audio_expectation', '')}\n\n"
            "Return strict JSON: "
            '{"av_sync": number, "audio_event_match": number, "audio_realism": number, '
            '"audio_artifact_control": number, "reason": "short"}'
        )
        payload = _score_video_json(client, prompt, av_clip)
        llm_av_sync = _num_range(payload, "av_sync", 1.0, 5.0)
        audio_event_match = _num_range(payload, "audio_event_match", 1.0, 5.0)
        audio_realism = _num_range(payload, "audio_realism", 1.0, 5.0)
        audio_artifact_control = _num_range(payload, "audio_artifact_control", 1.0, 5.0)
        llm_audq = (
            AUDQ_WEIGHTS["audio_event_match"] * audio_event_match
            + AUDQ_WEIGHTS["audio_realism"] * audio_realism
            + AUDQ_WEIGHTS["audio_artifact_control"] * audio_artifact_control
        )

        av_sync_mos = round(
            AUDIO_BLEND_WEIGHTS["av_sync_llm"] * llm_av_sync
            + (1.0 - AUDIO_BLEND_WEIGHTS["av_sync_llm"]) * algorithm_av_sync_mos,
            4,
        )
        audq_mos = round(
            AUDIO_BLEND_WEIGHTS["audq_llm"] * llm_audq
            + (1.0 - AUDIO_BLEND_WEIGHTS["audq_llm"]) * algorithm_audq_mos,
            4,
        )
        rows.append(
            {
                "event_id": event_id,
                "duration_sec": _event_duration(event),
                "av_clip_path": av_clip.as_posix(),
                "audio_segment_path": wav_path.as_posix(),
                "llm_av_sync_mos": round(llm_av_sync, 4),
                "algorithm_av_sync_mos": round(algorithm_av_sync_mos, 4),
                "av_sync_mos": av_sync_mos,
                "audio_event_match": round(audio_event_match, 4),
                "audio_realism": round(audio_realism, 4),
                "audio_artifact_control": round(audio_artifact_control, 4),
                "algorithm_audq_mos": round(algorithm_audq_mos, 4),
                "audq_mos": audq_mos,
                "audio_quality_raw": round(audio_quality_raw, 4),
                "rms": round(float(audio_quality_details.get("rms", 0.0)), 6),
                "dynamic_range": round(float(audio_quality_details.get("dynamic_range", 0.0)), 6),
                "clipping_ratio": round(float(audio_quality_details.get("clipping_ratio", 0.0)), 6),
                "silence_ratio": round(float(audio_quality_details.get("silence_ratio", 0.0)), 6),
                "sync_correlation": round(float(sync_details.get("correlation", 0.0)), 6),
                "sync_raw_score": round(algorithm_sync_raw, 4),
                "reason": payload.get("reason", ""),
            }
        )
    return rows


def _score_audlong(
    client,
    model: str,
    canonical: dict[str, Any],
    events: list[dict[str, Any]],
    video_path: Path,
    diagnostics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    preview = _make_av_preview(video_path, output_dir / "audio_previews" / "full_av.mp4", fps=3, max_width=480)
    algorithm_raw = _algorithm_audlong_score(diagnostics)
    algorithm_mos = _norm_0_100_to_mos(algorithm_raw)
    prompt = (
        "Evaluate long-range audio coherence for this complete generated video with audio.\n"
        "Use 1-5 MOS scores. 1=failed/unstable, 3=acceptable with issues, 5=excellent.\n"
        "Focus on minute-scale audio behavior, not visual quality.\n"
        "- audio_continuity: whether the soundtrack avoids unexplained dropouts, hard cuts, or broken segments.\n"
        "- ambience_stability: whether background ambience/music remains coherent across events.\n"
        "- source_consistency: whether recurring voices, sound sources, and sound effects stay plausible.\n"
        "- volume_stability: whether volume, loudness, and mixing avoid abrupt jumps, clipping, or distortion.\n\n"
        f"Model: {model}\n\n"
        f"Global description:\n{canonical.get('global_description', '')}\n\n"
        f"Event audio expectations:\n{_audio_expectation_lines(events)}\n\n"
        "Return strict JSON: "
        '{"audio_continuity": number, "ambience_stability": number, "source_consistency": number, '
        '"volume_stability": number, "reason": "short"}'
    )
    payload = _score_video_json(client, prompt, preview)
    row = {
        "audio_continuity": _num_range(payload, "audio_continuity", 1.0, 5.0),
        "ambience_stability": _num_range(payload, "ambience_stability", 1.0, 5.0),
        "source_consistency": _num_range(payload, "source_consistency", 1.0, 5.0),
        "volume_stability": _num_range(payload, "volume_stability", 1.0, 5.0),
        "algorithm_audlong_mos": round(algorithm_mos, 4),
        "algorithm_audlong_raw": round(algorithm_raw, 4),
        "reason": payload.get("reason", ""),
    }
    llm_audlong = sum(AUDLONG_WEIGHTS[key] * row[key] for key in AUDLONG_WEIGHTS)
    row["llm_audlong_mos"] = round(llm_audlong, 4)
    row["audlong_mos"] = round(
        AUDIO_BLEND_WEIGHTS["audlong_llm"] * llm_audlong
        + (1.0 - AUDIO_BLEND_WEIGHTS["audlong_llm"]) * algorithm_mos,
        4,
    )
    return row


def _audio_diagnostics(video_path: Path, output_dir: Path) -> dict[str, Any]:
    info = probe_audio(video_path)
    if info is None:
        return {
            "audio_available": False,
            "source_video_path": video_path.as_posix(),
        }

    wav_path = output_dir / "audio" / "full_audio.wav"
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        normalize_audio(video_path, wav_path)
    quality_raw, quality_details = audio_quality_score(wav_path)
    coherence_raw, coherence_details = audio_coherence_score(wav_path)
    temporal = _audio_temporal_diagnostics(wav_path)
    temporal_score = _audio_temporal_score(temporal)

    return {
        "audio_available": True,
        "source_video_path": video_path.as_posix(),
        "normalized_audio_path": wav_path.as_posix(),
        "codec": info.codec,
        "duration_sec": info.duration_sec,
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "audio_quality_raw": round(quality_raw, 4),
        "audio_coherence_raw": round(coherence_raw, 4),
        "audio_temporal_score_raw": round(temporal_score, 4),
        **{key: round(float(value), 6) for key, value in quality_details.items()},
        **{key: round(float(value), 6) for key, value in coherence_details.items()},
        **temporal,
    }


def _audio_temporal_diagnostics(wav_path: Path) -> dict[str, Any]:
    audio, sample_rate = load_wav_mono(wav_path)
    if audio.size == 0 or sample_rate <= 0:
        return {
            "audio_window_count": 0,
            "long_silence_window_ratio": 1.0,
            "volume_jump_count": 0,
            "volume_jump_ratio": 0.0,
            "repeated_window_ratio": 0.0,
        }

    window = sample_rate
    chunks = [audio[start : start + window] for start in range(0, audio.size - window + 1, window)]
    if not chunks:
        chunks = [audio]
    rms_values = [float(np.sqrt(np.mean(np.square(chunk))) + 1e-8) for chunk in chunks]
    silence_flags = [value < 0.008 for value in rms_values]

    volume_jump_count = 0
    for first, second in zip(rms_values, rms_values[1:]):
        ratio = max(first, second) / max(min(first, second), 1e-5)
        if ratio > 3.5 and abs(second - first) > 0.02:
            volume_jump_count += 1

    repeated_count = 0
    comparable_count = 0
    for first, second, rms_first, rms_second in zip(chunks, chunks[1:], rms_values, rms_values[1:]):
        if rms_first < 0.01 or rms_second < 0.01:
            continue
        length = min(first.size, second.size)
        if length < sample_rate // 2:
            continue
        comparable_count += 1
        corr = safe_correlation(first[:length].tolist(), second[:length].tolist())
        if corr > 0.985:
            repeated_count += 1

    transition_count = max(0, len(rms_values) - 1)
    return {
        "audio_window_count": len(rms_values),
        "long_silence_window_ratio": round(sum(silence_flags) / len(silence_flags), 6),
        "volume_jump_count": volume_jump_count,
        "volume_jump_ratio": round(volume_jump_count / transition_count, 6) if transition_count else 0.0,
        "repeated_window_ratio": round(repeated_count / comparable_count, 6) if comparable_count else 0.0,
    }


def _audio_temporal_score(temporal: dict[str, Any]) -> float:
    penalty = (
        50.0 * float(temporal.get("long_silence_window_ratio") or 0.0)
        + 30.0 * float(temporal.get("volume_jump_ratio") or 0.0)
        + 20.0 * float(temporal.get("repeated_window_ratio") or 0.0)
    )
    return _clip(100.0 - penalty, 0.0, 100.0)


def _algorithm_audlong_score(diagnostics: dict[str, Any]) -> float:
    quality = _safe_float(diagnostics.get("audio_quality_raw"), 0.0)
    coherence = _safe_float(diagnostics.get("audio_coherence_raw"), 0.0)
    temporal = _safe_float(diagnostics.get("audio_temporal_score_raw"), 0.0)
    return _clip(0.35 * quality + 0.45 * coherence + 0.20 * temporal, 0.0, 100.0)


def _algorithm_av_sync_score(av_clip: Path, wav_path: Path, frames_dir: Path) -> tuple[float, dict[str, Any]]:
    frames = _extract_sync_frames(av_clip, frames_dir, fps=1, max_width=160)
    motion_values = []
    for first, second in zip(frames, frames[1:]):
        motion_values.append(float(np.mean(np.abs(second - first))))
    audio_values = _audio_rms_windows(wav_path, window_sec=1.0)
    if len(audio_values) == len(motion_values) + 1:
        audio_values = audio_values[1:]

    count = min(len(motion_values), len(audio_values))
    if count < 2:
        correlation = 0.0
    else:
        correlation = safe_correlation(motion_values[:count], audio_values[:count])
    score = map_correlation_to_score(correlation)
    return score, {
        "correlation": correlation,
        "motion_window_count": len(motion_values),
        "audio_window_count": len(audio_values),
        "paired_window_count": count,
    }


def _audio_rms_windows(wav_path: Path, window_sec: float) -> list[float]:
    audio, sample_rate = load_wav_mono(wav_path)
    if audio.size == 0 or sample_rate <= 0:
        return []
    window = max(1, int(sample_rate * window_sec))
    values = []
    for start in range(0, audio.size, window):
        chunk = audio[start : start + window]
        if chunk.size:
            values.append(float(np.sqrt(np.mean(np.square(chunk)))))
    return values


def _extract_sync_frames(video_path: Path, output_dir: Path, fps: int, max_width: int) -> list[np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in output_dir.glob("frame_*.jpg") if path.stem.removeprefix("frame_").isdigit())
    if not existing:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-an",
                    "-vf",
                    f"fps={fps},scale='min({max_width},iw)':-2",
                    str(tmp_dir / "frame_%04d.jpg"),
                ]
            )
            for frame in sorted(tmp_dir.glob("frame_*.jpg")):
                target = output_dir / frame.name
                target.write_bytes(frame.read_bytes())
        existing = sorted(path for path in output_dir.glob("frame_*.jpg") if path.stem.removeprefix("frame_").isdigit())
    return [_load_gray_frame(path) for path in existing]


def _make_av_segment(source: Path, output: Path, start_sec: float, end_sec: float, max_width: int = 512) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return output
    duration = max(0.1, float(end_sec) - float(start_sec))
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{float(start_sec):.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            f"scale='min({max_width},iw)':-2",
            *video_encoder_args(crf=30),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return output


def _make_audio_segment(source: Path, output: Path, start_sec: float, end_sec: float) -> Path:
    if output.exists() and output.stat().st_size > 0:
        return output
    extract_segment_audio(source, output, start_sec, end_sec)
    return output


def _make_av_preview(source: Path, output: Path, fps: int = 3, max_width: int = 480) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return output
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            f"fps={fps},scale='min({max_width},iw)':-2",
            *video_encoder_args(crf=32),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return output


def _audio_expectation_lines(events: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {item.get('event_id')}: {item.get('audio_expectation') or 'No explicit audio expectation.'}"
        for item in events
    )


def _weighted_optional_event_average(rows: list[dict[str, Any]], key: str) -> float | None:
    valid = [item for item in rows if _optional_float(item.get(key)) is not None]
    if not valid:
        return None
    total_duration = sum(float(item.get("duration_sec") or 0.0) for item in valid)
    if total_duration <= 0:
        return _average([float(item[key]) for item in valid])
    return sum(float(item[key]) * float(item.get("duration_sec") or 0.0) for item in valid) / total_duration


def _empty_audio_evaluation(status: str, reason: str, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "audio_available": bool((diagnostics or {}).get("audio_available")),
        "audio_score": None,
        "av_sync_mos": None,
        "audq_mos": None,
        "audlong_mos": None,
        "av_sync_norm": None,
        "audq_norm": None,
        "audlong_norm": None,
        "audio_diagnostics": diagnostics or {},
        "audio_event_scores": [],
        "audlong": {},
        "reason": reason,
    }


def _skipped_audio_evaluation(reason: str) -> dict[str, Any]:
    return _empty_audio_evaluation("skipped", reason)


def _audio_summary_row(audio_eval: dict[str, Any]) -> dict[str, Any]:
    diagnostics = audio_eval.get("audio_diagnostics") or {}
    audlong = audio_eval.get("audlong") or {}
    return {
        "status": audio_eval.get("status"),
        "audio_available": bool(audio_eval.get("audio_available")),
        "audio_score": _round_optional(audio_eval.get("audio_score")),
        "av_sync_mos": _round_optional(audio_eval.get("av_sync_mos")),
        "audq_mos": _round_optional(audio_eval.get("audq_mos")),
        "audlong_mos": _round_optional(audio_eval.get("audlong_mos")),
        "av_sync_norm": _round_optional(audio_eval.get("av_sync_norm")),
        "audq_norm": _round_optional(audio_eval.get("audq_norm")),
        "audlong_norm": _round_optional(audio_eval.get("audlong_norm")),
        "codec": diagnostics.get("codec"),
        "audio_duration_sec": diagnostics.get("duration_sec"),
        "sample_rate": diagnostics.get("sample_rate"),
        "channels": diagnostics.get("channels"),
        "audio_quality_raw": diagnostics.get("audio_quality_raw"),
        "audio_coherence_raw": diagnostics.get("audio_coherence_raw"),
        "audio_temporal_score_raw": diagnostics.get("audio_temporal_score_raw"),
        "rms": diagnostics.get("rms"),
        "dynamic_range": diagnostics.get("dynamic_range"),
        "clipping_ratio": diagnostics.get("clipping_ratio"),
        "silence_ratio": diagnostics.get("silence_ratio"),
        "long_silence_window_ratio": diagnostics.get("long_silence_window_ratio"),
        "volume_jump_count": diagnostics.get("volume_jump_count"),
        "volume_jump_ratio": diagnostics.get("volume_jump_ratio"),
        "repeated_window_ratio": diagnostics.get("repeated_window_ratio"),
        "llm_audlong_mos": audlong.get("llm_audlong_mos"),
        "algorithm_audlong_mos": audlong.get("algorithm_audlong_mos"),
        "reason": audio_eval.get("reason") or audlong.get("reason", ""),
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _round_optional(value: Any, digits: int = 4) -> float | None:
    number = _optional_float(value)
    return None if number is None else round(number, digits)


def _load_gray_frame(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _normalize_questions(event_id: str, questions: Any) -> list[dict[str, Any]]:
    normalized = []
    if not isinstance(questions, list):
        return normalized
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        normalized.append(
            {
                "id": str(item.get("id") or f"{event_id}_q{index}"),
                "modality": str(item.get("modality") or "visual"),
                "dimension": str(item.get("dimension") or "general"),
                "question": question,
                "answer_type": str(item.get("answer_type") or "yes_partial_no"),
                "expected_answer": str(item.get("answer") or item.get("expected_answer") or "yes"),
                "weight": _safe_float(item.get("weight"), 1.0),
                "required": bool(item.get("required", True)),
            }
        )
    return normalized[:3]


def _fallback_questions(event: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{event.get('event_id', 'event')}_q1",
            "modality": "visual",
            "dimension": "general",
            "question": f"Does the clip visually show the described event: {event.get('text', '')}",
            "answer_type": "yes_partial_no",
            "expected_answer": "yes",
            "weight": 1.0,
            "required": True,
        }
    ]


def _extract_sampled_frames(video_path: Path, output_dir: Path, fps: int, max_frames: int, max_width: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing:
        return existing[:max_frames]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps},scale='min({max_width},iw)':-2",
                "-frames:v",
                str(max_frames),
                str(tmp_dir / "frame_%04d.jpg"),
            ]
        )
        for frame in sorted(tmp_dir.glob("frame_*.jpg")):
            target = output_dir / frame.name
            target.write_bytes(frame.read_bytes())
    return sorted(output_dir.glob("frame_*.jpg"))[:max_frames]


def _aesthetic_proxy_score(frame_path: Path) -> float:
    with Image.open(frame_path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    contrast = min(float(np.std(gray)) * 4.0, 1.0)
    exposure = 1.0 - min(1.0, abs(float(np.mean(gray)) - 0.5) / 0.5)
    saturation = min(float(np.mean(np.max(arr, axis=2) - np.min(arr, axis=2))) * 2.5, 1.0)
    score = 100.0 * (0.35 * contrast + 0.35 * exposure + 0.30 * saturation)
    return _clip(score, 0.0, 100.0)


def _event_lines(canonical: dict[str, Any]) -> str:
    return "\n".join(
        f"- {item.get('event_id')}: {item.get('start_sec')}-{item.get('end_sec')}s | {item.get('text')}"
        for item in canonical.get("events", [])
    )


def _weighted_event_average(rows: list[dict[str, Any]], key: str) -> float:
    total_duration = sum(float(item.get("duration_sec") or 0.0) for item in rows)
    if total_duration <= 0:
        return _average([float(item[key]) for item in rows])
    return sum(float(item[key]) * float(item.get("duration_sec") or 0.0) for item in rows) / total_duration


def _event_duration(event: dict[str, Any]) -> float:
    return max(0.0, float(event.get("adjusted_end_sec", 0.0)) - float(event.get("adjusted_start_sec", 0.0)))


def _answer_score(answer: str) -> float:
    if answer == "yes":
        return 1.0
    if answer in {"partial", "uncertain"}:
        return 0.5
    if answer == "no":
        return 0.0
    return 0.5


def _norm_mos(score: float) -> float:
    return _clip((float(score) - 1.0) / 4.0, 0.0, 1.0)


def _norm_0_100_to_mos(score: float) -> float:
    return 1.0 + 4.0 * _clip(float(score) / 100.0, 0.0, 1.0)


def _num_range(payload: dict[str, Any], key: str, low: float, high: float) -> float:
    value = payload.get(key)
    if isinstance(value, (int, float)):
        return _clip(float(value), low, high)
    if isinstance(value, str):
        try:
            return _clip(float(value.strip()), low, high)
        except ValueError:
            pass
    raise ValueError(f"Expected numeric {key} in response: {payload}")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _average(values: list[float]) -> float:
    valid = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
