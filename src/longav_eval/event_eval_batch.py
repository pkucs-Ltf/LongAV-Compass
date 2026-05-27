from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .event_eval import run_event_evaluation


MAX_SAMPLE_WORKERS = 256


def run_event_evaluation_batch(
    sample_dirs: list[str | Path],
    output_root: str | Path,
    api_keys_path: str | Path = "configs/api_keys.yaml",
    models: list[str] | None = None,
    max_events: int | None = None,
    max_boundaries: int | None = None,
    skip_qa: bool = False,
    qa_root: str | Path | None = None,
    reuse_root: str | Path | None = None,
    only_missing: bool = False,
    skip_audio: bool = False,
    max_workers: int = 8,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    samples = [Path(path) for path in sample_dirs]
    workers = _bounded_workers(max_workers)

    statuses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_one_sample,
                sample_dir=sample_dir,
                output_root=output_root,
                api_keys_path=api_keys_path,
                models=models,
                max_events=max_events,
                max_boundaries=max_boundaries,
                skip_qa=skip_qa,
                qa_root=qa_root,
                reuse_root=Path(reuse_root) if reuse_root else None,
                only_missing=only_missing,
                skip_audio=skip_audio,
            ): sample_dir
            for sample_dir in samples
        }
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            _write_json(output_root / "batch_status.json", _batch_summary(output_root, workers, statuses))

    summary = _batch_summary(output_root, workers, statuses)
    _write_json(output_root / "batch_status.json", summary)
    _write_all_model_scores(output_root, statuses)
    return summary


def discover_sample_dirs(sample_root: str | Path, pattern: str = "t2av__*") -> list[Path]:
    root = Path(sample_root)
    return sorted(path for path in root.glob(pattern) if (path / "sample_index.json").exists())


def _run_one_sample(
    sample_dir: Path,
    output_root: Path,
    api_keys_path: str | Path,
    models: list[str] | None,
    max_events: int | None,
    max_boundaries: int | None,
    skip_qa: bool,
    qa_root: str | Path | None,
    reuse_root: Path | None,
    only_missing: bool,
    skip_audio: bool,
) -> dict[str, Any]:
    output_dir = output_root / sample_dir.name
    reuse_run = (reuse_root / sample_dir.name) if reuse_root else None
    try:
        summary = run_event_evaluation(
            sample_dir=sample_dir,
            output_dir=output_dir,
            api_keys_path=api_keys_path,
            models=models,
            max_events=max_events,
            max_boundaries=max_boundaries,
            skip_qa=skip_qa,
            qa_root=qa_root,
            reuse_run=reuse_run,
            only_missing=only_missing,
            skip_audio=skip_audio,
        )
        return {
            "sample_dir": sample_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "status": "ok",
            "sample_id": summary.get("sample_id"),
            "model_count": len(summary.get("models", [])),
        }
    except Exception as exc:  # noqa: BLE001
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        return {
            "sample_dir": sample_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "status": "error",
            "error": str(exc),
        }


def _bounded_workers(value: int) -> int:
    return max(1, min(MAX_SAMPLE_WORKERS, int(value)))


def _batch_summary(output_root: Path, workers: int, statuses: list[dict[str, Any]]) -> dict[str, Any]:
    completed = sorted(statuses, key=lambda item: item["sample_dir"])
    return {
        "output_root": output_root.as_posix(),
        "max_workers": workers,
        "sample_count": len(completed),
        "ok_count": sum(1 for item in completed if item.get("status") == "ok"),
        "error_count": sum(1 for item in completed if item.get("status") == "error"),
        "samples": completed,
    }


def _write_all_model_scores(output_root: Path, statuses: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for status in statuses:
        if status.get("status") != "ok":
            continue
        scores_path = Path(status["output_dir"]) / "model_scores.csv"
        if not scores_path.exists():
            continue
        with scores_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({"sample_output_dir": status["output_dir"], **row})
    _write_csv(output_root / "all_model_scores.csv", rows)


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
