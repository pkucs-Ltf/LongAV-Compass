from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class BalancedComponent:
    source_column: str
    output_column: str
    scale: float

    def normalize(self, value: float) -> float:
        return value * self.scale


BALANCED_COMPONENTS: tuple[BalancedComponent, ...] = (
    BalancedComponent("event_fulfillment", "vqa_100", 100.0),
    BalancedComponent("event_realization", "vq_100", 20.0),
    BalancedComponent("long_form_structure", "cont_100", 20.0),
    BalancedComponent("transition_stability", "trans_100", 20.0),
    BalancedComponent("holistic_presentation", "hol_100", 20.0),
    BalancedComponent("text_video_alignment_clip", "tvalign_100", 100.0),
)


def compute_balanced_score(row: dict[str, Any], require_complete: bool = True) -> dict[str, Any]:
    """Compute the paper-facing 0-100 balanced score for one sample-model row."""
    output: dict[str, Any] = {}
    values: list[float] = []
    missing: list[str] = []

    for component in BALANCED_COMPONENTS:
        value = _float_or_none(row.get(component.source_column))
        if value is None:
            output[component.output_column] = ""
            missing.append(component.source_column)
            continue
        normalized = component.normalize(value)
        output[component.output_column] = round(normalized, 6)
        values.append(normalized)

    if missing and require_complete:
        output["balanced_score"] = ""
    else:
        output["balanced_score"] = round(sum(values) / len(values), 6) if values else ""
    output["balanced_missing_metrics"] = "|".join(missing)
    output["balanced_metric_count"] = len(values)
    return output


def write_balanced_scores(
    input_csv: str | Path,
    output_csv: str | Path,
    require_complete: bool = True,
) -> dict[str, Any]:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    rows = list(_read_csv(input_csv))
    enriched = [{**row, **compute_balanced_score(row, require_complete=require_complete)} for row in rows]
    _write_csv(output_csv, enriched)
    return {
        "input_csv": input_csv.as_posix(),
        "output_csv": output_csv.as_posix(),
        "row_count": len(enriched),
        "scored_count": sum(1 for row in enriched if row.get("balanced_score") not in {"", None}),
        "require_complete": require_complete,
    }


def write_grouped_balanced_scores(
    input_csv: str | Path,
    output_csv: str | Path,
    group_by: Iterable[str],
    score_column: str = "balanced_score",
) -> dict[str, Any]:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    group_columns = tuple(group_by)
    groups: dict[tuple[str, ...], list[float]] = {}

    for row in _read_csv(input_csv):
        score = _float_or_none(row.get(score_column))
        if score is None:
            continue
        key = tuple(str(row.get(column, "")) for column in group_columns)
        groups.setdefault(key, []).append(score)

    rows: list[dict[str, Any]] = []
    for key, scores in sorted(groups.items()):
        row = {column: value for column, value in zip(group_columns, key)}
        row["score"] = round(sum(scores) / len(scores), 6)
        row["n"] = len(scores)
        rows.append(row)

    _write_csv(output_csv, rows)
    return {
        "input_csv": input_csv.as_posix(),
        "output_csv": output_csv.as_posix(),
        "group_by": list(group_columns),
        "group_count": len(rows),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
