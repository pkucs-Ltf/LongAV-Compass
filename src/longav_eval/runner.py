from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .aggregation import aggregate_scores
from .config import load_api_keys, load_pipeline_config
from .evaluate import evaluate_plan
from .io import dump_yaml
from .manifests import load_manifest
from .planner import build_execution_plan, render_execution_plan
from .preprocess import prepare_sample
from .profiles import load_profile
from .registry import build_metric_registry
from .runtime import RunSummary


def run_evaluation(
    manifest_path: str,
    profile_path: str,
    config_path: str = "configs/pipeline.yaml",
    api_keys_path: str | None = None,
    run_id: str | None = None,
) -> RunSummary:
    registry = build_metric_registry()
    manifest = load_manifest(manifest_path)
    profile = load_profile(profile_path, set(registry))
    plan = build_execution_plan(manifest, profile, registry)

    pipeline_config = load_pipeline_config(config_path)
    runs_dir = Path(pipeline_config.get("paths", {}).get("runs_dir", "runs")).resolve()
    run_name = run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{manifest.sample_id}"
    run_dir = runs_dir / run_name
    (run_dir / "report").mkdir(parents=True, exist_ok=True)

    resolved_api_keys_path = api_keys_path or pipeline_config.get("paths", {}).get("api_keys_file", "configs/api_keys.yaml")
    api_keys = load_api_keys(resolved_api_keys_path)

    prepared = prepare_sample(manifest, run_dir)
    metric_results = evaluate_plan(manifest, plan, prepared, api_keys)
    aggregate = aggregate_scores(manifest, profile, metric_results)

    summary = RunSummary(
        sample_id=manifest.sample_id,
        profile_name=profile.profile_name,
        run_dir=run_dir.as_posix(),
        metrics={metric_id: result.to_dict() for metric_id, result in metric_results.items()},
        scores=aggregate,
    )
    _write_artifacts(run_dir, plan, prepared.to_dict(), metric_results, aggregate, summary)
    return summary


def _write_artifacts(
    run_dir: Path,
    plan,
    prepared_data: dict[str, object],
    metric_results,
    aggregate,
    summary: RunSummary,
) -> None:
    report_dir = run_dir / "report"
    with (report_dir / "plan.txt").open("w", encoding="utf-8") as handle:
        handle.write(render_execution_plan(plan))
        handle.write("\n")
    with (report_dir / "plan.yaml").open("w", encoding="utf-8") as handle:
        handle.write(dump_yaml(plan.to_dict()))
    with (report_dir / "prepared_sample.json").open("w", encoding="utf-8") as handle:
        json.dump(prepared_data, handle, indent=2)
    with (report_dir / "metric_results.json").open("w", encoding="utf-8") as handle:
        json.dump({metric_id: result.to_dict() for metric_id, result in metric_results.items()}, handle, indent=2)
    with (report_dir / "scores.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    with (report_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2)
    with (report_dir / "run_summary.yaml").open("w", encoding="utf-8") as handle:
        handle.write(dump_yaml(summary.to_dict()))
