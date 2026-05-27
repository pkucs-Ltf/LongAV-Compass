from __future__ import annotations

from .runtime import MetricResult
from .schemas import Profile, SampleManifest


def aggregate_scores(
    manifest: SampleManifest,
    profile: Profile,
    metric_results: dict[str, MetricResult],
) -> dict[str, dict[str, object]]:
    """Return a lightweight raw-metric report for the plugin/profile pipeline.

    Paper-facing 0-100 aggregation is intentionally handled only by
    `longav_eval.paper_scores`, where the balanced-score definition is explicit
    and reproducible from a wide CSV.
    """
    computed = sorted(metric_id for metric_id, result in metric_results.items() if result.status == "computed")
    skipped = sorted(metric_id for metric_id, result in metric_results.items() if result.status == "skipped")
    failed = sorted(metric_id for metric_id, result in metric_results.items() if result.status == "failed")
    return {
        "metric_report": {
            "status": "computed",
            "score": None,
            "task": manifest.task,
            "profile": profile.profile_name,
            "computed_metrics": computed,
            "skipped_metrics": skipped,
            "failed_metrics": failed,
            "note": "Use compute-balanced for paper-facing 0-100 balanced scores.",
        }
    }
