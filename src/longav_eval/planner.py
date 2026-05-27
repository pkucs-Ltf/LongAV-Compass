from __future__ import annotations

from .schemas import ConfigError, ExecutionPlan, MetricDecision, Profile, SampleManifest, ScoreAvailability


def build_execution_plan(
    manifest: SampleManifest,
    profile: Profile,
    registry: dict[str, object],
) -> ExecutionPlan:
    available_inputs = manifest.available_inputs()
    metric_decisions: list[MetricDecision] = []
    status_by_metric: dict[str, str] = {}

    for metric_id, definition in registry.items():
        enabled = profile.metric_enabled(metric_id)
        if not enabled:
            decision = MetricDecision(
                metric_id=metric_id,
                display_name=definition.display_name,
                enabled_in_profile=False,
                status="disabled",
                evaluator_type=definition.evaluator_type,
                providers=definition.providers,
                reason=f"Disabled by profile '{profile.profile_name}'",
            )
        elif manifest.task not in definition.tasks:
            decision = MetricDecision(
                metric_id=metric_id,
                display_name=definition.display_name,
                enabled_in_profile=True,
                status="skipped",
                evaluator_type=definition.evaluator_type,
                providers=definition.providers,
                reason=f"Metric not defined for task '{manifest.task}'",
            )
        else:
            missing_inputs = tuple(sorted(definition.required_inputs - available_inputs))
            if missing_inputs:
                if not profile.allow_auto_skip:
                    raise ConfigError(
                        f"Metric '{metric_id}' is enabled but missing inputs: {', '.join(missing_inputs)}"
                    )
                decision = MetricDecision(
                    metric_id=metric_id,
                    display_name=definition.display_name,
                    enabled_in_profile=True,
                    status="skipped",
                    evaluator_type=definition.evaluator_type,
                    providers=definition.providers,
                    missing_inputs=missing_inputs,
                    reason="Required inputs are unavailable in the sample manifest",
                )
            else:
                decision = MetricDecision(
                    metric_id=metric_id,
                    display_name=definition.display_name,
                    enabled_in_profile=True,
                    status="ready",
                    evaluator_type=definition.evaluator_type,
                    providers=definition.providers,
                )
        metric_decisions.append(decision)
        status_by_metric[metric_id] = decision.status

    scores = tuple(_build_scores(manifest.task, profile, status_by_metric))
    return ExecutionPlan(
        sample_id=manifest.sample_id,
        task=manifest.task,
        profile_name=profile.profile_name,
        mode=profile.mode,
        available_inputs=tuple(sorted(available_inputs)),
        metrics=tuple(metric_decisions),
        scores=scores,
    )


def _build_scores(task: str, profile: Profile, status_by_metric: dict[str, str]) -> list[ScoreAvailability]:
    return [
        ScoreAvailability(
            score_id="metric_report",
            status="ready",
            required_metrics=tuple(sorted(metric_id for metric_id, enabled in profile.metrics.items() if enabled)),
            reason="Profile pipeline emits metric-level diagnostics. Use compute-balanced for paper-facing aggregate scores.",
        )
    ]


def render_execution_plan(plan: ExecutionPlan) -> str:
    lines: list[str] = []
    lines.append(f"sample_id: {plan.sample_id}")
    lines.append(f"task: {plan.task}")
    lines.append(f"profile: {plan.profile_name} ({plan.mode})")
    lines.append(f"available_inputs: {', '.join(plan.available_inputs)}")
    lines.append("")
    lines.append("metrics:")
    for metric in plan.metrics:
        provider_text = ", ".join(metric.providers)
        line = f"  - {metric.metric_id}: {metric.status} [{metric.evaluator_type}; {provider_text}]"
        if metric.reason:
            line += f" - {metric.reason}"
        if metric.missing_inputs:
            line += f" (missing: {', '.join(metric.missing_inputs)})"
        lines.append(line)
    lines.append("")
    lines.append("scores:")
    for score in plan.scores:
        line = f"  - {score.score_id}: {score.status}"
        if score.reason:
            line += f" - {score.reason}"
        if score.missing_metrics:
            line += f" (missing metrics: {', '.join(score.missing_metrics)})"
        lines.append(line)
    return "\n".join(lines)
