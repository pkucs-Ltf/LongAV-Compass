from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import dump_yaml
from .event_dataset import prepare_event_testset
from .event_eval import run_event_evaluation
from .event_eval_batch import discover_sample_dirs, run_event_evaluation_batch
from .manifests import load_manifest
from .paper_scores import write_balanced_scores, write_grouped_balanced_scores
from .planner import build_execution_plan, render_execution_plan
from .profiles import load_profile
from .registry import build_metric_registry
from .runner import run_evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongAV-Compass evaluation framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Build an execution plan for one sample/profile pair")
    plan_parser.add_argument("--manifest", required=True, help="Path to a sample manifest YAML file")
    plan_parser.add_argument("--profile", required=True, help="Path to a profile YAML file")
    plan_parser.add_argument(
        "--format",
        choices=("text", "json", "yaml"),
        default="text",
        help="Output format for the generated execution plan",
    )

    run_parser = subparsers.add_parser("run", help="Prepare media and execute enabled metrics for one sample")
    run_parser.add_argument("--manifest", required=True, help="Path to a sample manifest YAML file")
    run_parser.add_argument("--profile", required=True, help="Path to a profile YAML file")
    run_parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline config path")
    run_parser.add_argument("--api-keys", default=None, help="Override API key config path")
    run_parser.add_argument("--run-id", default=None, help="Optional run directory name")
    run_parser.add_argument(
        "--format",
        choices=("text", "json", "yaml"),
        default="text",
        help="Output format for the run summary",
    )

    subparsers.add_parser("list-metrics", help="List all registered metrics and their evaluators")
    profiles_parser = subparsers.add_parser("list-profiles", help="List available profile templates")
    profiles_parser.add_argument(
        "--directory",
        default="configs/profiles",
        help="Directory that contains profile YAML files",
    )

    prepare_event_parser = subparsers.add_parser(
        "prepare-event-testset",
        help="Copy full videos, metadata, canonical events, and event/boundary clips into a test set",
    )
    prepare_event_parser.add_argument("--output-root", required=True, help="Root output directory with model/category/sample folders")
    prepare_event_parser.add_argument("--source-root", required=True, help="Root LongAVBench dataset directory or legacy source prompt directory")
    prepare_event_parser.add_argument("--test-root", default="test_samples_event", help="Destination test-set directory")
    prepare_event_parser.add_argument(
        "--sample",
        action="append",
        required=True,
        help="Sample formatted as V2AV_001 or task:category:sample_id",
    )
    prepare_event_parser.add_argument("--boundary-margin-sec", type=float, default=2.0, help="Seconds before/after event boundary")
    prepare_event_parser.add_argument("--no-overwrite", action="store_true", help="Keep existing copied media and clips")

    event_eval_parser = subparsers.add_parser(
        "run-event-eval",
        help="Run event QA, MOS, transition, objective-video, and optional audio evaluation for one prepared sample",
    )
    event_eval_parser.add_argument("--sample-dir", required=True, help="Prepared sample directory")
    event_eval_parser.add_argument("--output-dir", required=True, help="Directory for event-evaluation reports")
    event_eval_parser.add_argument("--api-keys", default="configs/api_keys.yaml", help="API key config path")
    event_eval_parser.add_argument("--model", action="append", default=None, help="Restrict evaluation to one or more model aliases")
    event_eval_parser.add_argument("--max-events", type=int, default=None, help="Limit events for smoke tests")
    event_eval_parser.add_argument("--max-boundaries", type=int, default=None, help="Limit boundaries for smoke tests")
    event_eval_parser.add_argument("--skip-qa", action="store_true", help="Skip Event Fulfillment QA")
    event_eval_parser.add_argument(
        "--qa-root",
        default=None,
        help="Optional external QA dataset root containing fixed QA files; when omitted, local sample checklists are used or fallback checklists are generated",
    )
    event_eval_parser.add_argument("--reuse-run", default=None, help="Reuse metric results from a previous output directory")
    event_eval_parser.add_argument("--only-missing", action="store_true", help="Compute only metrics missing from the current/reuse run")
    event_eval_parser.add_argument("--skip-audio", action="store_true", help="Skip audio diagnostics and AV leaderboard scoring")

    event_eval_batch_parser = subparsers.add_parser(
        "run-event-eval-batch",
        help="Run evaluation for many prepared samples concurrently",
    )
    event_eval_batch_parser.add_argument("--sample-dir", action="append", default=None, help="Prepared sample directory; may be repeated")
    event_eval_batch_parser.add_argument("--sample-root", default=None, help="Root directory containing prepared sample directories")
    event_eval_batch_parser.add_argument("--sample-glob", default="t2av__*", help="Glob under --sample-root for sample directories")
    event_eval_batch_parser.add_argument("--output-root", required=True, help="Root directory for per-sample reports")
    event_eval_batch_parser.add_argument("--api-keys", default="configs/api_keys.yaml", help="API key config path")
    event_eval_batch_parser.add_argument("--model", action="append", default=None, help="Restrict evaluation to one or more model aliases")
    event_eval_batch_parser.add_argument("--max-events", type=int, default=None, help="Limit events for smoke tests")
    event_eval_batch_parser.add_argument("--max-boundaries", type=int, default=None, help="Limit boundaries for smoke tests")
    event_eval_batch_parser.add_argument("--skip-qa", action="store_true", help="Skip Event Fulfillment QA")
    event_eval_batch_parser.add_argument(
        "--qa-root",
        default=None,
        help="Optional external QA dataset root containing fixed QA files; when omitted, local sample checklists are used or fallback checklists are generated",
    )
    event_eval_batch_parser.add_argument("--reuse-root", default=None, help="Root containing previous per-sample output directories")
    event_eval_batch_parser.add_argument("--only-missing", action="store_true", help="Compute only metrics missing from current/reuse outputs")
    event_eval_batch_parser.add_argument("--skip-audio", action="store_true", help="Skip audio diagnostics and AV leaderboard scoring")
    event_eval_batch_parser.add_argument("--max-workers", type=int, default=8, help="Concurrent sample workers, capped at 256")

    balanced_parser = subparsers.add_parser(
        "compute-balanced",
        help="Compute paper-facing 0-100 balanced scores from a sample-model metric CSV",
    )
    balanced_parser.add_argument("--input", required=True, help="Input wide CSV with sample-model metric columns")
    balanced_parser.add_argument("--output", required=True, help="Output CSV with balanced-score component columns")
    balanced_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Average available balanced-score components instead of requiring all six shared video metrics",
    )
    balanced_parser.add_argument(
        "--group-output",
        default=None,
        help="Optional grouped summary CSV computed from the balanced-score output",
    )
    balanced_parser.add_argument(
        "--group-by",
        nargs="+",
        default=None,
        help="Columns used for --group-output, e.g. task category model",
    )
    return parser


def _cmd_plan(manifest_path: str, profile_path: str, output_format: str) -> int:
    registry = build_metric_registry()
    manifest = load_manifest(Path(manifest_path))
    profile = load_profile(Path(profile_path), set(registry))
    plan = build_execution_plan(manifest, profile, registry)

    if output_format == "json":
        print(json.dumps(plan.to_dict(), indent=2))
    elif output_format == "yaml":
        print(dump_yaml(plan.to_dict()))
    else:
        print(render_execution_plan(plan))
    return 0


def _cmd_list_metrics() -> int:
    registry = build_metric_registry()
    for metric in registry.values():
        providers = ", ".join(metric.providers)
        tasks = ", ".join(sorted(metric.tasks))
        print(f"{metric.metric_id}\t{metric.evaluator_type}\t{providers}\t{tasks}")
    return 0


def _cmd_list_profiles(directory: str) -> int:
    profile_dir = Path(directory)
    for path in sorted(profile_dir.glob("*.yaml")):
        print(path.as_posix())
    return 0


def _cmd_run(
    manifest_path: str,
    profile_path: str,
    config_path: str,
    api_keys_path: str | None,
    run_id: str | None,
    output_format: str,
) -> int:
    summary = run_evaluation(
        manifest_path=manifest_path,
        profile_path=profile_path,
        config_path=config_path,
        api_keys_path=api_keys_path,
        run_id=run_id,
    )
    if output_format == "json":
        print(json.dumps(summary.to_dict(), indent=2))
    elif output_format == "yaml":
        print(dump_yaml(summary.to_dict()))
    else:
        print(f"sample_id: {summary.sample_id}")
        print(f"profile: {summary.profile_name}")
        print(f"run_dir: {summary.run_dir}")
        print("")
        print("metric_results:")
        for metric_id, result in summary.metrics.items():
            line = f"  - {metric_id}: {result['status']}"
            if result["score"] is not None:
                line += f" score={result['score']}"
            line += f" backend={result['backend']}"
            if result.get("reason"):
                line += f" reason={result['reason']}"
            print(line)
        print("")
        print("scores:")
        for score_id, value in summary.scores.items():
            line = f"  - {score_id}: {value['status']}"
            if value.get("score") is not None:
                line += f" score={value['score']}"
            if value.get("missing_metrics"):
                line += f" missing={','.join(value['missing_metrics'])}"
            print(line)
    return 0


def _cmd_prepare_event_testset(args) -> int:
    summary = prepare_event_testset(
        output_root=args.output_root,
        source_root=args.source_root,
        test_root=args.test_root,
        samples=args.sample,
        boundary_margin_sec=args.boundary_margin_sec,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_run_event_eval(args) -> int:
    summary = run_event_evaluation(
        sample_dir=args.sample_dir,
        output_dir=args.output_dir,
        api_keys_path=args.api_keys,
        models=args.model,
        max_events=args.max_events,
        max_boundaries=args.max_boundaries,
        skip_qa=args.skip_qa,
        qa_root=args.qa_root,
        reuse_run=args.reuse_run,
        only_missing=args.only_missing,
        skip_audio=args.skip_audio,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_run_event_eval_batch(args) -> int:
    sample_dirs = [Path(path) for path in (args.sample_dir or [])]
    if args.sample_root:
        sample_dirs.extend(discover_sample_dirs(args.sample_root, args.sample_glob))
    if not sample_dirs:
        raise SystemExit("No samples provided. Use --sample-dir or --sample-root.")
    summary = run_event_evaluation_batch(
        sample_dirs=sample_dirs,
        output_root=args.output_root,
        api_keys_path=args.api_keys,
        models=args.model,
        max_events=args.max_events,
        max_boundaries=args.max_boundaries,
        skip_qa=args.skip_qa,
        qa_root=args.qa_root,
        reuse_root=args.reuse_root,
        only_missing=args.only_missing,
        skip_audio=args.skip_audio,
        max_workers=args.max_workers,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_compute_balanced(args) -> int:
    summary = write_balanced_scores(
        input_csv=args.input,
        output_csv=args.output,
        require_complete=not args.allow_partial,
    )
    if args.group_output:
        if not args.group_by:
            raise SystemExit("--group-output requires --group-by")
        summary["grouped"] = write_grouped_balanced_scores(
            input_csv=args.output,
            output_csv=args.group_output,
            group_by=args.group_by,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "plan":
        return _cmd_plan(args.manifest, args.profile, args.format)
    if args.command == "run":
        return _cmd_run(args.manifest, args.profile, args.config, args.api_keys, args.run_id, args.format)
    if args.command == "list-metrics":
        return _cmd_list_metrics()
    if args.command == "list-profiles":
        return _cmd_list_profiles(args.directory)
    if args.command == "prepare-event-testset":
        return _cmd_prepare_event_testset(args)
    if args.command == "run-event-eval":
        return _cmd_run_event_eval(args)
    if args.command == "run-event-eval-batch":
        return _cmd_run_event_eval_batch(args)
    if args.command == "compute-balanced":
        return _cmd_compute_balanced(args)
    parser.error(f"Unknown command '{args.command}'")
    return 2
