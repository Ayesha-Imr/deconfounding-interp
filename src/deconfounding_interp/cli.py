from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deconfounding_interp.config import ConfigError, load_config_bundle
from deconfounding_interp.manifest import build_manifest, write_manifest

DEFAULT_CONFIG = "configs/experiments/main.yaml"
DEFAULT_RUN_DIR = "outputs/runs/latest"


def _load_or_exit(config_path: str):
    try:
        return load_config_bundle(config_path)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def cmd_validate_config(args: argparse.Namespace) -> int:
    bundle = _load_or_exit(args.config)
    print(
        f"OK: {bundle.experiment.id} "
        f"({len(bundle.models)} models, {len(bundle.traits)} traits)"
    )
    return 0


def cmd_list_traits(args: argparse.Namespace) -> int:
    bundle = _load_or_exit(args.config)
    for trait in bundle.traits.values():
        print(f"{trait.id}\t{trait.display_name}\t{trait.kind}\t{trait.expected_surface_confound}")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    bundle = _load_or_exit(args.config)
    for model in bundle.models.values():
        print(f"{model.id}\t{model.model_name}\t{model.num_layers} layers")
    return 0


def cmd_make_manifest(args: argparse.Namespace) -> int:
    bundle = _load_or_exit(args.config)
    manifest = build_manifest(bundle)
    if args.output:
        path = write_manifest(manifest, args.output)
        print(f"Wrote {len(manifest['jobs'])} jobs to {path}")
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _load_manifest(args) -> dict:
    """Load manifest from file or build from config."""
    if hasattr(args, "manifest") and args.manifest:
        with open(args.manifest) as f:
            return json.load(f)
    bundle = _load_or_exit(args.config)
    return build_manifest(bundle)


def cmd_sample_questions(args: argparse.Namespace) -> int:
    bundle = _load_or_exit(args.config)
    gen = bundle.experiment.generation
    n_total = gen.get("extraction_questions", 20) + gen.get("evaluation_questions", 20)
    questions_path = args.output or (bundle.project_root / "data" / "questions.json")

    from deconfounding_interp.data_utils import sample_ultrachat_questions
    result = sample_ultrachat_questions(
        questions_path=Path(questions_path),
        n_questions=n_total,
        seed=bundle.experiment.random_seed,
        source_name=args.dataset,
    )
    print(
        f"Sampled {len(result['extraction_questions'])} extraction + "
        f"{len(result['evaluation_questions'])} evaluation questions "
        f"from {result['source']} (seed={result['seed']})"
    )
    print(f"Saved to {questions_path}")
    return 0


def cmd_audit_run(args: argparse.Namespace) -> int:
    from deconfounding_interp.audit import audit_run

    bundle = _load_or_exit(args.config)
    from deconfounding_interp.io import resolve_paths

    report_root = args.report_root
    if report_root is None and args.include_responses:
        report_root = resolve_paths(bundle)["report_dir"]
    result = audit_run(
        manifest_path=args.manifest,
        run_dir=args.run_dir,
        project_root=bundle.project_root,
        report_root=report_root,
        phases=args.phase,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    bundle = _load_or_exit(args.config)
    manifest = _load_manifest(args)
    run_dir = Path(args.run_dir)

    from deconfounding_interp.runner import PipelineRunner
    runner = PipelineRunner(bundle=bundle, run_dir=run_dir, dry_run=args.dry_run)
    results = runner.run_manifest(manifest)
    print(f"Completed {len(results)} jobs (run_dir={run_dir})")
    return 0


def cmd_run_stage(args: argparse.Namespace) -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    bundle = _load_or_exit(args.config)
    manifest = _load_manifest(args)
    run_dir = Path(args.run_dir)

    from deconfounding_interp.runner import PipelineRunner
    runner = PipelineRunner(bundle=bundle, run_dir=run_dir, dry_run=args.dry_run)
    results = runner.run_phase(manifest, args.phase)
    print(f"Completed {len(results)} jobs for phase={args.phase}")
    return 0


def cmd_run_job(args: argparse.Namespace) -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    bundle = _load_or_exit(args.config)
    manifest = _load_manifest(args)
    run_dir = Path(args.run_dir)

    from deconfounding_interp.runner import PipelineRunner
    runner = PipelineRunner(bundle=bundle, run_dir=run_dir, dry_run=args.dry_run)
    result = runner.run_job(manifest, args.job_id)
    print(f"Job {args.job_id}: {result.get('status', 'done')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deconfound")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate experiment config refs")
    validate.add_argument("--config", default=DEFAULT_CONFIG)
    validate.set_defaults(func=cmd_validate_config)

    traits = subparsers.add_parser("list-traits", help="List configured traits")
    traits.add_argument("--config", default=DEFAULT_CONFIG)
    traits.set_defaults(func=cmd_list_traits)

    models = subparsers.add_parser("list-models", help="List configured models")
    models.add_argument("--config", default=DEFAULT_CONFIG)
    models.set_defaults(func=cmd_list_models)

    manifest = subparsers.add_parser("make-manifest", help="Expand config into runnable jobs")
    manifest.add_argument("--config", default=DEFAULT_CONFIG)
    manifest.add_argument("--output", type=Path)
    manifest.set_defaults(func=cmd_make_manifest)

    # --- Data setup ---
    sample_q = subparsers.add_parser("sample-questions", help="Sample questions from UltraChat")
    sample_q.add_argument("--config", default=DEFAULT_CONFIG)
    sample_q.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    sample_q.add_argument("--output", type=Path, default=None)
    sample_q.set_defaults(func=cmd_sample_questions)

    audit = subparsers.add_parser(
        "audit-run",
        help="Audit manifest metadata, checkpoint completion, config hashes, and responses",
    )
    audit.add_argument("--config", default=DEFAULT_CONFIG)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--run-dir", type=Path, required=True)
    audit.add_argument(
        "--phase",
        action="append",
        help="Audit only this phase; repeat for multiple phases (default: all)",
    )
    audit.add_argument(
        "--report-root",
        type=Path,
        help="Response-artifact root; omit to skip response validation",
    )
    audit.add_argument(
        "--include-responses",
        action="store_true",
        help="Validate response files under the config's report_dir",
    )
    audit.set_defaults(func=cmd_audit_run)

    # --- Pipeline execution commands ---
    run_pipeline = subparsers.add_parser("run-pipeline", help="Run all jobs in the manifest")
    run_pipeline.add_argument("--config", default=DEFAULT_CONFIG)
    run_pipeline.add_argument("--manifest", type=Path, default=None, help="Pre-built manifest JSON")
    run_pipeline.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    run_pipeline.add_argument("--dry-run", action="store_true")
    run_pipeline.set_defaults(func=cmd_run_pipeline)

    run_stage = subparsers.add_parser("run-stage", help="Run all jobs for a single phase")
    run_stage.add_argument("--config", default=DEFAULT_CONFIG)
    run_stage.add_argument("--phase", required=True, help="Phase name (e.g. prompt_assets)")
    run_stage.add_argument("--manifest", type=Path, default=None)
    run_stage.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    run_stage.add_argument("--dry-run", action="store_true")
    run_stage.set_defaults(func=cmd_run_stage)

    run_job = subparsers.add_parser("run-job", help="Run a single job by ID")
    run_job.add_argument("--config", default=DEFAULT_CONFIG)
    run_job.add_argument("--job-id", required=True)
    run_job.add_argument("--manifest", type=Path, default=None)
    run_job.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    run_job.add_argument("--dry-run", action="store_true")
    run_job.set_defaults(func=cmd_run_job)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
