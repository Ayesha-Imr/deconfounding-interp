from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deconfounding_interp.config import ConfigError, load_config_bundle
from deconfounding_interp.manifest import build_manifest, write_manifest

DEFAULT_CONFIG = "configs/experiments/main.yaml"


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
