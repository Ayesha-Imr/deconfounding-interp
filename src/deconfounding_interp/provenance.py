"""Reproducibility metadata and checksum helpers for pipeline runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_checksum_manifest(directory: Path, *, pattern: str = "*.npy") -> Path:
    """Write checksums for matching files in a directory."""
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(pattern)):
        entries[path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest_path = directory / ".checksums.json"
    manifest_path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    return manifest_path


def git_commit(project_root: Path) -> str | None:
    override = os.environ.get("DECONFOUND_CODE_COMMIT")
    if override is not None:
        return override.strip() or None
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_run_metadata(bundle, manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Build non-secret metadata sufficient to identify a pipeline run."""
    source_paths = [bundle.experiment.source]
    source_paths.extend(model.source for model in bundle.models.values())
    source_paths.extend(trait.source for trait in bundle.traits.values())
    config_files = {
        str(path.relative_to(bundle.project_root)): sha256_file(path)
        for path in source_paths
        if path.exists()
    }
    questions_path = bundle.project_root / "data" / "questions.json"
    objective_tasks_ref = bundle.experiment.scoring.get("objective_tasks_path")
    objective_tasks_path = (
        bundle.project_root / str(objective_tasks_ref)
        if objective_tasks_ref else None
    )
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "experiment_id": bundle.experiment.id,
        "random_seed": bundle.experiment.random_seed,
        "code_commit": git_commit(bundle.project_root),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "config_files": config_files,
        "questions_sha256": sha256_file(questions_path) if questions_path.exists() else None,
        "objective_tasks_path": (
            str(objective_tasks_path.relative_to(bundle.project_root))
            if objective_tasks_path is not None else None
        ),
        "objective_tasks_sha256": (
            sha256_file(objective_tasks_path)
            if objective_tasks_path is not None and objective_tasks_path.exists()
            else None
        ),
        "models": {
            model_id: {
                "model_name": model.model_name,
                "tokenizer_name": model.tokenizer_name,
                "source": str(model.source.relative_to(bundle.project_root)),
            }
            for model_id, model in bundle.models.items()
        },
        "traits": sorted(bundle.traits),
        "python": sys.version,
        "platform": platform.platform(),
    }


def write_run_metadata(bundle, manifest: dict[str, Any], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_metadata.json"
    path.write_text(
        json.dumps(build_run_metadata(bundle, manifest, run_dir), indent=2, sort_keys=True)
        + "\n"
    )
    return path
