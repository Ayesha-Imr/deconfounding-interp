"""Fail-fast audits for immutable pipeline manifests and run artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from deconfounding_interp.provenance import sha256_file


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _phase_job_ids(manifest: dict[str, Any], phases: Iterable[str] | None) -> set[str]:
    jobs = manifest.get("jobs", [])
    phase_set = set(phases or ())
    return {
        str(job["job_id"])
        for job in jobs
        if not phase_set or job.get("phase") in phase_set
    }


def _audit_response_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: cannot parse JSON ({exc})"]
    if not isinstance(rows, list):
        return [f"{path}: response artifact must be a list"]

    required = {"question", "response", "alpha", "direction_type", "direction_scale"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"{path}[{index}]: response row must be an object")
            continue
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"{path}[{index}]: missing {','.join(missing)}")
        if not isinstance(row.get("response"), str):
            errors.append(f"{path}[{index}]: response must be a string")
        for key in ("alpha", "direction_scale"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"{path}[{index}]: {key} must be finite numeric")
        if isinstance(row.get("direction_scale"), (int, float)) and row["direction_scale"] < 0:
            errors.append(f"{path}[{index}]: direction_scale must be non-negative")
    return errors


def audit_run(
    *,
    manifest_path: str | Path,
    run_dir: str | Path,
    project_root: str | Path | None = None,
    report_root: str | Path | None = None,
    phases: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Audit a run and return structured, machine-readable findings.

    ``phases`` allows a resumable run to be audited one completed phase at a
    time. Missing response files are warnings because geometry-only runs do not
    produce downstream response artifacts.
    """

    manifest_path = Path(manifest_path)
    run_dir = Path(run_dir)
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, Any] = {}

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "errors": [f"manifest: cannot parse JSON ({exc})"],
            "warnings": [],
            "facts": {},
        }
    if not isinstance(manifest, dict) or not isinstance(manifest.get("jobs"), list):
        return {
            "status": "failed",
            "errors": ["manifest: expected an object with a jobs list"],
            "warnings": [],
            "facts": {},
        }

    metadata_path = run_dir / "run_metadata.json"
    checkpoint_path = run_dir / "checkpoint.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"run_metadata.json: cannot parse JSON ({exc})")
        metadata = {}
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"checkpoint.json: cannot parse JSON ({exc})")
        checkpoint = {}

    expected_hash = _manifest_hash(manifest)
    facts["manifest_sha256"] = expected_hash
    recorded_hash = metadata.get("manifest_sha256")
    if recorded_hash != expected_hash:
        errors.append(
            "run_metadata.json: manifest_sha256 does not match the supplied manifest"
        )

    recorded_configs = metadata.get("config_files", {})
    if isinstance(recorded_configs, dict):
        config_checks = {}
        for relative, expected in recorded_configs.items():
            path = project_root / relative
            actual = sha256_file(path) if path.exists() else None
            config_checks[relative] = actual == expected
            if actual != expected:
                errors.append(f"config hash mismatch or missing: {relative}")
        facts["config_checks"] = config_checks
    else:
        errors.append("run_metadata.json: config_files must be an object")

    all_job_ids = _phase_job_ids(manifest, phases)
    completed = checkpoint.get("completed", {})
    if not isinstance(completed, dict):
        errors.append("checkpoint.json: completed must be an object")
        completed = {}
    completed_ids = set(completed)
    unexpected = sorted(completed_ids - {str(job["job_id"]) for job in manifest["jobs"]})
    if unexpected:
        errors.append(f"checkpoint.json: unexpected job IDs: {unexpected[:5]}")
    selected_completed = completed_ids & all_job_ids
    missing = sorted(all_job_ids - selected_completed)
    if missing:
        errors.append(
            f"checkpoint.json: {len(missing)} selected jobs are incomplete"
        )
    facts["selected_job_count"] = len(all_job_ids)
    facts["selected_completed_count"] = len(selected_completed)
    facts["selected_phases"] = sorted(set(phases or ()))
    facts["code_commit"] = metadata.get("code_commit")

    response_files: list[Path] = []
    if report_root is not None:
        response_files = sorted(Path(report_root).rglob("*_responses.json"))
    for path in response_files:
        errors.extend(_audit_response_file(path))
    facts["response_file_count"] = len(response_files)
    if report_root is not None and not response_files:
        warnings.append("no downstream response artifacts found under report_root")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "facts": facts,
    }
