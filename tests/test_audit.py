"""Tests for the fail-fast run audit."""

import hashlib
import json

from deconfounding_interp.audit import audit_run


def _write_fixture(tmp_path, *, complete=True, response=True):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "config.yaml"
    config.write_text("id: test\n")
    manifest = {
        "experiment_id": "test",
        "jobs": [
            {"job_id": "job_a", "phase": "prompt_assets"},
            {"job_id": "job_b", "phase": "downstream_evaluation"},
        ],
    }
    manifest_path = project / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    run_dir = project / "run"
    run_dir.mkdir()
    checkpoint = {"completed": {"job_a": {"result": {"status": "completed"}}}}
    if complete:
        checkpoint["completed"]["job_b"] = {"result": {"status": "completed"}}
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint))
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "manifest_sha256": manifest_hash,
        "config_files": {"config.yaml": config_hash},
        "code_commit": "abc123",
    }))
    report_root = project / "reports"
    if response:
        report_root.mkdir()
        (report_root / "steering_standard_responses.json").write_text(json.dumps([{
            "question": "q",
            "response": "answer",
            "alpha": 0.0,
            "direction_type": "standard",
            "direction_scale": 1.0,
        }]))
    return project, manifest_path, run_dir, report_root


def test_audit_run_passes_selected_complete_phase(tmp_path):
    project, manifest, run_dir, report_root = _write_fixture(tmp_path)
    result = audit_run(
        manifest_path=manifest,
        run_dir=run_dir,
        project_root=project,
        report_root=report_root,
        phases=["downstream_evaluation"],
    )
    assert result["status"] == "passed"
    assert result["facts"]["selected_completed_count"] == 1
    assert result["facts"]["response_file_count"] == 1


def test_audit_run_fails_incomplete_selected_phase(tmp_path):
    project, manifest, run_dir, report_root = _write_fixture(tmp_path, complete=False)
    result = audit_run(
        manifest_path=manifest,
        run_dir=run_dir,
        project_root=project,
        report_root=report_root,
        phases=["downstream_evaluation"],
    )
    assert result["status"] == "failed"
    assert any("incomplete" in error for error in result["errors"])


def test_audit_run_rejects_malformed_response(tmp_path):
    project, manifest, run_dir, report_root = _write_fixture(tmp_path)
    (report_root / "steering_standard_responses.json").write_text(
        json.dumps([{"alpha": "not-a-number", "response": 3}])
    )
    result = audit_run(
        manifest_path=manifest,
        run_dir=run_dir,
        project_root=project,
        report_root=report_root,
        phases=["downstream_evaluation"],
    )
    assert result["status"] == "failed"
    assert any("missing" in error for error in result["errors"])
