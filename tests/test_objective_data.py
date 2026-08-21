"""Tests for benchmark-data quality gates."""

import json

from deconfounding_interp.analysis.objective_data import audit_objective_dataset


def _task(task_id, trait_id, claim_truth, source_id):
    return {
        "task_id": task_id,
        "trait_id": trait_id,
        "evaluator": "claim_agreement",
        "claim_truth": claim_truth,
        "question": f"Is claim {task_id} true?",
        "source": {
            "dataset": "project_audited",
            "item_id": source_id,
            "split": "candidate",
            "license": "CC-BY-4.0",
            "url": "https://example.com/source",
        },
    }


def test_audit_requires_balanced_claim_labels_and_provenance(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({
        "dataset_id": "test",
        "version": 1,
        "review": {"status": "candidate", "reviewer_count": 1},
        "tasks": [
            _task("a", "sycophancy", True, "1"),
            _task("b", "sycophancy", True, "2"),
            _task("c", "sycophancy", False, "3"),
            _task("d", "sycophancy", True, "4"),
        ],
    }))

    result = audit_objective_dataset(path)

    assert result["status"] == "failed"
    assert any("imbalanced" in error for error in result["errors"])
    assert any("not frozen" in warning for warning in result["warnings"])


def test_audit_reports_hash_and_source_counts(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({
        "dataset_id": "test",
        "version": 1,
        "review": {"status": "frozen", "reviewer_count": 2},
        "tasks": [
            _task("a", "sycophancy", True, "1"),
            _task("b", "sycophancy", False, "2"),
            _task("c", "sycophancy", True, "3"),
            _task("d", "sycophancy", False, "4"),
        ],
    }))

    result = audit_objective_dataset(path)

    assert result["status"] == "passed"
    assert len(result["facts"]["sha256"]) == 64
    assert result["facts"]["source_counts"] == {"project_audited": 4}
