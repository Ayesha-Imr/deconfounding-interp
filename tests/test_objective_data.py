"""Tests for benchmark-data quality gates."""

import json
from pathlib import Path

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


def test_audit_counts_expected_positions_for_abstention_items(tmp_path):
    def task(task_id, evaluator, expected):
        return {
            "task_id": task_id,
            "trait_id": "hallucination",
            "evaluator": evaluator,
            "options": ["A", "B", "C"],
            "expected_option": expected,
            "expected_behavior": "abstain" if evaluator == "abstention_choice" else None,
            "question": f"Question {task_id}: choose one option.",
            "source": {
                "dataset": "project_audited",
                "item_id": task_id,
                "split": "candidate",
                "license": "CC-BY-4.0",
                "url": "https://example.com/source",
            },
        }

    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({
        "dataset_id": "test-options",
        "version": 1,
        "review": {"status": "candidate", "reviewer_count": 1},
        "tasks": [
            task("a", "choice_accuracy", "A"),
            task("b", "abstention_choice", "C"),
            task("c", "abstention_choice", "C"),
            task("d", "abstention_choice", "C"),
        ],
    }))

    result = audit_objective_dataset(path)

    assert result["facts"]["expected_option_counts"] == {
        "hallucination": {"expected_A": 1, "expected_C": 3},
    }
    assert any("answer-option positions are imbalanced" in warning
               for warning in result["warnings"])


def test_candidate_v4_preserves_v1_labels_and_provenance():
    root = Path(__file__).parents[1]
    v1 = json.loads((root / "data/objective_tasks_8b_data_candidate_v1.json").read_text())
    v4 = json.loads((root / "data/objective_tasks_8b_data_candidate_v4.json").read_text())
    old = {task["task_id"]: task for task in v1["tasks"]}
    new = {task["task_id"]: task for task in v4["tasks"]}

    assert set(old) == set(new)
    for task_id, before in old.items():
        after = new[task_id]
        for field in ("trait_id", "evaluator", "claim_truth", "difficulty", "source"):
            assert after.get(field) == before.get(field), (task_id, field)
        if not task_id.startswith("hallucination_unknown_"):
            assert after["question"] == before["question"]
            assert after.get("expected_option") == before.get("expected_option")
