"""Quality gates for deterministic objective benchmark files.

The steering scorer answers "did the response match the declared key?".  This
module checks the stronger question we need before spending GPU time: whether
the task file itself is balanced, provenance-preserving, and hard enough to
measure a change.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from deconfounding_interp.analysis.objective_scoring import load_objective_tasks


def audit_objective_dataset(
    path: str | Path,
    *,
    min_tasks_per_trait: int = 4,
    require_provenance: bool = True,
) -> dict[str, Any]:
    """Return a deterministic, machine-readable quality report.

    A dataset may be useful as an exploratory candidate while still failing
    the freeze gate.  ``status=passed`` therefore means only that the file is
    structurally ready for a smoke run; it does not claim that a model has
    non-ceiling behavior on it.
    """

    dataset_path = Path(path)
    raw = json.loads(dataset_path.read_text())
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(raw, dict):
        return {
            "status": "failed",
            "errors": ["top-level dataset must be a JSON object"],
            "warnings": [],
            "path": str(dataset_path),
        }

    for key in ("dataset_id", "version", "tasks"):
        if key not in raw:
            errors.append(f"missing top-level field: {key}")
    if not isinstance(raw.get("tasks"), list):
        errors.append("top-level tasks must be a list")
        raw["tasks"] = []

    try:
        tasks = load_objective_tasks(dataset_path)
    except (ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
        tasks = []

    trait_counts = Counter(str(task.get("trait_id")) for task in tasks)
    evaluator_counts = Counter(str(task.get("evaluator")) for task in tasks)
    labels_by_trait: dict[str, Counter[str]] = defaultdict(Counter)
    normalized_questions: dict[str, list[str]] = defaultdict(list)
    source_counts = Counter()

    for task in tasks:
        task_id = str(task["task_id"])
        question = str(task["question"])
        normalized = re.sub(r"\W+", " ", question.casefold()).strip()
        normalized_questions[normalized].append(task_id)

        source = task.get("source")
        if require_provenance:
            if not isinstance(source, dict):
                errors.append(f"{task_id}: source metadata is required")
            else:
                for key in ("dataset", "item_id", "split", "license", "url"):
                    if not str(source.get(key, "")).strip():
                        errors.append(f"{task_id}: source.{key} is required")
                source_counts[str(source.get("dataset", "<missing>"))] += 1

        evaluator = str(task["evaluator"])
        if evaluator in {"claim_agreement", "claim_choice"}:
            labels_by_trait[str(task["trait_id"])][
                "claim_true" if bool(task.get("claim_truth")) else "claim_false"
            ] += 1
        elif evaluator == "choice_accuracy":
            labels_by_trait[str(task["trait_id"])][
                f"expected_{str(task['expected_option']).upper()}"
            ] += 1
        elif evaluator == "abstention_choice":
            labels_by_trait[str(task["trait_id"])][
                "abstain" if str(task.get("expected_behavior")) == "abstain" else "answer"
            ] += 1

    for normalized, task_ids in normalized_questions.items():
        if normalized and len(task_ids) > 1:
            errors.append(f"duplicate normalized questions: {', '.join(task_ids)}")

    for trait_id, count in sorted(trait_counts.items()):
        if count < min_tasks_per_trait:
            errors.append(
                f"trait {trait_id!r} has {count} tasks; minimum is {min_tasks_per_trait}"
            )

    for trait_id, labels in sorted(labels_by_trait.items()):
        if "claim_true" in labels and "claim_false" in labels:
            if labels["claim_true"] != labels["claim_false"]:
                errors.append(f"{trait_id}: claim truth labels are imbalanced: {dict(labels)}")
        if any(key.startswith("expected_") for key in labels):
            option_counts = {
                key: value for key, value in labels.items() if key.startswith("expected_")
            }
            if (
                len(option_counts) > 1
                and max(option_counts.values()) - min(option_counts.values()) > 1
            ):
                warnings.append(
                    f"{trait_id}: answer-option positions are imbalanced: {option_counts}"
                )

    review = raw.get("review", {})
    if not isinstance(review, dict):
        errors.append("review must be an object")
        review = {}
    if review.get("status") != "frozen":
        warnings.append("dataset is not frozen; it is eligible only for exploratory smoke runs")
    if int(review.get("reviewer_count", 0) or 0) < 2:
        warnings.append("fewer than two independent reviewers are recorded")

    payload = dataset_path.read_bytes()
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "facts": {
            "dataset_id": raw.get("dataset_id"),
            "version": raw.get("version"),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "task_count": len(tasks),
            "trait_counts": dict(sorted(trait_counts.items())),
            "evaluator_counts": dict(sorted(evaluator_counts.items())),
            "labels_by_trait": {
                trait: dict(sorted(labels.items()))
                for trait, labels in sorted(labels_by_trait.items())
            },
            "source_counts": dict(sorted(source_counts.items())),
            "review": review,
        },
        "path": str(dataset_path),
    }
