"""Data utilities: dataset sampling for extraction and evaluation questions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_MIN_QUESTION_CHARS = 20
_MAX_QUESTION_CHARS = 500
_MAX_SCAN_ROWS = 20_000


def sample_ultrachat_questions(
    questions_path: Path,
    n_questions: int,
    seed: int,
    source_name: str = "HuggingFaceH4/ultrachat_200k",
) -> dict:
    """Sample user messages from UltraChat and save to JSON.

    Extracts the first user message from each conversation, filters
    by length, collects up to 3× the needed count, then uniformly
    samples *n_questions* from the pool using the given seed.  The
    first half are assigned to extraction questions, the second half
    to evaluation questions.

    Parameters
    ----------
    questions_path:
        Where to write ``questions.json`` (parent dirs created if needed).
    n_questions:
        Total number of questions to sample.
    seed:
        Random seed for reproducible selection.
    source_name:
        HuggingFace dataset identifier.

    Returns
    -------
    dict with keys ``source``, ``seed``, ``extraction_questions``,
    ``evaluation_questions``.
    """
    from datasets import load_dataset

    ds = load_dataset(source_name, split="train_sft", streaming=True)

    pool: list[str] = []
    for row in ds:
        messages = row.get("messages", [])
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "").strip()
            if _MIN_QUESTION_CHARS <= len(content) <= _MAX_QUESTION_CHARS:
                pool.append(content)
            break  # only first user message per conversation
        if len(pool) >= n_questions * 3:
            break
        if len(pool) >= _MAX_SCAN_ROWS:
            break

    if len(pool) < n_questions:
        raise RuntimeError(
            f"Only found {len(pool)} qualifying user messages from "
            f"{source_name} (need {n_questions}). Try a larger pool or "
            "looser length filters."
        )

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(pool), size=n_questions, replace=False)
    selected = [pool[i] for i in sorted(indices)]

    mid = n_questions // 2
    result = {
        "source": source_name,
        "seed": seed,
        "extraction_questions": selected[:mid],
        "evaluation_questions": selected[mid:],
    }

    questions_path.parent.mkdir(parents=True, exist_ok=True)
    questions_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result
