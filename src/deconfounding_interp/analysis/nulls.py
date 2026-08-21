"""Null-calibrated stability checks for contrastive activation directions."""

from __future__ import annotations

from typing import Any

import numpy as np

from deconfounding_interp.directions import cosine_similarity, difference_in_means, normalize


def summarize_null(values: list[float] | np.ndarray) -> dict[str, Any]:
    """Summarize a null sample with quantiles while preserving empty status."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"status": "insufficient_samples", "n": 0}
    return {
        "status": "completed",
        "n": int(arr.size),
        "n_unique": int(np.unique(arr).size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q025": float(np.quantile(arr, 0.025)),
        "q975": float(np.quantile(arr, 0.975)),
    }


def split_half_stability_null(
    variant_activations: dict[int, dict[str, np.ndarray]],
    *,
    repeats: int = 90,
    seed: int = 17,
) -> dict[str, Any]:
    """Estimate within-variant split-half direction stability.

    Each repeat independently splits positive and negative examples within
    every eligible variant, then summarizes pairwise cosine across the
    resulting half directions. Four examples per side are required so each
    half has at least two examples. Small pilots therefore return an explicit
    insufficient-sample status rather than a misleading point estimate.
    """
    rng = np.random.default_rng(seed)
    eligible = [
        vi for vi, sides in sorted(variant_activations.items())
        if all(
            side in sides and sides[side].ndim == 2 and sides[side].shape[0] >= 4
            for side in ("pos", "neg")
        )
    ]
    null_values: list[float] = []
    for _ in range(int(repeats)):
        directions: list[np.ndarray] = []
        for vi in eligible:
            sides = variant_activations[vi]
            halfs: list[np.ndarray] = []
            for side in ("pos", "neg"):
                indices = rng.permutation(sides[side].shape[0])
                halfs.append(sides[side][indices[: indices.size // 2]])
            try:
                directions.append(normalize(difference_in_means(halfs[0], halfs[1])))
            except ValueError:
                continue
        if len(directions) >= 2:
            pairwise = np.asarray(directions) @ np.asarray(directions).T
            upper = pairwise[np.triu_indices_from(pairwise, k=1)]
            if upper.size:
                null_values.append(float(np.mean(upper)))
    return {
        "eligible_variants": eligible,
        "min_examples_per_side": 4,
        "repeats_requested": int(repeats),
        "summary": summarize_null(null_values),
    }


def label_shuffle_null(
    variant_activations: dict[int, dict[str, np.ndarray]],
    *,
    repeats: int = 90,
    seed: int = 17,
) -> dict[str, Any]:
    """Compare shuffled-label directions to the observed pooled direction."""
    pos = [sides["pos"] for sides in variant_activations.values() if "pos" in sides]
    neg = [sides["neg"] for sides in variant_activations.values() if "neg" in sides]
    if not pos or not neg:
        return {
            "repeats_requested": int(repeats),
            "summary": {"status": "insufficient_samples", "n": 0},
        }
    pooled_pos = np.concatenate(pos)
    pooled_neg = np.concatenate(neg)
    observed = normalize(difference_in_means(pooled_pos, pooled_neg))
    pooled = np.concatenate([pooled_pos, pooled_neg])
    n_pos = pooled_pos.shape[0]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(repeats)):
        labels = rng.permutation(pooled.shape[0])
        shuffled_pos = pooled[labels[:n_pos]]
        shuffled_neg = pooled[labels[n_pos:]]
        try:
            shuffled = normalize(difference_in_means(shuffled_pos, shuffled_neg))
            values.append(cosine_similarity(observed, shuffled))
        except ValueError:
            continue
    return {
        "n_positive": int(n_pos),
        "n_negative": int(pooled_neg.shape[0]),
        "repeats_requested": int(repeats),
        "summary": summarize_null(values),
    }


def run_null_audit(
    variant_activations: dict[int, dict[str, np.ndarray]],
    *,
    repeats: int = 90,
    seed: int = 17,
) -> dict[str, Any]:
    """Run both preregistered nulls and return auditable metadata."""
    return {
        "status": "completed",
        "seed": int(seed),
        "split_half": split_half_stability_null(
            variant_activations, repeats=repeats, seed=seed,
        ),
        "label_shuffle": label_shuffle_null(
            variant_activations, repeats=repeats, seed=seed + 1,
        ),
    }
