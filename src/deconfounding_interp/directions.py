from __future__ import annotations

from typing import Any

import numpy as np

ArrayLike = Any


def normalize(vector: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector")
    return arr / norm


def activation_rms(activations: ArrayLike) -> float:
    """Return the RMS L2 norm of a batch of residual activations.

    Steering strengths are calibrated against this quantity rather than an
    arbitrary unit-norm direction.  This keeps the perturbation scale tied to
    the model's residual stream, which can differ substantially across models
    and layers.
    """
    arr = np.asarray(activations, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Activations must have shape (n_examples, hidden_dim)")
    if arr.shape[0] == 0:
        raise ValueError("Need at least one activation to compute RMS")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Activations must be finite")
    return float(np.sqrt(np.mean(np.sum(arr**2, axis=1))))


def calibrate_steering_scale(
    direction: ArrayLike,
    reference_activations: ArrayLike,
    target_rms_ratio: float = 0.05,
) -> float:
    """Choose a vector magnitude as a fraction of residual-stream RMS.

    ``direction`` is only used for its norm, so callers may pass either a raw
    DiM vector or a unit vector.  The returned scale is the coefficient to use
    with a unit-normalized direction.  A five-percent ratio means the added
    vector has norm equal to 5% of a typical activation vector.
    """
    if target_rms_ratio <= 0:
        raise ValueError("target_rms_ratio must be positive")
    direction_norm = float(np.linalg.norm(np.asarray(direction, dtype=np.float64)))
    if direction_norm < 1e-12:
        raise ValueError("Cannot calibrate a near-zero direction")
    return target_rms_ratio * activation_rms(reference_activations)


def difference_in_means(
    positive_activations: ArrayLike,
    negative_activations: ArrayLike,
) -> np.ndarray:
    pos = np.asarray(positive_activations, dtype=np.float64)
    neg = np.asarray(negative_activations, dtype=np.float64)
    if pos.ndim != 2 or neg.ndim != 2:
        raise ValueError("Activations must have shape (n_examples, hidden_dim)")
    if pos.shape[1] != neg.shape[1]:
        raise ValueError("Positive and negative activations must share hidden_dim")
    return pos.mean(axis=0) - neg.mean(axis=0)


def cosine_similarity(left: ArrayLike, right: ArrayLike, eps: float = 1e-12) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    denom = np.linalg.norm(left_arr) * np.linalg.norm(right_arr)
    if denom < eps:
        raise ValueError("Cannot compute cosine with a near-zero vector")
    return float(np.dot(left_arr, right_arr) / denom)


def pairwise_cosine(directions: ArrayLike) -> np.ndarray:
    arr = np.asarray(directions, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Directions must have shape (n_directions, hidden_dim)")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Cannot compute pairwise cosine with near-zero direction")
    unit = arr / norms
    return unit @ unit.T


def average_directions(directions: ArrayLike) -> np.ndarray:
    arr = np.asarray(directions, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Directions must have shape (n_directions, hidden_dim)")
    return normalize(arr.mean(axis=0))


def orthonormal_basis(
    directions: ArrayLike,
    max_rank: int | None = None,
    variance_threshold: float | None = None,
) -> np.ndarray:
    arr = np.asarray(directions, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Directions must have shape (n_directions, hidden_dim)")
    if arr.shape[0] == 0:
        raise ValueError("Need at least one direction to build a basis")

    _, singular_values, vt = np.linalg.svd(arr, full_matrices=False)
    rank = len(singular_values)
    if variance_threshold is not None:
        if not 0 < variance_threshold <= 1:
            raise ValueError("variance_threshold must be in (0, 1]")
        explained = singular_values**2
        cumulative = np.cumsum(explained) / np.sum(explained)
        rank = int(np.searchsorted(cumulative, variance_threshold) + 1)
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return vt[:rank]


def project_onto_subspace(vector: ArrayLike, basis: ArrayLike) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64)
    bas = np.asarray(basis, dtype=np.float64)
    if bas.ndim != 2:
        raise ValueError("Basis must have shape (rank, hidden_dim)")
    if bas.shape[1] != vec.shape[0]:
        raise ValueError("Basis hidden_dim must match vector hidden_dim")
    return bas.T @ (bas @ vec)


def remove_subspace(vector: ArrayLike, basis: ArrayLike) -> np.ndarray:
    return normalize(np.asarray(vector, dtype=np.float64) - project_onto_subspace(vector, basis))


def subspace_overlap_fraction(vector: ArrayLike, basis: ArrayLike, eps: float = 1e-12) -> float:
    vec = np.asarray(vector, dtype=np.float64)
    denom = float(np.dot(vec, vec))
    if denom < eps:
        raise ValueError("Cannot compute overlap for a near-zero vector")
    projection = project_onto_subspace(vec, basis)
    return float(np.dot(projection, projection) / denom)
