from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deconfounding_interp.directions import (
    cosine_similarity,
    orthonormal_basis,
    project_onto_subspace,
    subspace_overlap_fraction,
)


@dataclass(frozen=True)
class SurfaceOverlapResult:
    overlap_fraction: float
    cosine_with_mean_surface: float
    basis_rank: int
    projection_norm: float


def compute_surface_overlap(
    standard_direction,
    surface_directions,
    max_rank: int | None = 5,
    variance_threshold: float | None = 0.90,
) -> SurfaceOverlapResult:
    surface = np.asarray(surface_directions, dtype=np.float64)
    if surface.ndim != 2:
        raise ValueError("surface_directions must have shape (n_directions, hidden_dim)")
    basis = orthonormal_basis(surface, max_rank=max_rank, variance_threshold=variance_threshold)
    projection = project_onto_subspace(standard_direction, basis)
    return SurfaceOverlapResult(
        overlap_fraction=subspace_overlap_fraction(standard_direction, basis),
        cosine_with_mean_surface=cosine_similarity(standard_direction, surface.mean(axis=0)),
        basis_rank=int(basis.shape[0]),
        projection_norm=float(np.linalg.norm(projection)),
    )
