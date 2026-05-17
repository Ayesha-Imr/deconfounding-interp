from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deconfounding_interp.directions import pairwise_cosine


@dataclass(frozen=True)
class StabilitySummary:
    mean_cosine: float
    std_cosine: float
    min_cosine: float
    max_cosine: float
    n_pairs: int


def summarize_stability(variant_directions) -> StabilitySummary:
    matrix = pairwise_cosine(variant_directions)
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    if upper.size == 0:
        raise ValueError("Need at least two variant directions for stability analysis")
    return StabilitySummary(
        mean_cosine=float(np.mean(upper)),
        std_cosine=float(np.std(upper)),
        min_cosine=float(np.min(upper)),
        max_cosine=float(np.max(upper)),
        n_pairs=int(upper.size),
    )
