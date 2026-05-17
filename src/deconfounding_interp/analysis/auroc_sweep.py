"""Layer selection via AUROC probe sweep across all residual stream layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from deconfounding_interp.directions import difference_in_means


@dataclass(frozen=True)
class SweepResult:
    best_layer: int
    best_auroc: float
    all_aurocs: dict[int, float]


def auroc_probe_sweep(
    pos_activations: dict[int, np.ndarray],
    neg_activations: dict[int, np.ndarray],
) -> SweepResult:
    """Run AUROC probe sweep across layers to find the best separation layer.

    For each layer, computes the DiM direction between positive and negative
    response activations, then evaluates how well that direction separates the
    two groups using AUROC. Returns the layer with the highest AUROC.

    Parameters
    ----------
    pos_activations:
        Mapping from layer number to positive-side activations,
        each with shape (n_pos, hidden_dim).
    neg_activations:
        Mapping from layer number to negative-side activations,
        each with shape (n_neg, hidden_dim).

    Returns
    -------
    SweepResult with best_layer, best_auroc, and full per-layer AUROC dict.
    """
    if not pos_activations or not neg_activations:
        raise ValueError("Need at least one layer of activations for each side")
    if pos_activations.keys() != neg_activations.keys():
        raise ValueError("Layer keys must match between pos and neg activations")

    all_layers = sorted(pos_activations)
    per_layer: dict[int, float] = {}

    for layer in all_layers:
        pos = np.asarray(pos_activations[layer], dtype=np.float64)
        neg = np.asarray(neg_activations[layer], dtype=np.float64)
        direction = difference_in_means(pos, neg)
        scores = np.concatenate([pos @ direction, neg @ direction])
        labels = np.concatenate([
            np.ones(pos.shape[0], dtype=np.uint8),
            np.zeros(neg.shape[0], dtype=np.uint8),
        ])
        per_layer[layer] = float(roc_auc_score(labels, scores))

    best_layer = max(per_layer, key=per_layer.get)
    return SweepResult(
        best_layer=best_layer,
        best_auroc=per_layer[best_layer],
        all_aurocs=per_layer,
    )
