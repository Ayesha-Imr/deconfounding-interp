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
    min_layer: int = 0,
    *,
    pos_eval_activations: dict[int, np.ndarray] | None = None,
    neg_eval_activations: dict[int, np.ndarray] | None = None,
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
    min_layer:
        Skip layers below this index. Early layers tend to encode surface-level
        features that separate well by AUROC but steer poorly; setting this to
        ``num_layers // 5`` is a sensible default.

    Returns
    -------
    SweepResult with best_layer, best_auroc, and full per-layer AUROC dict.
    """
    if not pos_activations or not neg_activations:
        raise ValueError("Need at least one layer of activations for each side")
    if pos_activations.keys() != neg_activations.keys():
        raise ValueError("Layer keys must match between pos and neg activations")
    if (pos_eval_activations is None) != (neg_eval_activations is None):
        raise ValueError("Provide both evaluation sides or neither")
    if pos_eval_activations is not None and neg_eval_activations is not None:
        if pos_eval_activations.keys() != neg_eval_activations.keys():
            raise ValueError("Evaluation layer keys must match between pos and neg")

    eligible = sorted(idx for idx in pos_activations if idx >= min_layer)
    if not eligible:
        raise ValueError(
            f"No layers at or above min_layer={min_layer} "
            f"(available: {sorted(pos_activations)})"
        )

    per_layer: dict[int, float] = {}

    for layer in eligible:
        pos = np.asarray(pos_activations[layer], dtype=np.float64)
        neg = np.asarray(neg_activations[layer], dtype=np.float64)
        direction = difference_in_means(pos, neg)
        if pos_eval_activations is None:
            pos_eval = pos
            neg_eval = neg
        else:
            if layer not in pos_eval_activations or layer not in neg_eval_activations:
                raise ValueError(f"Evaluation activations are missing layer {layer}")
            pos_eval = np.asarray(pos_eval_activations[layer], dtype=np.float64)
            neg_eval = np.asarray(neg_eval_activations[layer], dtype=np.float64)
        scores = np.concatenate([pos_eval @ direction, neg_eval @ direction])
        labels = np.concatenate([
            np.ones(pos_eval.shape[0], dtype=np.uint8),
            np.zeros(neg_eval.shape[0], dtype=np.uint8),
        ])
        per_layer[layer] = float(roc_auc_score(labels, scores))

    best_layer = max(per_layer, key=per_layer.get)
    return SweepResult(
        best_layer=best_layer,
        best_auroc=per_layer[best_layer],
        all_aurocs=per_layer,
    )
