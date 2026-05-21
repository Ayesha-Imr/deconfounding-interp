"""Probing: project activations onto a direction and compute classification metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class ProbeResult:
    auroc: float
    accuracy: float
    optimal_threshold: float
    n_pos: int
    n_neg: int
    fpr: list[float]
    tpr: list[float]


def probe_with_direction(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    direction: np.ndarray,
) -> ProbeResult:
    pos_scores = pos_activations @ direction
    neg_scores = neg_activations @ direction

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    auroc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Optimal threshold via Youden's J
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])

    preds = (scores >= best_thresh).astype(int)
    accuracy = float((preds == labels).mean())

    return ProbeResult(
        auroc=auroc,
        accuracy=accuracy,
        optimal_threshold=best_thresh,
        n_pos=len(pos_scores),
        n_neg=len(neg_scores),
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
    )
