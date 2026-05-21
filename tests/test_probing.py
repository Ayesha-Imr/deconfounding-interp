"""Tests for probing analysis: probe_with_direction."""

import numpy as np

from deconfounding_interp.analysis.probing import probe_with_direction


def test_separable_data_high_auroc():
    rng = np.random.default_rng(42)
    d = 100
    direction = np.zeros(d)
    direction[0] = 1.0
    pos = rng.normal(loc=2.0, scale=0.5, size=(50, d))
    neg = rng.normal(loc=-2.0, scale=0.5, size=(50, d))

    result = probe_with_direction(pos, neg, direction)
    assert result.auroc > 0.95
    assert result.accuracy > 0.90
    assert result.n_pos == 50
    assert result.n_neg == 50


def test_random_direction_low_auroc():
    rng = np.random.default_rng(42)
    d = 100
    # Separation only along dimension 0
    pos = rng.standard_normal((50, d))
    pos[:, 0] += 3.0
    neg = rng.standard_normal((50, d))
    neg[:, 0] -= 3.0

    # Direction orthogonal to the separation
    random_dir = np.zeros(d)
    random_dir[50] = 1.0
    result = probe_with_direction(pos, neg, random_dir)
    assert result.auroc < 0.7


def test_probe_result_fields():
    rng = np.random.default_rng(123)
    d = 20
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)
    pos = rng.standard_normal((10, d))
    neg = rng.standard_normal((10, d))

    result = probe_with_direction(pos, neg, direction)
    assert 0.0 <= result.auroc <= 1.0
    assert 0.0 <= result.accuracy <= 1.0
    assert isinstance(result.fpr, list)
    assert isinstance(result.tpr, list)
    assert len(result.fpr) == len(result.tpr)
