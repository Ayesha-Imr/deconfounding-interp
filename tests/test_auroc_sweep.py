import numpy as np
import pytest

from deconfounding_interp.analysis.auroc_sweep import auroc_probe_sweep


def test_sweep_picks_separable_layer():
    """When one layer cleanly separates pos/neg, sweep returns that layer."""
    rng = np.random.RandomState(42)
    n = 50
    d = 64

    pos = {0: rng.randn(n, d), 5: rng.randn(n, d) + 5.0}
    neg = {0: rng.randn(n, d), 5: rng.randn(n, d)}

    result = auroc_probe_sweep(pos, neg)
    assert result.best_layer == 5
    assert result.best_auroc > 0.95
    assert result.best_auroc >= result.all_aurocs[0]


def test_sweep_requires_matching_keys():
    pos = {0: np.zeros((5, 10))}
    neg = {1: np.zeros((5, 10))}
    with pytest.raises(ValueError, match="Layer keys must match"):
        auroc_probe_sweep(pos, neg)


def test_sweep_requires_nonempty():
    with pytest.raises(ValueError, match="at least one layer"):
        auroc_probe_sweep({}, {})


def test_sweep_all_aurocs_populated():
    pos = {0: np.zeros((5, 10)), 7: np.zeros((5, 10)), 15: np.zeros((5, 10)) + 3.0}
    neg = {0: np.zeros((5, 10)), 7: np.zeros((5, 10)), 15: np.zeros((5, 10))}
    result = auroc_probe_sweep(pos, neg)
    assert set(result.all_aurocs) == {0, 7, 15}
    assert result.best_layer == 15


def test_sweep_min_layer_excludes_early_layers():
    """min_layer filters out layers below the threshold."""
    pos = {0: np.zeros((5, 10)) + 3.0, 10: np.zeros((5, 10)) + 3.0}
    neg = {0: np.zeros((5, 10)), 10: np.zeros((5, 10))}
    result = auroc_probe_sweep(pos, neg, min_layer=5)
    assert result.best_layer == 10
    assert 0 not in result.all_aurocs


def test_sweep_min_layer_raises_when_no_eligible():
    pos = {0: np.zeros((5, 10)), 1: np.zeros((5, 10))}
    neg = {0: np.zeros((5, 10)), 1: np.zeros((5, 10))}
    with pytest.raises(ValueError, match="min_layer=5"):
        auroc_probe_sweep(pos, neg, min_layer=5)


def test_sweep_min_layer_default_includes_all():
    """Default min_layer=0 includes all layers, same as before."""
    rng = np.random.RandomState(42)
    pos = {0: rng.randn(20, 8), 5: rng.randn(20, 8) + 3.0}
    neg = {0: rng.randn(20, 8), 5: rng.randn(20, 8)}
    result = auroc_probe_sweep(pos, neg)
    assert result.best_layer == 5
    assert len(result.all_aurocs) == 2
    pos = {1: np.ones((5, 10)), 2: np.ones((5, 10))}
    neg = {1: np.zeros((5, 10)), 2: np.zeros((5, 10))}
    result = auroc_probe_sweep(pos, neg)
    assert result.best_layer in {1, 2}
    assert isinstance(result.best_auroc, float)
    with pytest.raises(AttributeError):
        result.best_layer = 99  # type: ignore[misc]
