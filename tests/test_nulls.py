"""Tests for null-calibrated geometry checks."""

import numpy as np

from deconfounding_interp.analysis.nulls import (
    label_shuffle_null,
    run_null_audit,
    split_half_stability_null,
)


def test_small_pilot_is_marked_insufficient_for_split_half():
    acts = {
        0: {
            "pos": np.ones((1, 3)),
            "neg": np.zeros((1, 3)),
        },
        1: {
            "pos": np.ones((1, 3)),
            "neg": np.zeros((1, 3)),
        },
    }

    result = split_half_stability_null(acts, repeats=10)

    assert result["eligible_variants"] == []
    assert result["summary"]["status"] == "insufficient_samples"


def test_null_audit_has_reproducible_finite_summaries():
    rng = np.random.default_rng(5)
    acts = {
        vi: {
            "pos": rng.normal(loc=1.0, size=(8, 6)),
            "neg": rng.normal(loc=-1.0, size=(8, 6)),
        }
        for vi in range(3)
    }

    result = run_null_audit(acts, repeats=20, seed=17)

    assert result["split_half"]["summary"]["status"] == "completed"
    assert result["split_half"]["summary"]["n"] == 20
    assert result["label_shuffle"]["summary"]["status"] == "completed"
    assert result["label_shuffle"]["summary"]["n"] == 20
    assert np.isfinite(result["label_shuffle"]["summary"]["mean"])


def test_label_shuffle_preserves_class_counts():
    acts = {
        0: {
            "pos": np.ones((4, 2)),
            "neg": -np.ones((4, 2)),
        },
        1: {
            "pos": np.ones((4, 2)),
            "neg": -np.ones((4, 2)),
        },
    }

    result = label_shuffle_null(acts, repeats=7, seed=3)

    assert result["n_positive"] == 8
    assert result["n_negative"] == 8
    assert 0 < result["summary"]["n"] <= 7
