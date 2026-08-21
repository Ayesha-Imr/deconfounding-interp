"""Tests for leakage-safe all-layer probing."""

import numpy as np

from deconfounding_interp.pipelines.layer_robustness import compute_layer_results


def test_compute_layer_results_uses_train_variants_and_controls():
    train = {}
    holdout = {}
    for layer in (2, 5):
        train[layer] = {
            "pos": [np.array([[2.0, 0.0], [3.0, 0.0]])],
            "neg": [np.array([[0.0, 0.0], [0.0, 0.0]])],
        }
        holdout[layer] = {
            "pos": np.array([[1.0, 0.0], [2.0, 0.0]]),
            "neg": np.array([[0.0, 0.0], [-1.0, 0.0]]),
        }

    result = compute_layer_results(
        train_by_layer=train,
        holdout=holdout,
        base_seed=23,
        model_id="qwen",
        trait_id="toxicity",
        train_variant_indices=[0, 1, 2, 3],
        holdout_index=4,
    )

    assert result["status"] == "completed"
    assert [row["layer"] for row in result["layers"]] == [2, 5]
    assert all(row["holdout_index"] == 4 for row in result["layers"])
    for row in result["layers"]:
        assert row["directions"]["standard"]["auroc"] == 1.0
        assert row["directions"]["sign_reversed"]["auroc"] == 0.0
        assert np.isclose(row["directions"]["random"]["direction_norm"], 1.0)


def test_compute_layer_results_blocks_without_complete_holdout():
    result = compute_layer_results(
        train_by_layer={1: {"pos": [np.ones((2, 3))], "neg": [np.zeros((2, 3))]}},
        holdout={1: {"pos": np.ones((2, 3))}},
        base_seed=23,
        model_id="qwen",
        trait_id="toxicity",
        train_variant_indices=[0, 1],
        holdout_index=2,
    )
    assert result["status"] == "blocked"
    assert result["reason"] == "no_complete_train_holdout_layers"
