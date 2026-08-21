"""Tests for deterministic causal-control direction resolution."""

import numpy as np

from deconfounding_interp.pipelines.direction_controls import (
    resolve_direction,
    stable_control_seed,
)


def test_stable_seed_is_cross_process_deterministic():
    assert stable_control_seed(23, "qwen", "toxicity") == stable_control_seed(
        23, "qwen", "toxicity"
    )
    assert stable_control_seed(23, "qwen", "toxicity") != stable_control_seed(
        23, "llama", "toxicity"
    )


def test_sign_reversed_uses_standard_direction(tmp_path):
    standard = np.array([3.0, 4.0, 0.0])
    np.save(tmp_path / "standard.npy", standard)

    direction, metadata = resolve_direction(
        tmp_path,
        "sign_reversed",
        base_seed=23,
        model_id="qwen",
        trait_id="toxicity",
    )

    np.testing.assert_allclose(direction, -standard / 5.0)
    assert metadata["direction_source"] == "standard.npy"
    assert metadata["control_type"] == "sign_reversed"


def test_random_control_is_deterministic_and_unit_norm(tmp_path):
    np.save(tmp_path / "standard.npy", np.ones(7))

    first, first_meta = resolve_direction(
        tmp_path,
        "random",
        base_seed=23,
        model_id="qwen",
        trait_id="toxicity",
    )
    second, second_meta = resolve_direction(
        tmp_path,
        "random",
        base_seed=23,
        model_id="qwen",
        trait_id="toxicity",
    )

    np.testing.assert_array_equal(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)
    assert first_meta == second_meta
    assert first_meta["control_seed"] == stable_control_seed(23, "qwen", "toxicity")


def test_controls_prefer_leakage_safe_standard_fit(tmp_path):
    np.save(tmp_path / "standard.npy", np.array([1.0, 0.0]))
    np.save(tmp_path / "standard_fit_excluding_variant_04.npy", np.array([0.0, 2.0]))

    direction, metadata = resolve_direction(
        tmp_path,
        "sign_reversed",
        base_seed=23,
        model_id="qwen",
        trait_id="toxicity",
        fit_excluded_variant=4,
    )

    np.testing.assert_allclose(direction, np.array([0.0, -1.0]))
    assert metadata["direction_source"] == "standard_fit_excluding_variant_04.npy"
