"""Tests for data I/O utilities."""


import numpy as np

from deconfounding_interp.io import (
    load_activations,
    load_direction,
    load_llm_audit_csv,
    load_responses_json,
    load_results_json,
    save_activations,
    save_direction,
    save_llm_audit_csv,
    save_responses_json,
    save_results_json,
)


def test_activation_roundtrip(tmp_path):
    acts = {
        0: {"pos": np.random.randn(10, 64), "neg": np.random.randn(8, 64)},
        1: {"pos": np.random.randn(10, 64), "neg": np.random.randn(8, 64)},
    }
    save_activations(tmp_path / "acts", acts)
    loaded = load_activations(tmp_path / "acts")
    assert set(loaded.keys()) == {0, 1}
    np.testing.assert_array_almost_equal(loaded[0]["pos"], acts[0]["pos"])
    np.testing.assert_array_almost_equal(loaded[1]["neg"], acts[1]["neg"])


def test_activation_load_single_layer(tmp_path):
    acts = {
        0: {"pos": np.random.randn(5, 32)},
        1: {"pos": np.random.randn(5, 32)},
    }
    save_activations(tmp_path / "acts", acts)
    loaded = load_activations(tmp_path / "acts", layer=1)
    assert set(loaded.keys()) == {1}


def test_direction_roundtrip(tmp_path):
    d = np.random.randn(128)
    d = d / np.linalg.norm(d)
    save_direction(tmp_path, "standard", d)
    loaded = load_direction(tmp_path, "standard")
    np.testing.assert_array_almost_equal(loaded, d)


def test_responses_json_roundtrip(tmp_path):
    data = [{"question": "q1", "response": "r1", "score": 75}]
    path = tmp_path / "responses.json"
    save_responses_json(path, data)
    loaded = load_responses_json(path)
    assert loaded == data


def test_results_json_roundtrip(tmp_path):
    data = {"mean_cosine": 0.87, "std": 0.05}
    path = tmp_path / "results.json"
    save_results_json(path, data)
    loaded = load_results_json(path)
    assert loaded == data


def test_audit_csv_append(tmp_path):
    path = tmp_path / "audit.csv"
    records1 = [{"timestamp": "2026-01-01", "model": "gpt-4.1-mini", "prompt_hash": "abc"}]
    records2 = [{"timestamp": "2026-01-02", "model": "gpt-4.1-mini", "prompt_hash": "def"}]
    save_llm_audit_csv(path, records1)
    save_llm_audit_csv(path, records2)
    loaded = load_llm_audit_csv(path)
    assert len(loaded) == 2
    assert loaded[0]["prompt_hash"] == "abc"
    assert loaded[1]["prompt_hash"] == "def"
