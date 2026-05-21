"""Tests for Phase 3 manifest jobs: downstream_evaluation, probing, phase3_summary."""

from unittest.mock import MagicMock

from deconfounding_interp.manifest import build_manifest


def _make_bundle():
    bundle = MagicMock()
    bundle.experiment.generation = {
        "system_prompt_pairs": 5,
        "paraphrases_per_prompt": 1,
    }
    bundle.experiment.corrections = {
        "direction_types": ["standard", "averaged", "subtracted", "single_variant"],
    }
    bundle.experiment.steering = {
        "alpha_values": [0.0, 1.0, 2.0],
    }
    bundle.experiment.random_seed = 17

    model = MagicMock()
    model.trait_layers = {"sycophancy": 20, "toxicity": None}
    bundle.models = {"qwen": model}
    bundle.traits = {"sycophancy": MagicMock(), "toxicity": MagicMock()}
    bundle.experiment.id = "test"
    return bundle


def test_manifest_has_probing_jobs():
    bundle = _make_bundle()
    manifest = build_manifest(bundle)
    jobs = manifest["jobs"]
    probing_jobs = [j for j in jobs if j["phase"] == "probing"]
    # 1 model x 2 traits = 2 probing jobs
    assert len(probing_jobs) == 2


def test_manifest_has_downstream_eval_jobs():
    bundle = _make_bundle()
    manifest = build_manifest(bundle)
    jobs = manifest["jobs"]
    eval_jobs = [j for j in jobs if j["phase"] == "downstream_evaluation"]
    # 1 model x 2 traits x 4 direction types = 8
    assert len(eval_jobs) == 8


def test_manifest_has_phase3_summary():
    bundle = _make_bundle()
    manifest = build_manifest(bundle)
    jobs = manifest["jobs"]
    summary_jobs = [j for j in jobs if j["phase"] == "phase3_summary"]
    assert len(summary_jobs) == 1


def test_probing_payload_has_variant_count():
    bundle = _make_bundle()
    manifest = build_manifest(bundle)
    jobs = manifest["jobs"]
    probing_jobs = [j for j in jobs if j["phase"] == "probing"]
    for j in probing_jobs:
        assert j["payload"]["variant_count"] == 10
        assert "direction_types" in j["payload"]
