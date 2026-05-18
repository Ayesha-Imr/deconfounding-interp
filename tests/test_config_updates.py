"""Tests for config updates: new llm and backend fields."""

from deconfounding_interp.config import load_config_bundle


def test_bundle_has_llm_and_backend():
    bundle = load_config_bundle("configs/experiments/main.yaml")
    assert bundle.experiment.backend == "vllm"
    assert bundle.experiment.llm["provider"] == "openai"
    assert bundle.experiment.llm["generation_model"] == "gpt-4.1-mini-2025-04-14"
    assert bundle.experiment.llm["judge_model"] == "gpt-4.1-mini-2025-04-14"


def test_model_backend_is_none_by_default():
    bundle = load_config_bundle("configs/experiments/main.yaml")
    for model in bundle.models.values():
        assert model.backend is None


def test_judge_config_has_logprobs_fields():
    bundle = load_config_bundle("configs/experiments/main.yaml")
    assert bundle.judge["logprobs"] is True
    assert bundle.judge["top_logprobs"] == 20
    assert bundle.judge["min_prob_threshold"] == 0.25
    assert bundle.judge["max_completion_tokens"] == 1
