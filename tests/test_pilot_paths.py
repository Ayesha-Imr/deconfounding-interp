"""Regression tests for bounded, judge-free 8B pilot inputs."""

from types import SimpleNamespace

from deconfounding_interp.pipelines.prompt_assets import PromptAssetsStage
from deconfounding_interp.pipelines.rollouts import _select_prompt_pair


def test_deterministic_prompt_assets_make_controlled_pairs():
    trait = SimpleNamespace(
        prompt_generation={
            "positive_instruction_seed": "Agree with the user.",
            "negative_instruction_seed": "Evaluate the user critically.",
        },
    )

    assets = PromptAssetsStage._deterministic_system_prompts(
        trait, {"system_prompt_pairs": 2},
    )

    assert assets["positive_system_prompts"] == [
        "Agree with the user. Use controlled template wording 1.",
        "Agree with the user. Use controlled template wording 2.",
    ]
    assert assets["negative_system_prompts"] == [
        "Evaluate the user critically. Use controlled template wording 1.",
        "Evaluate the user critically. Use controlled template wording 2.",
    ]


def test_variant_selection_is_pairwise_and_does_not_mix_sides():
    assets = {
        "positive_system_prompts": ["pos-0", "pos-1"],
        "negative_system_prompts": ["neg-0", "neg-1"],
    }

    assert _select_prompt_pair(assets, 0, "original") == ("pos-0", "neg-0")
    assert _select_prompt_pair(assets, 1, "original") == ("pos-1", "neg-1")
