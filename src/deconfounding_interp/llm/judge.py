"""Logprobs-based trait scoring and TraitJudge."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

from deconfounding_interp.llm.base import LLMClient


@dataclass
class JudgeResult:
    score: float | None
    raw_logprobs: list[dict[str, Any]]


def get_judge_score_from_logprobs(
    logprobs: list[dict[str, Any]],
    scale_min: int = 0,
    scale_max: int = 100,
    min_prob_threshold: float = 0.25,
) -> float | None:
    """Compute probability-weighted score from top logprobs of a single-token judge response.

    Extracts all tokens parseable as integers in [scale_min, scale_max],
    converts logprobs to probabilities, and returns their weighted average.
    Returns None if total probability mass on valid tokens < min_prob_threshold.
    """
    if not logprobs or not logprobs[0].get("top_logprobs"):
        return None

    weighted_sum = 0.0
    total_prob = 0.0

    for entry in logprobs[0]["top_logprobs"]:
        token = entry["token"].strip()
        try:
            value = int(token)
        except (ValueError, TypeError):
            continue
        if value < scale_min or value > scale_max:
            continue
        prob = math.exp(entry["logprob"])
        weighted_sum += value * prob
        total_prob += prob

    if total_prob < min_prob_threshold:
        return None
    return weighted_sum / total_prob


class TraitJudge:
    """Async trait-expression judge using logprobs scoring."""

    def __init__(
        self,
        client: LLMClient,
        judge_config: dict[str, Any],
    ):
        self.client = client
        self.config = judge_config
        self._template = judge_config.get("prompt_template", "")
        self._scale_min = judge_config.get("scale", {}).get("min", 0)
        self._scale_max = judge_config.get("scale", {}).get("max", 100)
        self._min_prob = judge_config.get("min_prob_threshold", 0.25)
        self._top_logprobs = judge_config.get("top_logprobs", 20)

    def _format_prompt(
        self,
        response: str,
        question: str,
        trait,
    ) -> list[dict[str, str]]:
        text = self._template.format(
            trait_display_name=trait.display_name,
            positive_definition=trait.positive_definition,
            negative_definition=trait.negative_definition,
            high_anchor=trait.judge.get("high_anchor", ""),
            low_anchor=trait.judge.get("low_anchor", ""),
            question=question,
            response=response,
        )
        return [{"role": "user", "content": text}]

    async def score_response(
        self,
        response: str,
        question: str,
        trait,
    ) -> JudgeResult:
        messages = self._format_prompt(response, question, trait)
        result = await self.client.generate_with_logprobs(
            messages,
            temperature=0.0,
            max_tokens=1,
            top_logprobs=self._top_logprobs,
        )
        score = get_judge_score_from_logprobs(
            result.logprobs,
            scale_min=self._scale_min,
            scale_max=self._scale_max,
            min_prob_threshold=self._min_prob,
        )
        return JudgeResult(score=score, raw_logprobs=result.logprobs)

    async def score_batch(
        self,
        items: list[dict[str, Any]],
        trait,
        concurrency: int = 50,
    ) -> list[JudgeResult]:
        """Score a batch of responses with bounded concurrency.

        Each item in ``items`` must have keys 'response' and 'question'.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _score_one(item: dict) -> JudgeResult:
            async with semaphore:
                return await self.score_response(
                    item["response"], item["question"], trait,
                )

        return await asyncio.gather(*[_score_one(item) for item in items])
