"""Coherence scoring using the same logprobs mechanism as TraitJudge."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from deconfounding_interp.llm.base import LLMClient
from deconfounding_interp.llm.judge import JudgeResult, get_judge_score_from_logprobs

_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "judge_coherence.txt"


class CoherenceJudge:

    def __init__(self, client: LLMClient, top_logprobs: int = 20):
        self.client = client
        self._top_logprobs = top_logprobs
        self._template = _TEMPLATE_PATH.read_text()

    async def score_response(self, response: str, question: str) -> JudgeResult:
        text = self._template.format(question=question, response=response)
        messages = [{"role": "user", "content": text}]
        result = await self.client.generate_with_logprobs(
            messages, temperature=0.0, max_tokens=1,
            top_logprobs=self._top_logprobs,
        )
        score = get_judge_score_from_logprobs(result.logprobs)
        return JudgeResult(score=score, raw_logprobs=result.logprobs)

    async def score_batch(
        self,
        items: list[dict[str, Any]],
        concurrency: int = 50,
    ) -> list[JudgeResult]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _score_one(item: dict) -> JudgeResult:
            async with semaphore:
                return await self.score_response(item["response"], item["question"])

        return await asyncio.gather(*[_score_one(item) for item in items])
