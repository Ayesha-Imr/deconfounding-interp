"""OpenAI LLM client implementation."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path
from typing import Any

from deconfounding_interp.io import save_llm_audit_csv
from deconfounding_interp.llm.base import LLMClient, LLMResponse, LogprobsResponse


class OpenAIClient(LLMClient):

    def __init__(self, model: str, audit_dir: Path | None = None):
        super().__init__(model=model, audit_dir=audit_dir)
        import openai
        self._client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        resp, latency = await self._call_with_retry(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
        result = LLMResponse(
            content=content,
            model=resp.model,
            usage=usage,
            latency_ms=latency,
            raw=resp.model_dump(),
        )
        self._maybe_audit(messages, result)
        return result

    async def generate_with_logprobs(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1,
        top_logprobs: int = 20,
    ) -> LogprobsResponse:
        resp, latency = await self._call_with_retry(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        content = resp.choices[0].message.content or ""
        raw_logprobs = []
        if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            for token_info in resp.choices[0].logprobs.content:
                raw_logprobs.append({
                    "token": token_info.token,
                    "logprob": token_info.logprob,
                    "top_logprobs": [
                        {"token": t.token, "logprob": t.logprob}
                        for t in (token_info.top_logprobs or [])
                    ],
                })
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
        result = LogprobsResponse(
            content=content,
            model=resp.model,
            usage=usage,
            latency_ms=latency,
            raw=resp.model_dump(),
            logprobs=raw_logprobs,
        )
        self._maybe_audit(messages, result)
        return result

    async def _call_with_retry(
        self, *, messages, max_retries: int = 3, **kwargs,
    ) -> tuple[Any, float]:
        import openai
        for attempt in range(max_retries):
            try:
                t0 = time.monotonic()
                resp = await self._client.chat.completions.create(
                    model=self.model, messages=messages, **kwargs,
                )
                latency = (time.monotonic() - t0) * 1000
                return resp, latency
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError):
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("Unreachable")

    def _maybe_audit(self, messages: list[dict], result: LLMResponse) -> None:
        if self.audit_dir is None:
            return
        prompt_text = messages[-1].get("content", "")[:200]
        prompt_hash = hashlib.sha256(str(messages).encode()).hexdigest()[:12]
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": result.model,
            "prompt_hash": prompt_hash,
            "prompt_preview": prompt_text,
            "response_preview": result.content[:200],
            "tokens_used": (
                result.usage.get("prompt_tokens", 0)
                + result.usage.get("completion_tokens", 0)
            ),
            "latency_ms": f"{result.latency_ms:.0f}",
            "metadata_json": "",
        }
        save_llm_audit_csv(self.audit_dir / "llm_audit.csv", [record])
