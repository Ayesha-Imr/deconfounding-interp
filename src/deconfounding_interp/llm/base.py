"""Base LLM client interface and response types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int]
    latency_ms: float
    raw: dict[str, Any] = field(repr=False)


@dataclass
class LogprobsResponse(LLMResponse):
    logprobs: list[dict[str, Any]] = field(default_factory=list)


class LLMClient(ABC):
    """Base class for LLM API clients.

    All methods are async for consistency with the async judging pipeline.
    Subclasses may set ``audit_dir`` to auto-log every call to CSV.
    """

    def __init__(self, model: str, audit_dir: Path | None = None):
        self.model = model
        self.audit_dir = audit_dir

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    @abstractmethod
    async def generate_with_logprobs(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1,
        top_logprobs: int = 20,
    ) -> LogprobsResponse: ...


def create_client(
    provider: str,
    model: str,
    *,
    audit_dir: Path | None = None,
) -> LLMClient:
    if provider == "openai":
        from deconfounding_interp.llm.openai_client import OpenAIClient
        return OpenAIClient(model=model, audit_dir=audit_dir)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Supported: 'openai'")
