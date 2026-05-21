"""LLM client abstraction and implementations."""

from deconfounding_interp.llm.base import (
    LLMClient,
    LLMResponse,
    LogprobsResponse,
    create_client,
)
from deconfounding_interp.llm.coherence_judge import CoherenceJudge
from deconfounding_interp.llm.judge import TraitJudge, get_judge_score_from_logprobs

__all__ = [
    "CoherenceJudge",
    "LLMClient",
    "LLMResponse",
    "LogprobsResponse",
    "TraitJudge",
    "create_client",
    "get_judge_score_from_logprobs",
]
