"""Base model backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ModelBackend(ABC):
    """Abstract base for model inference backends."""

    @abstractmethod
    def load_model(self, model_config) -> None:
        """Load model and tokenizer from model_config."""

    @abstractmethod
    def generate_responses(
        self,
        prompts: list[dict[str, str]],
        *,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Generate text responses for a list of {system_prompt, question} dicts."""

    @abstractmethod
    def extract_activations(
        self,
        texts: list[str],
        prompt_lengths: list[int],
        *,
        layers: list[int] | None = None,
        position: str = "response_average",
    ) -> dict[int, np.ndarray]:
        """Extract residual stream activations.

        Returns {layer_idx: array of shape (n_samples, d_model)}.
        """

    @abstractmethod
    def format_chat(
        self,
        system_prompt: str,
        question: str,
        response: str | None = None,
    ) -> str:
        """Format a chat conversation using the model's chat template.

        Returns the full formatted string. If response is given, includes it.
        """

    @abstractmethod
    def get_prompt_token_length(self, system_prompt: str, question: str) -> int:
        """Return the number of tokens in the prompt (before the response)."""

    def unload_model(self) -> None:  # noqa: B027
        """Release model resources. Override in subclasses if needed."""


def create_backend(provider: str, **kwargs: Any) -> ModelBackend:
    if provider == "huggingface":
        from deconfounding_interp.backends.hf_backend import HFBackend
        return HFBackend(**kwargs)
    if provider == "vllm":
        from deconfounding_interp.backends.vllm_backend import VLLMBackend
        return VLLMBackend(**kwargs)
    raise ValueError(f"Unknown backend provider: {provider!r}. Supported: 'huggingface', 'vllm'")
