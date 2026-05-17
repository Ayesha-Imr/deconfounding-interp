"""Model inference backends (HuggingFace Transformers, vLLM)."""

from deconfounding_interp.backends.base import ModelBackend, create_backend

__all__ = ["ModelBackend", "create_backend"]
