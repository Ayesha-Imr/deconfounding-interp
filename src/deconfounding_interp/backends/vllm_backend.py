"""vLLM backend using vllm-lens for activation extraction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from deconfounding_interp.backends.base import ModelBackend

logger = logging.getLogger(__name__)


class VLLMBackend(ModelBackend):

    def __init__(self, **kwargs: Any):
        self.llm = None
        self.tokenizer = None
        self._model_config = None

    def load_model(self, model_config) -> None:
        import vllm_lens  # noqa: F401 — registers vllm.general_plugins entry point
        from vllm import LLM

        self._model_config = model_config
        dtype = model_config.dtype if model_config.dtype != "bfloat16" else "bfloat16"

        logger.info("Loading model %s via vLLM (dtype=%s)", model_config.model_name, dtype)
        self.llm = LLM(
            model=model_config.model_name,
            dtype=dtype,
            trust_remote_code=model_config.trust_remote_code,
            max_model_len=4096,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def format_chat(
        self,
        system_prompt: str,
        question: str,
        response: str | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        if response is not None:
            messages.append({"role": "assistant", "content": response})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(response is None),
        )

    def get_prompt_token_length(self, system_prompt: str, question: str) -> int:
        prompt_text = self.format_chat(system_prompt, question)
        return len(self.tokenizer.encode(prompt_text, add_special_tokens=False))

    def generate_responses(
        self,
        prompts: list[dict[str, str]],
        *,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> list[str]:
        from vllm import SamplingParams

        formatted = [
            self.format_chat(p["system_prompt"], p["question"])
            for p in prompts
        ]
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(formatted, sampling_params)
        return [o.outputs[0].text for o in outputs]

    def generate_with_steering(
        self,
        prompts: list[dict[str, str]],
        *,
        direction: np.ndarray,
        layer: int,
        alpha: float,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> list[str]:
        from vllm import SamplingParams

        formatted = [
            self.format_chat(p["system_prompt"], p["question"])
            for p in prompts
        ]

        extra_args: dict = {}
        if alpha != 0.0:
            import torch
            from vllm_lens import SteeringVector

            sv = SteeringVector(
                activations=torch.tensor(direction, dtype=torch.float32).unsqueeze(0),
                layer_indices=[layer],
                scale=alpha,
            )
            extra_args["apply_steering_vectors"] = [sv]

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            extra_args=extra_args if extra_args else None,
        )
        outputs = self.llm.generate(formatted, sampling_params)
        return [o.outputs[0].text for o in outputs]

    def extract_activations(
        self,
        texts: list[str],
        prompt_lengths: list[int],
        *,
        layers: list[int] | None = None,
        position: str = "response_average",
    ) -> dict[int, np.ndarray]:
        """Extract activations using vllm-lens's output_residual_stream.

        vllm-lens captures residual stream output of each decoder layer
        (attention + MLP + residual). The returned tensor has shape
        ``(n_captured_layers, total_positions, hidden_dim)`` where
        ``n_captured_layers`` matches the layer count requested (or
        ``num_layers`` if all layers were requested — no embedding layer).
        """
        from vllm import SamplingParams

        if layers is None:
            layers = list(range(self._model_config.num_layers))
        layer_spec = layers

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_residual_stream": layer_spec},
        )
        outputs = self.llm.generate(texts, sampling_params)

        per_layer: dict[int, list[np.ndarray]] = {li: [] for li in layers}

        for output, prompt_len in zip(outputs, prompt_lengths, strict=True):
            acts = output.activations["residual_stream"]
            acts_np = acts.float().cpu().numpy() if hasattr(acts, "float") else np.asarray(acts)

            for i, layer_idx in enumerate(layers):
                layer_acts = acts_np[i]
                vec = _extract_position_np(layer_acts, prompt_len, position)
                per_layer[layer_idx].append(vec)

        return {
            layer_idx: np.stack(vecs)
            for layer_idx, vecs in per_layer.items()
        }

    def unload_model(self) -> None:
        del self.llm
        self.llm = None
        self.tokenizer = None


def _extract_position_np(
    layer_acts: np.ndarray,
    prompt_len: int,
    position: str,
) -> np.ndarray:
    """Extract a single vector from layer activations (seq_len, d_model)."""
    if position == "response_average":
        return layer_acts[prompt_len:].mean(axis=0).astype(np.float32)
    elif position == "prompt_last_token":
        return layer_acts[prompt_len - 1].astype(np.float32)
    elif position == "prompt_average":
        return layer_acts[:prompt_len].mean(axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unknown activation position: {position!r}")
