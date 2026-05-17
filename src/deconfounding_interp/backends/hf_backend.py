"""HuggingFace Transformers backend using output_hidden_states for activation extraction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from deconfounding_interp.backends.base import ModelBackend

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class HFBackend(ModelBackend):

    def __init__(self, **kwargs: Any):
        self.model = None
        self.tokenizer = None
        self._model_config = None

    def load_model(self, model_config) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_config = model_config
        dtype = _DTYPE_MAP.get(model_config.dtype, torch.bfloat16)

        logger.info("Loading model %s (dtype=%s)", model_config.model_name, model_config.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer_name,
            trust_remote_code=model_config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name,
            torch_dtype=dtype,
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        results = []
        for p in prompts:
            text = self.format_chat(p["system_prompt"], p["question"])
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(temperature > 0),
                )
            response_ids = output_ids[0, prompt_len:]
            results.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))
        return results

    def extract_activations(
        self,
        texts: list[str],
        prompt_lengths: list[int],
        *,
        layers: list[int] | None = None,
        position: str = "response_average",
    ) -> dict[int, np.ndarray]:
        if layers is None:
            layers = list(range(self._model_config.num_layers + 1))

        per_layer: dict[int, list[np.ndarray]] = {li: [] for li in layers}

        for text, prompt_len in zip(texts, prompt_lengths, strict=True):
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            for layer_idx in layers:
                hs = outputs.hidden_states[layer_idx]  # (1, seq_len, d_model)
                vec = _extract_position(hs, prompt_len, position)
                per_layer[layer_idx].append(vec)

        return {
            layer_idx: np.stack(vecs)
            for layer_idx, vecs in per_layer.items()
        }

    def unload_model(self) -> None:
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _extract_position(
    hidden_states: torch.Tensor,
    prompt_len: int,
    position: str,
) -> np.ndarray:
    """Extract a single vector from hidden states based on position strategy."""
    if position == "response_average":
        vec = hidden_states[:, prompt_len:, :].mean(dim=1)
    elif position == "prompt_last_token":
        vec = hidden_states[:, prompt_len - 1:prompt_len, :].mean(dim=1)
    elif position == "prompt_average":
        vec = hidden_states[:, :prompt_len, :].mean(dim=1)
    else:
        raise ValueError(f"Unknown activation position: {position!r}")
    return vec.squeeze(0).float().cpu().numpy()
