"""Tests for backend steering method — mock-based since no real GPU in tests."""

from unittest.mock import MagicMock

import numpy as np

from deconfounding_interp.backends.hf_backend import HFBackend


class TestHFBackendSteering:
    def test_alpha_zero_delegates_to_generate(self):
        backend = HFBackend()
        backend.generate_responses = MagicMock(return_value=["response1"])

        prompts = [{"system_prompt": "", "question": "hello"}]
        direction = np.ones(10)
        result = backend.generate_with_steering(
            prompts, direction=direction, layer=5, alpha=0.0,
        )

        backend.generate_responses.assert_called_once()
        assert result == ["response1"]

    def test_nonzero_alpha_uses_hook(self):
        backend = HFBackend()
        backend.generate_responses = MagicMock(return_value=["steered"])

        # Mock model layers
        mock_layer = MagicMock()
        mock_handle = MagicMock()
        mock_layer.register_forward_hook.return_value = mock_handle

        import torch
        mock_model = MagicMock()
        mock_model.model.layers.__getitem__ = MagicMock(return_value=mock_layer)
        mock_model.dtype = torch.float32
        mock_model.device = torch.device("cpu")
        backend.model = mock_model

        prompts = [{"system_prompt": "", "question": "hello"}]
        direction = np.ones(10)
        result = backend.generate_with_steering(
            prompts, direction=direction, layer=5, alpha=1.5,
        )

        mock_layer.register_forward_hook.assert_called_once()
        mock_handle.remove.assert_called_once()
        assert result == ["steered"]
