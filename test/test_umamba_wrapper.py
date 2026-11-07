"""
Unit and integration tests for UMambaWrapper adapter.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.models.umamba_adapter import UMambaWrapper


@pytest.fixture
def umamba_kwargs():
    """Default valid configuration dictionary for UMambaWrapper."""
    return {
        "input_channels": 3,
        "num_classes": 13,
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 320, 320],
        "kernel_sizes": [[3, 3]] * 6,
        "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
        "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
        "deep_supervision": False,
    }


class TestUMambaWrapper:
    def test_instantiation_with_python_types(self, umamba_kwargs):
        """Verify model instantiation and forward pass with standard Python types."""
        model = UMambaWrapper(**umamba_kwargs)
        x = torch.randn(1, 3, 240, 427)

        model.eval()
        with torch.no_grad():
            output = model(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 13, 240, 427)
        assert torch.isfinite(output).all()

    def test_instantiation_with_omegaconf(self, umamba_kwargs):
        """Verify model handles OmegaConf ListConfig and DictConfig seamlessly."""
        cfg = OmegaConf.create(umamba_kwargs)
        model = UMambaWrapper(**cfg)
        x = torch.randn(1, 3, 240, 427)

        model.eval()
        with torch.no_grad():
            output = model(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 13, 240, 427)
        assert torch.isfinite(output).all()

    def test_forward_backward_gradient_flow(self, umamba_kwargs):
        """Verify loss computation and that gradients flow into trainable parameters."""
        model = UMambaWrapper(**umamba_kwargs)
        model.train()

        x = torch.randn(1, 3, 240, 427)
        target = torch.randint(low=0, high=13, size=(1, 240, 427))

        output = model(x)
        assert output.shape == (1, 13, 240, 427)

        loss = torch.nn.functional.cross_entropy(output, target)
        assert torch.isfinite(loss)

        loss.backward()

        trainable_grads = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        assert len(trainable_grads) > 0, "No gradients were computed for trainable parameters"
        assert all(torch.isfinite(g).all() for g in trainable_grads), "Gradients contain NaN or Inf"
