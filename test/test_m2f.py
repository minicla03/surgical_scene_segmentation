"""
Integration tests for Mask2FormerWrapper model.
"""
from __future__ import annotations

import pytest
import torch

from src.models.mask2former_wrapper import Mask2FormerWrapper


@pytest.fixture(scope="module")
def checkpoint_name():
    return "facebook/mask2former-swin-base-ade-semantic"


@pytest.fixture(scope="module")
def num_classes():
    return 7


@pytest.fixture(scope="module")
def m2f_model(checkpoint_name, num_classes, device):
    """Instantiate Mask2FormerWrapper model on available device."""
    try:
        model = Mask2FormerWrapper(
            checkpoint=checkpoint_name,
            num_labels=num_classes,
        ).to(device)
        return model
    except Exception as exc:
        pytest.skip(f"Failed to load Mask2Former weights ({checkpoint_name}): {exc}")


@pytest.fixture
def synthetic_batch(num_classes, device):
    batch_size = 2
    height, width = 256, 256
    images = torch.randn(batch_size, 3, height, width, device=device)
    masks = torch.randint(0, num_classes, (batch_size, height, width), device=device)
    return images, masks


@pytest.mark.slow
class TestMask2FormerWrapper:
    def test_train_forward_and_backward(self, m2f_model, synthetic_batch):
        """Verify training step computes valid finite loss and gradients."""
        images, masks = synthetic_batch
        m2f_model.train()
        m2f_model.zero_grad(set_to_none=True)

        outputs = m2f_model(pixel_values=images, labels=masks)

        assert outputs.loss is not None, "Training loss is None"
        assert torch.isfinite(outputs.loss), f"Loss is not finite: {outputs.loss}"

        outputs.loss.backward()

        grads = [
            p.grad for p in m2f_model.parameters() if p.requires_grad and p.grad is not None
        ]
        assert len(grads) > 0, "No gradients were generated during backward pass"
        assert all(torch.isfinite(g).all() for g in grads), "Gradients contain NaN/Inf"

    def test_validation_forward(self, m2f_model, synthetic_batch, num_classes):
        """Verify evaluation mode outputs correct logit shapes without NaN/Inf."""
        images, _ = synthetic_batch
        batch_size, _, height, width = images.shape

        m2f_model.eval()
        with torch.no_grad():
            preds = m2f_model(pixel_values=images)

        assert isinstance(preds, torch.Tensor), f"Expected Tensor output, got {type(preds)}"
        expected_shape = (batch_size, num_classes, height, width)
        assert preds.shape == expected_shape, f"Shape mismatch: {preds.shape} vs {expected_shape}"
        assert torch.isfinite(preds).all(), "Validation outputs contain NaN or Inf"

    @pytest.mark.gpu
    def test_amp_training_step(self, m2f_model, synthetic_batch, require_cuda):
        """Verify forward and backward compatibility with automatic mixed precision (AMP)."""
        images, masks = synthetic_batch
        m2f_model.train()
        m2f_model.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=True):
            outputs = m2f_model(pixel_values=images, labels=masks)

        loss = outputs.loss
        assert loss is not None and torch.isfinite(loss), f"Invalid AMP loss: {loss}"

        loss.backward()

        grads = [
            p.grad for p in m2f_model.parameters() if p.requires_grad and p.grad is not None
        ]
        assert len(grads) > 0
