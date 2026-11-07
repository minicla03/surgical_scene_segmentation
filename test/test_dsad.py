"""
Smoke tests and sanity checks for Trainer pipeline, batch formatting, and metrics.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.engine.trainer import Trainer
from src.metrics.metric_manager import MetricManager


class DummySegmentationDataset(Dataset):
    """Synthetic dataset producing well-formed segmentation pairs."""

    def __init__(self, num_samples: int = 4, num_classes: int = 8, size=(64, 64)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        h, w = self.size
        image = torch.randn(3, h, w, dtype=torch.float32)
        mask = torch.randint(0, self.num_classes, (1, h, w), dtype=torch.long)
        return {"image": image, "mask": mask}


class DummySegmentationModel(torch.nn.Module):
    """Minimal CNN outputting logits [B, C, H, W] for testing trainer mechanics."""

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x, labels=None):
        logits = self.conv(x)
        if labels is not None:
            # Mask [B, 1, H, W] -> [B, H, W]
            target = labels.squeeze(1) if labels.ndim == 4 else labels
            loss = torch.nn.functional.cross_entropy(logits, target)
            return type("ModelOutput", (), {"logits": logits, "loss": loss})()
        return logits


@pytest.fixture
def minimal_trainer(device):
    """Instantiate a minimal but fully functional Trainer instance."""
    num_classes = 8
    model = DummySegmentationModel(num_classes=num_classes)
    train_ds = DummySegmentationDataset(num_samples=4, num_classes=num_classes)
    val_ds = DummySegmentationDataset(num_samples=2, num_classes=num_classes)

    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=False)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    metric_manager = MetricManager(
        metrics={},
        num_classes=num_classes,
        class_names={i: f"cls_{i}" for i in range(num_classes)},
    )

    cfg = OmegaConf.create({
        "data": {"name": "dsad", "num_classes": num_classes},
        "model": {"name": "dummy_model"},
        "output_dir": "/tmp/test_trainer_outputs",
        "train": {
            "max_epochs": 1,
            "accumulation_steps": 1,
            "eval_freq": 1,
            "early_stopping": {"enabled": False, "patience": 5, "monitor": "loss", "mode": "min"},
        },
        "logging": {"qualitative": {"enabled": False}},
    })

    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        scaler=scaler,
        metric_manager=metric_manager,
        cfg=cfg,
        device=device,
        logger=None,
    )


class TestTrainerSmoke:
    def test_batch_structure(self, minimal_trainer):
        """Verify train loader outputs correct 4D shapes and dtypes."""
        batch = next(iter(minimal_trainer.train_loader))
        images, masks = minimal_trainer._prepare_batch(batch)

        assert images.ndim == 4, f"Expected [B, C, H, W], got {images.shape}"
        assert masks.ndim == 4, f"Expected [B, 1, H, W], got {masks.shape}"
        assert masks.shape[1] == 1, "Mask channel dimension must be 1"
        assert masks.dtype == torch.long, f"Mask dtype should be torch.long, got {masks.dtype}"

        expected_classes = int(minimal_trainer.cfg.data.num_classes)
        assert masks.min() >= 0, "Negative class index found in mask"
        assert masks.max() < expected_classes, f"Class index {masks.max()} exceeds num_classes={expected_classes}"

    def test_train_step(self, minimal_trainer):
        """Verify forward pass, loss calculation and backward step on trainer."""
        minimal_trainer.model.train()
        minimal_trainer.optimizer.zero_grad(set_to_none=True)

        batch = next(iter(minimal_trainer.train_loader))
        images, masks = minimal_trainer._prepare_batch(batch)

        with minimal_trainer._autocast():
            logits, loss = minimal_trainer._forward(images, masks)
            if loss is None:
                loss = minimal_trainer.loss_fn(logits, masks.squeeze(1))

        assert logits.ndim == 4
        expected_classes = int(minimal_trainer.cfg.data.num_classes)
        assert logits.shape[1] == expected_classes
        assert loss is not None
        assert torch.isfinite(loss)

        loss.backward()
        minimal_trainer.optimizer.step()
        minimal_trainer.optimizer.zero_grad(set_to_none=True)

    def test_validation_step(self, minimal_trainer):
        """Verify validation forward and metric update without gradient tracking."""
        minimal_trainer.model.eval()
        minimal_trainer.metric_manager.reset()

        batch = next(iter(minimal_trainer.val_loader))
        images, masks = minimal_trainer._prepare_batch(batch)

        with torch.no_grad():
            with minimal_trainer._autocast():
                logits, loss = minimal_trainer._forward(images, masks)
                if loss is None:
                    loss = minimal_trainer.loss_fn(logits, masks.squeeze(1))

        assert logits.ndim == 4
        assert loss is not None
        assert torch.isfinite(loss)

        minimal_trainer.metric_manager.update(logits, masks)
        metrics = minimal_trainer.metric_manager.compute()
        assert isinstance(metrics, dict)
