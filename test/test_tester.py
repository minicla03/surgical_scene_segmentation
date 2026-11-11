"""
Unit and integration tests for Tester evaluation pipeline.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.engine.tester import Tester as SegmentationTester
from src.metrics.metric_manager import MetricManager
from src.utils.checkpoint import atomic_torch_save
from src.utils.wandb_logging import WandbLogger


class SyntheticTestDataset(Dataset):
    """Deterministic synthetic dataset supporting 3D [H, W] or 4D [1, H, W] masks."""

    def __init__(
        self,
        num_samples: int = 4,
        num_classes: int = 4,
        size: tuple[int, int] = (16, 16),
        squeeze_mask: bool = False,
        seed: int = 7,
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        h, w = size
        self.images = torch.randn(num_samples, 3, h, w, generator=gen, dtype=torch.float32)
        masks = torch.randint(0, num_classes, (num_samples, 1, h, w), generator=gen, dtype=torch.long)
        self.masks = masks.squeeze(1) if squeeze_mask else masks

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[idx], "mask": self.masks[idx]}


class CountingSegNet(torch.nn.Module):
    """Minimal segmentation CNN counting forward calls."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.conv(x)


class FakeMask2FormerWrapper(torch.nn.Module):
    """Mimics Mask2FormerWrapper contract: (images, masks=None) -> (scores, loss)."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.is_mask2former = True
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)
        self.forward_calls = 0

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.forward_calls += 1
        logits = self.conv(images)
        probs = torch.softmax(logits, dim=1)
        loss = None
        if masks is not None:
            target = masks.squeeze(1) if masks.ndim == 4 else masks
            loss = torch.nn.functional.cross_entropy(logits, target)
        return probs, loss


def _make_cfg(
    num_classes: int = 4,
    qualitative_enabled: bool = True,
    max_samples: int = 2,
    precision: str = "32",
):
    return OmegaConf.create({
        "data": {"name": "sarrarp50", "num_classes": num_classes},
        "model": {"name": "U-Net"},
        "train": {"precision": precision},
        "logging": {
            "qualitative": {
                "enabled": qualitative_enabled,
                "max_samples": max_samples,
            }
        },
    })


def _make_metric_manager(num_classes: int = 4) -> MetricManager:
    return MetricManager(
        metrics={"dice": DiceMetric(include_background=True, reduction="mean")},
        num_classes=num_classes,
        dataset_name="sarrarp50",
        class_names={i: f"cls_{i}" for i in range(num_classes)},
    )


class TestTester:
    def test_standard_model_evaluation_and_3d_masks(self):
        num_classes = 4
        ds = SyntheticTestDataset(num_samples=4, num_classes=num_classes, squeeze_mask=True)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        model = CountingSegNet(num_classes=num_classes)
        loss_fn = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
        cfg = _make_cfg(num_classes=num_classes, qualitative_enabled=True, max_samples=2)

        tester = SegmentationTester(
            model=model,
            test_loader=loader,
            metric_manager=_make_metric_manager(num_classes),
            loss_fn=loss_fn,
            cfg=cfg,
            device=torch.device("cpu"),
            logger=None,
        )

        results = tester.test()

        assert model.forward_calls == 2
        assert set(results.keys()) == {"metrics", "per_class", "qualitative"}
        assert "dice" in results["metrics"]
        assert "loss" in results["metrics"]
        assert "fps" in results["metrics"]
        assert results["metrics"]["loss"] >= 0.0
        assert results["metrics"]["fps"] > 0.0
        assert len(results["per_class"]) == num_classes

        qual = results["qualitative"]
        assert qual is not None
        assert qual["image"].shape == (2, 3, 16, 16)
        assert qual["mask"].shape == (2, 1, 16, 16)
        assert qual["prediction"].shape == (2, 16, 16)
        assert qual["prediction"].dtype == torch.long

    def test_mask2former_forward_and_native_loss(self):
        num_classes = 4
        ds = SyntheticTestDataset(num_samples=4, num_classes=num_classes)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        model = FakeMask2FormerWrapper(num_classes=num_classes)
        cfg = _make_cfg(num_classes=num_classes, qualitative_enabled=True, max_samples=2)

        tester = SegmentationTester(
            model=model,
            test_loader=loader,
            metric_manager=_make_metric_manager(num_classes),
            loss_fn=None,
            cfg=cfg,
            device=torch.device("cpu"),
            logger=None,
        )

        results = tester.test()

        assert model.forward_calls == 2
        assert results["metrics"]["loss"] > 0.0
        assert "dice" in results["metrics"]
        assert results["qualitative"] is not None
        assert results["qualitative"]["prediction"].shape == (2, 16, 16)

        preds, no_mask_loss = tester._forward(torch.randn(1, 3, 16, 16))
        assert preds.shape == (1, num_classes, 16, 16)
        assert no_mask_loss is None

    def test_invalid_mask_shapes_and_missing_loss_raise(self):
        num_classes = 4
        model = CountingSegNet(num_classes=num_classes)
        cfg = _make_cfg(num_classes=num_classes, qualitative_enabled=False)

        ds = SyntheticTestDataset(num_samples=2, num_classes=num_classes)
        loader = DataLoader(ds, batch_size=2, shuffle=False)

        tester_no_loss = SegmentationTester(
            model=model,
            test_loader=loader,
            metric_manager=_make_metric_manager(num_classes),
            loss_fn=None,
            cfg=cfg,
            device=torch.device("cpu"),
        )

        with pytest.raises(RuntimeError, match="no loss function is configured"):
            tester_no_loss.test()

        bad_batch_onehot = {
            "image": torch.randn(2, 3, 16, 16),
            "mask": torch.zeros(2, num_classes, 16, 16, dtype=torch.long),
        }
        with pytest.raises(ValueError, match="Expected \\[B, 1, H, W\\]"):
            tester_no_loss._prepare_batch(bad_batch_onehot)

        bad_batch_2d = {
            "image": torch.randn(2, 3, 16, 16),
            "mask": torch.zeros(16, 16, dtype=torch.long),
        }
        with pytest.raises(ValueError, match="Invalid mask shape"):
            tester_no_loss._prepare_batch(bad_batch_2d)

    def test_fallback_loss_runs_inside_autocast(self, monkeypatch):
        num_classes = 4
        ds = SyntheticTestDataset(num_samples=2, num_classes=num_classes)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        model = CountingSegNet(num_classes=num_classes)
        cfg = _make_cfg(num_classes=num_classes, qualitative_enabled=False)

        state = {"in_autocast": False, "loss_called_in_autocast": False}

        class TrackingLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ce = torch.nn.CrossEntropyLoss()

            def forward(self, preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
                state["loss_called_in_autocast"] = state["in_autocast"]
                return self.ce(preds, masks.squeeze(1))

        tester = SegmentationTester(
            model=model,
            test_loader=loader,
            metric_manager=_make_metric_manager(num_classes),
            loss_fn=TrackingLoss(),
            cfg=cfg,
            device=torch.device("cpu"),
        )

        @contextmanager
        def fake_autocast():
            state["in_autocast"] = True
            try:
                yield
            finally:
                state["in_autocast"] = False

        monkeypatch.setattr(tester, "_autocast", fake_autocast)
        tester.test()

        assert state["loss_called_in_autocast"] is True

    def test_load_checkpoint_with_omegaconf_and_raw_state_dict(self, tmp_path: Path):
        num_classes = 4
        ds = SyntheticTestDataset(num_samples=2, num_classes=num_classes)
        loader = DataLoader(ds, batch_size=2, shuffle=False)

        src_model = CountingSegNet(num_classes=num_classes)
        with torch.no_grad():
            src_model.conv.weight.fill_(0.25)
            src_model.conv.bias.fill_(0.5)

        cfg = _make_cfg(num_classes=num_classes, qualitative_enabled=False)
        full_ckpt_path = tmp_path / "best_sarrarp50_U-Net_lr_1e-4.pt"
        atomic_torch_save(
            {
                "epoch": 12,
                "model_state_dict": src_model.state_dict(),
                "best_score": 0.8123,
                "config": cfg,
            },
            full_ckpt_path,
        )

        dst_model = CountingSegNet(num_classes=num_classes)
        tester = SegmentationTester(
            model=dst_model,
            test_loader=loader,
            metric_manager=_make_metric_manager(num_classes),
            loss_fn=DiceCELoss(include_background=True, to_onehot_y=True, softmax=True),
            cfg=cfg,
            device=torch.device("cpu"),
        )

        tester.test(checkpoint_path=full_ckpt_path)

        for key, tensor in src_model.state_dict().items():
            assert torch.allclose(dst_model.state_dict()[key], tensor)

        # Also verify raw state_dict checkpoint loading
        raw_ckpt_path = tmp_path / "raw_weights.pt"
        with torch.no_grad():
            src_model.conv.weight.fill_(-0.125)
        atomic_torch_save(src_model.state_dict(), raw_ckpt_path)

        tester.load_checkpoint(raw_ckpt_path)
        assert torch.allclose(dst_model.state_dict()["conv.weight"], src_model.state_dict()["conv.weight"])

    def test_wandb_logging_does_not_rerun_forward_or_overwrite_val_table(self, monkeypatch):
        num_classes = 4
        ds = SyntheticTestDataset(num_samples=4, num_classes=num_classes)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        model = CountingSegNet(num_classes=num_classes)
        cfg = _make_cfg(num_classes=num_classes, qualitative_enabled=True, max_samples=2)

        logged_calls: list[tuple[dict, dict]] = []

        class FakeTable:
            def __init__(self, columns):
                self.columns = columns
                self.rows = []

            def add_data(self, *row):
                self.rows.append(row)

        class FakeImage:
            def __init__(self, image, masks=None):
                self.image = image
                self.masks = masks

        class FakeWandb:
            def __init__(self):
                self.run = SimpleNamespace(id="run_test_1", resumed=False, summary={})

            def Table(self, columns):
                return FakeTable(columns)

            def Image(self, image, masks=None):
                return FakeImage(image, masks=masks)

            def log(self, data, **kwargs):
                logged_calls.append((data, kwargs))

            def finish(self):
                self.run = None

        fake_wandb = FakeWandb()
        import src.utils.wandb_logging as wl_mod

        monkeypatch.setattr(wl_mod, "wandb", fake_wandb)

        logger = WandbLogger(enabled=True)

        # Simulate final epoch validation logging from Trainer (leaves qualitative/segmentation at step 2, commit=False)
        logger.log_epoch(
            epoch=1,
            train_metrics={"loss": 0.5},
            val_metrics={"loss": 0.4},
            qualitative_batch={
                "image": torch.rand(2, 3, 16, 16),
                "mask": torch.zeros(2, 1, 16, 16, dtype=torch.long),
                "prediction": torch.zeros(2, 16, 16, dtype=torch.long),
            },
            class_labels={i: f"cls_{i}" for i in range(num_classes)},
        )

        tester = SegmentationTester(
            model=model,
            test_loader=loader,
            metric_manager=_make_metric_manager(num_classes),
            loss_fn=DiceCELoss(include_background=True, to_onehot_y=True, softmax=True),
            cfg=cfg,
            device=torch.device("cpu"),
            logger=logger,
        )

        results = tester.test()
        logger.log_final_results({"best_epoch": 1, "best_score": 0.75, "best_metrics": {"dice": 0.75}}, results)

        # Forward only ran once per batch (2 batches), NOT re-run for qualitative logging
        assert model.forward_calls == 2

        # Validation table logged on qualitative/segmentation, test table logged on qualitative/test
        val_table_log = logged_calls[1]
        test_table_log = logged_calls[2]
        test_metrics_log = logged_calls[3]

        assert "qualitative/segmentation" in val_table_log[0]
        assert "qualitative/test" in test_table_log[0]
        assert test_table_log[0]["qualitative/test"].rows[0][1] == "test"
        assert len(test_table_log[0]["qualitative/test"].rows) == 2

        assert "test/loss" in test_metrics_log[0]
        assert "test/fps" in test_metrics_log[0]
        assert "test/dice" in test_metrics_log[0]
        assert "test/dice_class_cls_0" in test_metrics_log[0]
        assert test_metrics_log[1]["commit"] is True

        assert fake_wandb.run.summary["test/loss"] == pytest.approx(results["metrics"]["loss"])
        assert fake_wandb.run.summary["test/dice"] == pytest.approx(results["metrics"]["dice"])

    def test_empty_loader_and_mixed_precision_on_cpu(self):
        import math

        num_classes = 4
        empty_ds = SyntheticTestDataset(num_samples=0, num_classes=num_classes)
        loader = DataLoader(empty_ds, batch_size=2, shuffle=False)
        model = CountingSegNet(num_classes=num_classes)
        cfg = _make_cfg(num_classes=num_classes, qualitative_enabled=True, precision="16-mixed")

        tester = SegmentationTester(
            model=model,
            test_loader=loader,
            metric_manager=_make_metric_manager(num_classes),
            loss_fn=DiceCELoss(include_background=True, to_onehot_y=True, softmax=True),
            cfg=cfg,
            device=torch.device("cpu"),
        )

        results = tester.test()
        assert results["metrics"]["loss"] == 0.0
        assert results["metrics"]["fps"] == 0.0
        assert math.isnan(results["metrics"]["dice"])
        assert results["qualitative"] is None

