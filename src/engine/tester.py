from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.utils.checkpoint import load_checkpoint_file
from src.utils.qualitative import QualitativeSampler


class Tester:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        metric_manager: Any,
        loss_fn: torch.nn.Module | None,
        cfg: Any,
        device: torch.device,
        logger: Any | None = None,
    ) -> None:
        self.model = model.to(device)
        self.test_loader = test_loader
        self.metric_manager = metric_manager
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.class_labels = getattr(self.metric_manager, "class_names", {})

        logging_cfg = cfg.get("logging") or {}
        qualitative_cfg = logging_cfg.get("qualitative") or {}
        self.qualitative_enabled = bool(qualitative_cfg.get("enabled", True))
        self.qualitative_cfg = qualitative_cfg
        self.num_classes = int(getattr(self.metric_manager, "num_classes", len(self.class_labels)))
        self.seed = int((cfg.get("experiment") or {}).get("seed", 0))

        self.metric_names = list(self.metric_manager.names)

    # Forward
    def _forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning (predictions, optional_model_loss)."""
        if getattr(self.model, "is_mask2former", False):
            return self.model(images, masks)

        predictions = self.model(images)

        return predictions, None

    # Batch preparation
    def _prepare_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)

        # Semantic segmentation masks:
        # [B, H, W] -> [B, 1, H, W]
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        if masks.ndim != 4:
            raise ValueError(
                f"Invalid mask shape: {tuple(masks.shape)}. "
                "Expected [B, 1, H, W] for multiclass semantic segmentation."
            )

        if masks.shape[1] != 1:
            raise ValueError(
                f"Invalid mask channel dimension: {tuple(masks.shape)}. "
                "Expected [B, 1, H, W], not one-hot [B, C, H, W]."
            )

        return images, masks

    # AMP
    def _autocast(self):
        train_cfg = self.cfg.get("train") or {}
        precision = str(train_cfg.get("precision", "32"))
        enabled = precision == "16-mixed" and self.device.type == "cuda"
        return torch.amp.autocast(device_type="cuda", enabled=enabled)

    # Checkpoint
    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        preloaded_checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load model weights from a saved checkpoint file."""
        ckpt_path = Path(checkpoint_path)
        checkpoint = (
            preloaded_checkpoint
            if preloaded_checkpoint is not None
            else load_checkpoint_file(ckpt_path)
        )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

        print(f"Loaded checkpoint: {ckpt_path}", flush=True)

        if "epoch" in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}", flush=True)

        best_score = checkpoint.get("best_score")
        if best_score is not None:
            print(f"Best score: {float(best_score):.4f}", flush=True)

        return checkpoint

    # Test
    @torch.no_grad()
    def test(
        self,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate the model on the test set.

        Returns:
            Dictionary containing:
                - metrics
                - per_class
                - qualitative
        """
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()
        self.metric_manager.reset()

        running_loss = 0.0
        num_batches = 0
        total_images = 0
        total_model_time = 0.0
        sampler = (
            QualitativeSampler.from_config(self.qualitative_cfg, self.num_classes, seed=self.seed)
            if self.qualitative_enabled
            else None
        )

        progress = tqdm(
            self.test_loader,
            desc="[Test]",
            leave=False,
        )

        for batch in progress:
            images, masks = self._prepare_batch(batch)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            with self._autocast():
                predictions, loss = self._forward(images, masks)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            total_model_time += elapsed
            total_images += images.shape[0]

            if loss is None:
                if self.loss_fn is None:
                    raise RuntimeError(
                        "Model returned no loss and no loss function is configured."
                    )
                with self._autocast():
                    loss = self.loss_fn(predictions, masks)

            self.metric_manager.update(predictions.detach(), masks.detach())

            loss_value = float(loss.detach().item())
            running_loss += loss_value
            num_batches += 1

            if sampler is not None:
                sampler.update(images, masks, predictions.detach())

            progress.set_postfix(loss=f"{loss_value:.4f}")

        qualitative_batch = sampler.result() if sampler is not None else None

        metrics = self.metric_manager.compute()
        per_class_results = self.metric_manager.compute_per_class()

        test_loss = running_loss / max(num_batches, 1)
        fps = total_images / max(total_model_time, 1e-8)

        metrics = {key: float(value) for key, value in metrics.items()}
        metrics["loss"] = test_loss
        metrics["fps"] = fps

        per_class_results = {
            key: float(value) for key, value in per_class_results.items()
        }

        results = {
            "metrics": metrics,
            "per_class": per_class_results,
            "qualitative": qualitative_batch,
        }

        self._log_results(results)

        return results

    # Logging
    def _resolve_class_labels(self) -> dict[int, str]:
        if self.class_labels:
            return {int(key): str(value) for key, value in self.class_labels.items()}

        data_cfg = self.cfg.get("data") or {}
        configured_labels = data_cfg.get(
            "classes",
            data_cfg.get("class_labels", None),
        )

        if configured_labels is not None:
            if isinstance(configured_labels, dict):
                return {int(key): str(value) for key, value in configured_labels.items()}
            if isinstance(configured_labels, (list, tuple)):
                return {index: str(name) for index, name in enumerate(configured_labels)}

        num_classes = data_cfg.get("num_classes")
        if num_classes is not None:
            return {index: f"class_{index}" for index in range(int(num_classes))}

        return {}

    def _log_results(
        self,
        results: dict[str, Any],
    ) -> None:
        if self.logger is None:
            return

        metrics = results["metrics"]
        per_class = results["per_class"]

        qualitative = results.get("qualitative")
        if self.qualitative_enabled and qualitative is not None:
            class_labels = self._resolve_class_labels()
            self.logger.log_qualitative(
                images=qualitative["image"],
                masks=qualitative["mask"],
                predictions=qualitative["prediction"],
                class_labels=class_labels,
                scenario="test",
                table_name="qualitative/test",
                groups=qualitative.get("group"),
                scores=qualitative.get("score"),
                indices=qualitative.get("index"),
            )
            if qualitative.get("confusion_matrix") is not None:
                self.logger.log_confusion_matrix(
                    qualitative["confusion_matrix"],
                    class_labels=class_labels,
                    key="test/confusion_matrix",
                )

        self.logger.log_test_results(
            {key: float(value) for key, value in metrics.items()},
            {key: float(value) for key, value in per_class.items()},
            commit=True,
        )
