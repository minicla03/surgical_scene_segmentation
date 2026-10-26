from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


class Evaluator:
    """
    Standalone evaluator that mirrors the Trainer validation loop.

    Computes loss, Dice, IoU and inference FPS on a test DataLoader
    using the same _prepare_batch / _forward / _compute_metrics logic
    as the Trainer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        scaler: Any,
        cfg: Any,
        device: torch.device,
        logger: Any | None = None,
    ) -> None:

        self.model = model.to(device)
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.cfg = cfg
        self.device = device
        self.logger = logger

        self.num_classes = int(cfg.data.num_classes)

    # =========================================================
    # AMP
    # =========================================================

    def _autocast(self):

        return torch.amp.autocast(
            device_type=self.device.type,
            enabled=(
                self.scaler is not None
                and self.device.type == "cuda"
            ),
        )

    # =========================================================
    # Batch preparation
    # =========================================================

    def _prepare_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x = batch["image"].to(
            self.device,
            non_blocking=True,
        )

        y = batch["mask"].to(
            self.device,
            non_blocking=True,
        )

        # [B, 1, H, W] -> [B, H, W]
        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]

        y = y.long()

        return x, y

    # =========================================================
    # Forward
    # =========================================================

    def _forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        output = self.model(x)

        if isinstance(output, (list, tuple)):
            losses = [
                self.loss_fn(logits, y)
                for logits in output
            ]
            loss = sum(losses) / len(losses)
            logits = output[0]
        else:
            logits = output
            loss = self.loss_fn(logits, y)

        return logits, loss

    # =========================================================
    # Per-class Dice / IoU
    # =========================================================

    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:

        predictions = predictions.long()
        labels = labels.long()

        dice_values: list[float] = []
        iou_values: list[float] = []

        for class_id in range(self.num_classes):

            pred_class = predictions == class_id
            label_class = labels == class_id

            intersection = (
                pred_class & label_class
            ).sum(dim=(1, 2)).float()

            pred_area = pred_class.sum(dim=(1, 2)).float()
            label_area = label_class.sum(dim=(1, 2)).float()

            union = pred_area + label_area - intersection

            dice = (2.0 * intersection + 1e-8) / (
                pred_area + label_area + 1e-8
            )

            iou = (intersection + 1e-8) / (union + 1e-8)

            valid = (pred_area + label_area) > 0

            if valid.any():
                dice_values.append(dice[valid].mean().item())
                iou_values.append(iou[valid].mean().item())

        return {
            "dice": (
                float(np.mean(dice_values))
                if dice_values
                else 0.0
            ),
            "iou": (
                float(np.mean(iou_values))
                if iou_values
                else 0.0
            ),
        }

    # =========================================================
    # Evaluate
    # =========================================================

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        """
        Run the full test loop and return aggregated metrics.

        Returns
        -------
        dict with keys: loss, dice, iou, fps
        """

        print("\n" + "=" * 60)
        print("TEST EVALUATION")
        print("=" * 60)

        self.model.eval()

        test_losses: list[float] = []
        dice_values: list[float] = []
        iou_values: list[float] = []

        total_images = 0
        total_model_time = 0.0

        progress = tqdm(
            self.test_loader,
            desc="Test",
        )

        for batch in progress:

            x, y = self._prepare_batch(batch)

            # -------------------------------------------------
            # Inference timing
            # -------------------------------------------------

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            with self._autocast():
                logits, loss = self._forward(x, y)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            total_model_time += (
                time.perf_counter() - start_time
            )

            test_losses.append(loss.detach().item())

            # -------------------------------------------------
            # Predictions & metrics
            # -------------------------------------------------

            predictions = torch.argmax(logits, dim=1)

            batch_metrics = self._compute_metrics(
                predictions,
                y,
            )

            dice_values.append(batch_metrics["dice"])
            iou_values.append(batch_metrics["iou"])

            total_images += x.shape[0]

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{batch_metrics['dice']:.4f}",
            )

        # =====================================================
        # Aggregate
        # =====================================================

        fps = total_images / max(total_model_time, 1e-8)

        results = {
            "loss": float(np.mean(test_losses)),
            "dice": (
                float(np.mean(dice_values))
                if dice_values
                else float("nan")
            ),
            "iou": (
                float(np.mean(iou_values))
                if iou_values
                else float("nan")
            ),
            "fps": fps,
        }

        # =====================================================
        # W&B logging
        # =====================================================

        if self.logger is not None:
            self.logger.log_test_metrics(results)

        return results