from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.utils.metrics import MetricComputer


class Evaluator:
    """
    Standalone test-set evaluator.

    Mirrors the Trainer validation loop but is driven entirely by
    ``cfg.evaluation.metrics``.  The same MetricComputer used by the
    Trainer is instantiated here, so training and test metrics are
    always computed identically.

    Always computed (hardcoded):
        loss  — cross-entropy / combined loss from the forward pass
        fps   — wall-clock inference frames per second

    Configurable via cfg.evaluation.metrics:
        dice / mdice, iou / miou, hd95, nsd
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

        # Metric computer — same factory as the Trainer uses.
        self.metric_computer = MetricComputer.from_cfg(cfg)

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

        x = batch["image"].to(self.device, non_blocking=True)
        y = batch["mask"].to(self.device, non_blocking=True)

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
            losses = [self.loss_fn(logits, y) for logits in output]
            loss = sum(losses) / len(losses)
            logits = output[0]
        else:
            logits = output
            loss = self.loss_fn(logits, y)

        return logits, loss

    # =========================================================
    # Evaluate
    # =========================================================

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        """
        Run the full test loop and return aggregated metrics.

        Always returns:
            loss  — mean loss over the test set
            fps   — inference frames per second

        Additionally returns every metric configured in
        ``cfg.evaluation.metrics`` (e.g. dice, iou, hd95, nsd).

        Returns
        -------
        dict[str, float]
        """

        print("\n" + "=" * 60)
        print("TEST EVALUATION")
        print("=" * 60)
        print(
            f"Metrics : {['loss', 'fps'] + self.metric_computer.names}"
        )
        print("=" * 60)

        self.model.eval()

        total_images     = 0
        total_model_time = 0.0

        # Accumulate per-batch values for every metric
        accumulated: dict[str, list[float]] = defaultdict(list)

        progress = tqdm(self.test_loader, desc="Test")

        for batch in progress:

            x, y = self._prepare_batch(batch)

            # -------------------------------------------------
            # Timed inference
            # -------------------------------------------------

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            with self._autocast():
                logits, loss = self._forward(x, y)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            total_model_time += time.perf_counter() - t0

            accumulated["loss"].append(loss.detach().item())

            # -------------------------------------------------
            # Predictions
            # -------------------------------------------------

            predictions = torch.argmax(logits, dim=1)

            # -------------------------------------------------
            # Configurable metrics (via MetricComputer)
            # -------------------------------------------------

            batch_metrics = self.metric_computer.compute(
                predictions, y
            )

            for name, value in batch_metrics.items():
                accumulated[name].append(value)

            total_images += x.shape[0]

            # Progress bar: show first soft metric if available
            first_metric = (
                self.metric_computer.names[0]
                if self.metric_computer.names
                else None
            )
            postfix: dict[str, str] = {
                "loss": f"{loss.item():.4f}",
            }
            if first_metric is not None:
                val = batch_metrics.get(first_metric, float("nan"))
                if not np.isnan(val):
                    postfix[first_metric] = f"{val:.4f}"

            progress.set_postfix(**postfix)

        # =====================================================
        # Aggregate  (nanmean handles hd95/nsd NaN batches)
        # =====================================================

        results: dict[str, float] = {
            name: float(np.nanmean(values)) if values else float("nan")
            for name, values in accumulated.items()
        }

        # FPS always hardcoded
        results["fps"] = total_images / max(total_model_time, 1e-8)

        # =====================================================
        # Console summary
        # =====================================================

        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)

        print(f"Loss    : {results.get('loss', float('nan')):.4f}")
        print(f"FPS     : {results.get('fps', float('nan')):.2f}")

        for metric_name in self.metric_computer.names:
            value = results.get(metric_name, float("nan"))
            label = metric_name.upper()
            if np.isnan(value):
                print(f"{label:<8}: N/A")
            else:
                print(f"{label:<8}: {value:.4f}")

        print("=" * 60)

        # =====================================================
        # W&B logging
        # =====================================================

        if self.logger is not None:
            self.logger.log_test_metrics(results)

        return results