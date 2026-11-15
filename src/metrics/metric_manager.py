"""Segmentation metrics shared by every model (see docs/metrics_convention.md).

Three families are computed from the same predictions:

* ``dice`` / ``iou`` (and any other MONAI metric in the config): **per image**, then averaged
  over the images in which the class is present in the ground truth.
* ``dice_acc`` / ``iou_acc``: **accumulated** over the whole evaluated set through a
  confusion matrix (TP / FP / FN summed over all pixels of all images).
* ``hd95`` (``HausdorffDistanceMetric``): per image, in pixels, with the conventions below.
  It is expensive, so it is skipped when ``update(..., extra=False)`` is used.

Besides the metrics, ``compute_per_class`` also reports ``pred_pixels`` and
``pred_pixel_percentage``: how many pixels are assigned to each class, for ALL classes. They are
the figure to report for the classes that are N/A (absent from the validation ground truth), whose
Dice is excluded from the averages even when the model predicts them.

Common rule: a class with no ground-truth pixels is N/A (NaN), never 0, and it is left out of
every average. Classes flagged as not evaluable (``evaluable_classes``, e.g. a K-fold fold in
which the class is missing from train or validation) are NaN as well.
"""
from __future__ import annotations

import math
import re
import warnings
from collections.abc import Sequence
from typing import Any

import torch
from monai.metrics import HausdorffDistanceMetric
from monai.networks.utils import one_hot

from .atlas_metrics import SegmentationAPEvaluator

ACCUMULATED_METRICS = ("dice_acc", "iou_acc")


class MetricManager:
    def __init__(
        self,
        metrics: dict[str, Any],
        num_classes: int,
        dataset_name: str | None = None,
        class_names: dict[int, str] | None = None,
        evaluable_classes: Sequence[bool] | None = None,
    ) -> None:
        """
        ``evaluable_classes`` (length ``num_classes``) marks the classes that can be
        evaluated in this run (e.g. in a K-fold fold). The others are reported as NaN
        per class and excluded from the macro average.
        """

        self.metrics = metrics
        self.num_classes = num_classes
        self._eval_mask: torch.Tensor | None = None
        if evaluable_classes is not None:
            mask = torch.as_tensor([bool(v) for v in evaluable_classes], dtype=torch.bool)
            if mask.numel() != num_classes:
                raise ValueError(
                    f"evaluable_classes has {mask.numel()} entries, expected {num_classes}."
                )
            self._eval_mask = mask
        self.dataset_name = dataset_name
        self.class_names = class_names or {i: f"class_{i}" for i in range(num_classes)}

        self.atlas_ap = (
            SegmentationAPEvaluator() if dataset_name == "atlas120k" else None
        )

        # metrics that are expensive and need special handling (HD95)
        self._distance_names = {
            name for name, metric in metrics.items() if isinstance(metric, HausdorffDistanceMetric)
        }

        self._num_updates = 0
        self._confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self._distance_gt_present: list[torch.Tensor] = []
        self._image_diagonal = 0.0

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        names = list(self.metrics)
        names.extend(ACCUMULATED_METRICS)
        if self.atlas_ap is not None:
            names.extend(["AP", "AP50", "AP75"])
        return names

    @property
    def has_data(self) -> bool:
        return self._num_updates > 0

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        if self.atlas_ap is not None:
            self.atlas_ap = SegmentationAPEvaluator()
        self._num_updates = 0
        self._confusion.zero_()
        self._distance_gt_present = []
        self._image_diagonal = 0.0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self, predictions: torch.Tensor, targets: torch.Tensor, extra: bool = True
    ) -> None:
        """Accumulate one batch.

        ``extra=False`` skips the expensive metrics (HD95): use it in the training loop.
        The accumulated Dice/IoU are always updated (a bincount per batch).
        """
        predictions = self._class_map(predictions)
        targets = self._class_map(targets)

        self._update_confusion(predictions, targets)

        predictions_one_hot = one_hot(predictions, num_classes=self.num_classes)
        targets_one_hot = one_hot(targets, num_classes=self.num_classes)

        distance_updated = False
        for name, metric in self.metrics.items():
            if name in self._distance_names:
                if not extra:
                    continue
                with warnings.catch_warnings():
                    # MONAI warns for every image where a class is empty; those cases are
                    # handled explicitly in _distance_buffer.
                    warnings.simplefilter("ignore")
                    metric(y_pred=predictions_one_hot, y=targets_one_hot)
                distance_updated = True
            else:
                metric(y_pred=predictions_one_hot, y=targets_one_hot)

        if distance_updated:
            self._distance_gt_present.append(
                (targets_one_hot.flatten(start_dim=2).sum(dim=2) > 0).cpu()
            )
            height, width = targets_one_hot.shape[-2:]
            self._image_diagonal = max(self._image_diagonal, math.hypot(height, width))

        self._num_updates += 1

        if self.atlas_ap is not None:
            self._update_atlas(predictions, targets)

    def _class_map(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            return tensor.unsqueeze(1).long()

        if tensor.ndim != 4:
            raise ValueError(
                f"Expected [B,H,W], [B,1,H,W] or [B,C,H,W], got {tuple(tensor.shape)}."
            )

        if tensor.shape[1] == 1:
            return tensor.long()

        if tensor.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected 1 or {self.num_classes} channels, got {tensor.shape[1]}."
            )

        return tensor.argmax(dim=1, keepdim=True).long()

    def _update_confusion(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Add the batch to the confusion matrix (rows: ground truth, columns: prediction)."""
        pred = predictions.flatten()
        target = targets.flatten()
        valid = (pred >= 0) & (pred < self.num_classes) & (target >= 0) & (target < self.num_classes)
        index = target[valid] * self.num_classes + pred[valid]
        counts = torch.bincount(index, minlength=self.num_classes**2)
        self._confusion += counts.reshape(self.num_classes, self.num_classes).cpu()

    def _update_atlas(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        for pred, target in zip(predictions, targets):
            self.atlas_ap.add_frame(
                gt_mask=target[0].cpu().numpy(),
                pred_mask=pred[0].cpu().numpy(),
            )

    # ------------------------------------------------------------------
    # Accumulated Dice / IoU
    # ------------------------------------------------------------------
    def confusion_matrix(self) -> torch.Tensor:
        """Confusion matrix of everything accumulated since ``reset`` (rows: GT)."""
        return self._confusion.clone()

    def accumulated_per_class(self) -> dict[str, torch.Tensor]:
        """Per-class accumulated Dice and IoU; NaN where the class has no GT pixels
        or is not evaluable."""
        confusion = self._confusion.double()
        true_positive = confusion.diag()
        false_negative = confusion.sum(dim=1) - true_positive
        false_positive = confusion.sum(dim=0) - true_positive
        has_gt = confusion.sum(dim=1) > 0
        nan = torch.full_like(true_positive, float("nan"))

        dice = torch.where(
            has_gt, 2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp(min=1), nan
        )
        iou = torch.where(
            has_gt, true_positive / (true_positive + false_positive + false_negative).clamp(min=1), nan
        )
        out = {"dice_acc": dice, "iou_acc": iou}
        if self._eval_mask is not None:
            for key, value in out.items():
                out[key] = torch.where(self._eval_mask, value, nan)
        return out

    def prediction_pixels(self) -> dict[str, torch.Tensor]:
        """Pixels the model assigns to each class (column sums of the confusion matrix).

        Not masked: it is meant to be read next to the N/A classes.
        """
        predicted = self._confusion.double().sum(dim=0)
        total = predicted.sum()
        percentage = 100.0 * predicted / total if total > 0 else torch.full_like(predicted, float("nan"))
        return {"pred_pixels": predicted, "pred_pixel_percentage": percentage}

    # ------------------------------------------------------------------
    # Distance metrics (HD95)
    # ------------------------------------------------------------------
    def _distance_buffer(self, metric: Any) -> torch.Tensor | None:
        """Per-image / per-class HD95 with the project convention.

        * class absent from the image ground truth -> NaN (N/A, left out of the average);
        * class present but not predicted (MONAI returns inf) -> image diagonal, the largest
          possible error, so that predicting nothing is never rewarded.
        """
        if not self._distance_gt_present:
            return None
        buffer = self._metric_buffer(metric)
        if buffer is None:
            return None

        buffer = buffer.float().cpu()
        gt_present = torch.cat(self._distance_gt_present)
        if buffer.shape != gt_present.shape:
            raise RuntimeError(
                f"Distance buffer {tuple(buffer.shape)} does not match the ground-truth "
                f"presence {tuple(gt_present.shape)} (include_background=False is not supported)."
            )

        nan = torch.tensor(float("nan"))
        diagonal = torch.tensor(float(self._image_diagonal))
        buffer = torch.where(gt_present & ~torch.isfinite(buffer), diagonal, buffer)
        return torch.where(gt_present, buffer, nan)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    def compute(self) -> dict[str, float]:
        if not self.has_data:
            return self._empty_metrics()

        results = {}
        for name, metric in self.metrics.items():
            if name in self._distance_names:
                buffer = self._distance_buffer(metric)
                if buffer is not None:
                    results[name] = self._masked_macro(buffer)
                continue

            buffer = self._metric_buffer(metric) if self._eval_mask is not None else None
            if buffer is not None and buffer.shape[1] == self.num_classes:
                results[name] = self._masked_macro(buffer)
                continue

            value = metric.aggregate()

            if isinstance(value, tuple):
                value = value[0]

            if torch.is_tensor(value):
                value = value.detach().cpu().nanmean().item()

            results[name] = float(value)

        for name, values in self.accumulated_per_class().items():
            results[name] = float(values.nanmean().item()) if not values.isnan().all() else float("nan")

        if self.atlas_ap is not None:
            results.update(self.atlas_ap.evaluate())

        return results

    def compute_per_class(self) -> dict[str, float]:
        results = {}

        for name, metric in self.metrics.items():
            if name in self._distance_names:
                buffer = self._distance_buffer(metric)
                if buffer is None:
                    continue  # not computed in this pass (extra=False)
            else:
                buffer = self._metric_buffer(metric)

            if buffer is None:
                self._add_nan_classes(results, name)
                continue

            values = buffer.float().nanmean(dim=0)

            for class_idx in range(self.num_classes):
                value = (
                    values[class_idx].item() if class_idx < values.shape[0] else float("nan")
                )
                if self._eval_mask is not None and not bool(self._eval_mask[class_idx]):
                    value = float("nan")
                results[self._metric_name(name, class_idx)] = value

        for name, values in {**self.accumulated_per_class(), **self.prediction_pixels()}.items():
            for class_idx in range(self.num_classes):
                results[self._metric_name(name, class_idx)] = float(values[class_idx].item())

        return results

    def _masked_macro(self, buffer: torch.Tensor) -> float:
        """Macro average over the evaluable classes only.

        Mirrors MONAI's ``reduction='mean'``: mean over the (non-NaN) classes of each
        image first, then mean over the images.
        """
        values = buffer.float()
        if self._eval_mask is not None:
            values = values[:, self._eval_mask.to(values.device)]
        if values.numel() == 0:
            return float("nan")
        return values.nanmean(dim=1).nanmean().item()

    def _metric_buffer(self, metric: Any) -> torch.Tensor | None:
        try:
            buffer = metric.get_buffer()
        except AttributeError:
            return None

        if not torch.is_tensor(buffer) or buffer.ndim < 2:
            return None

        return buffer

    def _empty_metrics(self) -> dict[str, float]:
        results = {name: float("nan") for name in self.metrics if name not in self._distance_names}
        results.update({name: float("nan") for name in ACCUMULATED_METRICS})

        if self.atlas_ap is not None:
            results.update({"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")})

        return results

    def _add_nan_classes(self, results: dict[str, float], metric_name: str) -> None:
        for class_idx in range(self.num_classes):
            results[self._metric_name(metric_name, class_idx)] = float("nan")

    def _metric_name(self, metric: str, class_idx: int) -> str:
        name = self.class_names.get(class_idx, f"class_{class_idx}")
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name)).strip("_")
        return f"{metric}_class_{name or class_idx}"
