from __future__ import annotations

import re
from typing import Any

import torch
from monai.networks.utils import one_hot

from .atlas_metrics import SegmentationAPEvaluator


class MetricManager:
    def __init__(
        self,
        metrics: dict[str, Any],
        num_classes: int,
        dataset_name: str | None = None,
        class_names: dict[int, str] | None = None,
    ) -> None:

        self.metrics = metrics
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.class_names = class_names or {i: f"class_{i}" for i in range(num_classes)}
        
        self.atlas_ap = (
            SegmentationAPEvaluator() if dataset_name == "atlas120k" else None
        )
        self._num_updates = 0

    @property
    def names(self) -> list[str]:
        names = list(self.metrics)
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

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = self._class_map(predictions)
        targets = self._class_map(targets)

        predictions_one_hot = one_hot(predictions, num_classes=self.num_classes)
        targets_one_hot = one_hot(targets, num_classes=self.num_classes)

        for metric in self.metrics.values():
            metric(y_pred=predictions_one_hot, y=targets_one_hot)

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

    def _update_atlas(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        for pred, target in zip(predictions, targets):
            self.atlas_ap.add_frame(
                gt_mask=target[0].cpu().numpy(),
                pred_mask=pred[0].cpu().numpy(),
            )

    def compute(self) -> dict[str, float]:
        if not self.has_data:
            return self._empty_metrics()

        results = {}
        for name, metric in self.metrics.items():
            value = metric.aggregate()

            if isinstance(value, tuple):
                value = value[0]

            if torch.is_tensor(value):
                value = value.detach().cpu().nanmean().item()

            results[name] = float(value)

        if self.atlas_ap is not None:
            results.update(self.atlas_ap.evaluate())

        return results

    def compute_per_class(self) -> dict[str, float]:
        results = {}

        for name, metric in self.metrics.items():
            buffer = self._metric_buffer(metric)

            if buffer is None:
                self._add_nan_classes(results, name)
                continue

            values = buffer.float().nanmean(dim=0)

            for class_idx in range(self.num_classes):
                value = (
                    values[class_idx].item() if class_idx < values.shape[0] else float("nan")
                )
                results[self._metric_name(name, class_idx)] = value

        return results

    def _metric_buffer(self, metric: Any) -> torch.Tensor | None:
        try:
            buffer = metric.get_buffer()
        except AttributeError:
            return None

        if not torch.is_tensor(buffer) or buffer.ndim < 2:
            return None

        return buffer

    def _empty_metrics(self) -> dict[str, float]:
        results = {name: float("nan") for name in self.metrics}

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