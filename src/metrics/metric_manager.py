from __future__ import annotations

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
    ) -> None:
        self.metrics = metrics
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.atlas_ap = None

        if self.dataset_name == "atlas120k":
            self.atlas_ap = SegmentationAPEvaluator()

    @property
    def names(self) -> list[str]:
        names = list(self.metrics.keys())
        if self.atlas_ap is not None:
            names.extend(["AP", "AP50", "AP75"])
        return names

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

        if self.atlas_ap is not None:
            self.atlas_ap = SegmentationAPEvaluator()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        # Predictions
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1, keepdim=True)

        # Targets
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        predictions_one_hot = one_hot(predictions, num_classes=self.num_classes)
        targets_one_hot = one_hot(targets, num_classes=self.num_classes)

        # MONAI metrics update
        for metric in self.metrics.values():
            metric(y_pred=predictions_one_hot, y=targets_one_hot)

        # ATLAS AP update
        if self.atlas_ap is not None:
            batch_size = logits.shape[0]
            for i in range(batch_size):
                pred_mask = predictions[i, 0].detach().cpu().numpy()
                gt_mask = targets[i, 0].detach().cpu().numpy()
                
                # Calcolo dello score basato sulla media delle probabilità massime
                score = probabilities[i].max(dim=0).values.mean().item()

                self.atlas_ap.add_frame(
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    score=score,
                )

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}

        for name, metric in self.metrics.items():
            value = metric.aggregate()

            if isinstance(value, tuple):
                value = value[0]

            if torch.is_tensor(value):
                value = value.detach().cpu().nanmean().item()
            else:
                value = float(value)

            results[name] = value

        if self.atlas_ap is not None:
            results.update(self.atlas_ap.evaluate())

        return results