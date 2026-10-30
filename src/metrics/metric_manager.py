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

        # Mapping:
        # class index -> class name
        #
        # Example:
        # {
        #     0: "background",
        #     1: "liver",
        #     2: "kidney",
        # }
        if class_names is None:
            self.class_names = {i: f"class_{i}" for i in range(num_classes)}
        else:
            self.class_names = class_names

        # ATLAS-specific AP metrics.
       
        self.atlas_ap = None

        if self.dataset_name == "atlas120k":
            self.atlas_ap = SegmentationAPEvaluator()

    
    @property
    def names(self) -> list[str]:
        """Return the names of all metrics managed by this class."""
        names = list(self.metrics.keys())

        if self.atlas_ap is not None:
            names.extend(["AP", "AP50", "AP75"])

        return names
    
    def reset(self) -> None:
        """Reset all accumulated metric state."""
        for metric in self.metrics.values():
            metric.reset()

        if self.atlas_ap is not None:
            self.atlas_ap = SegmentationAPEvaluator()


    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update all metrics using one batch.

        Parameters
        ----------
        logits:
            Model output with shape [B, C, H, W].

        targets:
            Ground-truth masks with shape [B, H, W]
            or [B, 1, H, W].
        """

        # Predictions
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1, keepdim=True)

        # Targets
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        # Convert to one-hot
        predictions_one_hot = one_hot(predictions, num_classes=self.num_classes)
        targets_one_hot = one_hot(targets, num_classes=self.num_classes)


        for metric in self.metrics.values():
            metric(y_pred=predictions_one_hot, y=targets_one_hot)

        # ATLAS AP
        if self.atlas_ap is not None:
            batch_size = logits.shape[0]

            for i in range(batch_size):
                pred_mask = predictions[i, 0].detach().cpu().numpy()
                gt_mask = targets[i, 0].detach().cpu().numpy()

                # Confidence score based on the average maximum class probability.
                score = probabilities[i].max(dim=0).values.mean().item()

                self.atlas_ap.add_frame(
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    score=score,
                )

    def compute(self) -> dict[str, float]:
        """
        Compute the aggregated value of every metric.
        """
        results: dict[str, float] = {}

        for name, metric in self.metrics.items():
            value = metric.aggregate()

            # Some MONAI metrics return tuples.
            if isinstance(value, tuple):
                value = value[0]

            if torch.is_tensor(value):
                value = value.detach().cpu().nanmean().item()
            else:
                value = float(value)

            results[name] = value

        # ATLAS AP metrics
        if self.atlas_ap is not None:
            results.update(self.atlas_ap.evaluate())

        return results

    
    def compute_per_class(self) -> dict[str, float]:
        """
        Compute metric values for each class.

        Returns
        -------
        dict[str, float]
            Flat dictionary suitable for logging.

        Example
        -------
        {
            "dice_class_background": 0.92,
            "dice_class_liver": 0.85,
            "iou_class_liver": 0.78,
        }

        Notes
        -----
        MONAI metric buffers are expected to have shape:

            [N_samples, N_classes]

        The mean is computed independently for every class.

        Classes whose value is NaN are skipped.
        """
        results: dict[str, float] = {}

        for metric_name, metric in self.metrics.items():
            
            try:
                buffer = metric.get_buffer()
            except AttributeError:
                continue

            if buffer is None or not torch.is_tensor(buffer):
                continue

            # Expected:
            # [N_samples, N_classes]
            if buffer.ndim < 2:
                continue

            # Mean over samples
            per_class = buffer.float().nanmean(dim=0)


            for class_idx, value in enumerate(per_class):
                # Skip NaN classes.
                if torch.isnan(value):
                    continue

                class_name = self.class_names.get(class_idx, f"class_{class_idx}")

                class_name = re.sub(r"[^a-zA-Z0-9_]+", "_", class_name).strip("_")

                results[f"{metric_name}_class_{class_name}"] = value.item()

        return results