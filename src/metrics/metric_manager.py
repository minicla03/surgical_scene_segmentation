from __future__ import annotations

import torch
from monai.networks.utils import one_hot


class MetricManager:

    def __init__(
        self,
        metrics,
        num_classes: int,
    ):
        self.metrics = metrics
        self.num_classes = num_classes

    @property
    def names(self) -> list[str]:
        return list(self.metrics.keys())

    def reset(self) -> None:

        for metric in self.metrics.values():
            metric.reset()

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:

        # -----------------------------------------------------
        # Logits
        #
        # [B, C, H, W]
        #        ↓
        # [B, 1, H, W]
        #        ↓
        # [B, C, H, W] one-hot
        # -----------------------------------------------------

        predictions = torch.argmax(
            logits,
            dim=1,
            keepdim=True,
        )

        # -----------------------------------------------------
        # Targets
        #
        # [B, H, W]
        #        ↓
        # [B, 1, H, W]
        #        ↓
        # [B, C, H, W] one-hot
        # -----------------------------------------------------

        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        predictions = one_hot(
            predictions,
            num_classes=self.num_classes,
        )

        targets = one_hot(
            targets,
            num_classes=self.num_classes,
        )

        # -----------------------------------------------------
        # Update MONAI metrics
        # -----------------------------------------------------

        for metric in self.metrics.values():

            metric(
                y_pred=predictions,
                y=targets,
            )

    def compute(self) -> dict[str, float]:

        results: dict[str, float] = {}

        for name, metric in self.metrics.items():

            value = metric.aggregate()

            # get_not_nans=True
            #
            # aggregate() can return:
            #     (metric_value, not_nans)
            #
            if isinstance(value, tuple):
                value = value[0]

            if torch.is_tensor(value):
                value = value.detach().cpu()

                # Defensive handling in case a tensor
                # contains more than one value.
                value = value.nanmean().item()

            else:
                value = float(value)

            results[name] = value

        return results