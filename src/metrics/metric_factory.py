from __future__ import annotations

from monai.metrics import (
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
)

from src.metrics.metric_manager import MetricManager


class MetricFactory:

    def __init__(self, cfg):

        self.cfg = cfg

        self.num_classes = int(
            cfg.data.num_classes
        )

        evaluation_cfg = cfg.get(
            "evaluation",
            {},
        )

        self.metric_names = list(
            evaluation_cfg.get(
                "metrics",
                [],
            )
        )

        self.foreground_only = bool(
            evaluation_cfg.get(
                "foreground_only",
                True,
            )
        )

        self.ignore_empty = bool(
            evaluation_cfg.get(
                "ignore_empty",
                True,
            )
        )

    def build(self) -> MetricManager:

        metrics = {}

        include_background = not self.foreground_only

        for name in self.metric_names:

            name = name.lower()

            # =================================================
            # Dice
            # =================================================

            if name == "dice":

                metrics["dice"] = DiceMetric(
                    include_background=include_background,
                    reduction="mean",
                    ignore_empty=self.ignore_empty,
                    get_not_nans=True,
                )

            # =================================================
            # IoU
            # =================================================

            elif name == "iou":

                metrics["iou"] = MeanIoU(
                    include_background=include_background,
                    reduction="mean",
                    get_not_nans=True,
                )

            # =================================================
            # Hausdorff Distance 95
            # =================================================

            elif name == "hd95":

                metrics["hd95"] = HausdorffDistanceMetric(
                    include_background=include_background,
                    percentile=95.0,
                    reduction="mean",
                    get_not_nans=True,
                )

            else:

                raise ValueError(
                    f"Unknown metric: {name}"
                )

        return MetricManager(
            metrics=metrics,
            num_classes=self.num_classes,
        )