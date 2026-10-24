from __future__ import annotations

from typing import Any

import numpy as np
import torch
import wandb


class WandbLogger:
    
    def __init__(
        self,
        enabled: bool = True,
        project: str | None = None,
        entity: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        mode: str = "online",
    ) -> None:
        
        self.enabled = enabled

        if not self.enabled:
            self.is_active = False
            return

        if wandb.run is None:
            wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                tags=tags,
                mode=mode,
            )

        self.is_active = wandb.run is not None

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
    ) -> None:

        if not self.is_active:
            return

        wandb.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float | None = None,
        val_loss: float | None = None,
        learning_rate: float | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Log metrics collected during a training epoch."""

        if not self.is_active:
            return

        log_dict: dict[str, Any] = {
            "epoch": epoch,
        }

        if train_loss is not None:
            log_dict["train/loss"] = train_loss

        if val_loss is not None:
            log_dict["val/loss"] = val_loss

        if learning_rate is not None:
            log_dict["optimizer/lr"] = learning_rate

        if metrics:
            log_dict.update(
                {
                    f"val/{name}": value
                    for name, value in metrics.items()
                }
            )

        wandb.log(log_dict, step=epoch)

    # ------------------------------------------------------------------
    # Test metrics
    # ------------------------------------------------------------------

    def log_test_metrics(
        self,
        metrics: dict[str, float],
        prefix: str = "test",
    ) -> None:
        """Log final test metrics."""

        if not self.is_active:
            return

        log_dict = {
            f"{prefix}/{name}": value
            for name, value in metrics.items()
        }

        wandb.log(log_dict)

    # ------------------------------------------------------------------
    # Scenario / robustness evaluation
    # ------------------------------------------------------------------

    def log_scenario_metrics(
        self,
        scenario: str,
        metrics: dict[str, float],
        prefix: str = "test",
    ) -> None:
        """Log metrics associated with a specific evaluation scenario.

        Example:
            test/stress/blur/dice
            test/stress/blur/iou
            test/stress/blur/hd95
        """

        if not self.is_active:
            return

        log_dict = {
            f"{prefix}/stress/{scenario}/{name}": value
            for name, value in metrics.items()
        }

        wandb.log(log_dict)

    # ------------------------------------------------------------------
    # Test results table
    # ------------------------------------------------------------------

    def log_results_table(
        self,
        results: dict[str, dict[str, float]],
        table_name: str = "test/results",
    ) -> None:
        """Log evaluation results as a W&B table.

        Args:
            results:
                Dictionary in the form:

                {
                    "clean": {
                        "dice": 0.85,
                        "iou": 0.74,
                        "hd95": 12.4,
                        "inference_fps": 42.1,
                    },
                    "blur": {
                        ...
                    },
                }
        """

        if not self.is_active:
            return

        if not results:
            return

        metric_names = sorted(
            {
                metric
                for scenario_metrics in results.values()
                for metric in scenario_metrics
            }
        )

        columns = ["Scenario"] + metric_names
        table = wandb.Table(columns=columns)

        for scenario, scenario_metrics in results.items():
            row = [scenario]

            for metric in metric_names:
                value = scenario_metrics.get(metric)

                if isinstance(value, (float, np.floating)):
                    value = round(float(value), 4)

                row.append(value)

            table.add_data(*row)

        wandb.log({table_name: table})

    # ------------------------------------------------------------------
    # Qualitative segmentation results
    # ------------------------------------------------------------------

    def log_qualitative_masks(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        class_labels: dict[int, str],
        scenario: str = "clean",
        epoch: int | None = None,
        max_samples: int = 4,
        table_name: str = "qualitative/segmentation",
    ) -> None:
        """Log qualitative segmentation predictions to W&B.

        Args:
            images:
                Tensor with shape [B, C, H, W].

            labels:
                Ground-truth masks.

            predictions:
                Predicted masks.

            class_labels:
                Mapping from class index to class name.

            scenario:
                Evaluation scenario name.

            epoch:
                Current epoch.

            max_samples:
                Maximum number of samples to log.
        """

        if not self.is_active:
            return

        # Support [B, T, C, H, W]
        if images.ndim == 5:
            images = images.flatten(0, 1)
            labels = labels.flatten(0, 1)
            predictions = predictions.flatten(0, 1)

        # Keep only RGB channels.
        if images.shape[1] > 3:
            images = images[:, :3]

        table = wandb.Table(
            columns=[
                "Epoch",
                "Scenario",
                "Sample",
                "Segmentation",
            ]
        )

        n_samples = min(images.shape[0], max_samples)

        for index in range(n_samples):
            image = self._prepare_image(images[index])

            ground_truth = (
                labels[index]
                .detach()
                .cpu()
                .numpy()
                .squeeze()
                .astype(np.uint8)
            )

            prediction = (
                predictions[index]
                .detach()
                .cpu()
                .numpy()
                .squeeze()
                .astype(np.uint8)
            )

            wandb_image = wandb.Image(
                image,
                masks={
                    "ground_truth": {
                        "mask_data": ground_truth,
                        "class_labels": class_labels,
                    },
                    "prediction": {
                        "mask_data": prediction,
                        "class_labels": class_labels,
                    },
                },
            )

            table.add_data(
                epoch,
                scenario,
                f"sample_{index}",
                wandb_image,
            )

        wandb.log({table_name: table})

    def log_checkpoint(
        self,
        checkpoint_path: str,
        name: str = "model",
        aliases: list[str] | None = None,
    ) -> None:

        if not self.is_active:
            return

        artifact = wandb.Artifact(
            name=name,
            type="model",
        )

        artifact.add_file(checkpoint_path)

        wandb.log_artifact(
            artifact,
            aliases=aliases or [],
        )

    # ------------------------------------------------------------------
    # Model information
    # ------------------------------------------------------------------

    def log_model_info(
        self,
        model: torch.nn.Module,
        model_name: str | None = None,
    ) -> None:
        """Log basic model information."""

        if not self.is_active:
            return

        total_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        metrics = {
            "model/parameters": total_parameters,
            "model/trainable_parameters": trainable_parameters,
        }

        if model_name is not None:
            metrics["model/name"] = model_name

        wandb.log(metrics)

        print("\n" + "=" * 60)
        print("MODEL INFORMATION")
        print("=" * 60)

        if model_name:
            print(f"Model:              {model_name}")

        print(f"Parameters:         {total_parameters:,}")
        print(f"Trainable params:   {trainable_parameters:,}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_image(image: torch.Tensor) -> np.ndarray:
        """Convert a tensor image to uint8 HWC format."""

        image = (
            image.detach()
            .cpu()
            .float()
            .numpy()
        )

        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        image = np.clip(image, 0.0, 1.0)

        image = (image * 255).astype(np.uint8)

        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)

        return image

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def log_config(self, config: dict[str, Any]) -> None:
        """Update the W&B run configuration."""

        if not self.is_active:
            return

        wandb.config.update(
            config,
            allow_val_change=True,
        )

    def finish(self) -> None:
        """Finish the current W&B run."""

        if self.is_active:
            wandb.finish()
            self.is_active = False