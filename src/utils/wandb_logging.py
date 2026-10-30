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
        self.is_active = False
        self.enabled = enabled

        if not self.enabled:
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

    # Generic metrics
    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = False,
    ) -> None:
        if not self.is_active:
            return

        wandb.log(metrics, step=step, commit=commit)

    # Epoch metrics
    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float | None = None,
        val_loss: float | None = None,
        learning_rate: float | None = None,
        metrics: dict[str, float] | None = None,
        step: int | None = None,
    ) -> None:
        if not self.is_active:
            return

        log_step = step if step is not None else epoch + 1
        log_dict: dict[str, Any] = {}

        if train_loss is not None:
            log_dict["train/loss"] = train_loss

        if val_loss is not None:
            log_dict["val/loss"] = val_loss

        if learning_rate is not None:
            log_dict["optimizer/lr"] = learning_rate

        if metrics:
            for name, value in metrics.items():
                # MetricManager può già fornire:
                #   dice
                #   dice/kidney
                #   iou/liver
                #
                # Qui aggiungiamo semplicemente il prefisso val/
                # senza modificare il nome della metrica.
                metric_name = name if name.startswith("val/") else f"val/{name}"
                log_dict[metric_name] = value

        wandb.log(log_dict, step=log_step, commit=True)

    # Scenario / robustness evaluation
    def log_scenario_metrics(
        self,
        scenario: str,
        metrics: dict[str, float],
        prefix: str = "test",
    ) -> None:
        if not self.is_active:
            return

        log_dict = {
            f"{prefix}/stress/{scenario}/{name}": value
            for name, value in metrics.items()
        }
        wandb.log(log_dict)

    # Test results table
    def log_results_table(
        self,
        results: dict[str, dict[str, float]],
        table_name: str = "test/results",
    ) -> None:
        if not self.is_active or not results:
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

    # Qualitative segmentation results
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
        if not self.is_active:
            return

        # Normalize class labels
        class_labels = {int(class_id): str(class_name) for class_id, class_name in class_labels.items()}

        # Flatten temporal/batch dimension if necessary
        if images.ndim == 5:
            images = images.flatten(0, 1)
            labels = labels.flatten(0, 1)
            predictions = predictions.flatten(0, 1)

        # Keep at most RGB channels
        if images.shape[1] > 3:
            images = images[:, :3]

        # W&B table
        table = wandb.Table(columns=["Epoch", "Scenario", "Sample", "Segmentation"])
        n_samples = min(images.shape[0], max_samples)

        for index in range(n_samples):
            # Image
            image = self._prepare_image(images[index])

            # Ground truth
            ground_truth = labels[index].detach().cpu().numpy().squeeze().astype(np.uint8)

            # Prediction
            prediction = predictions[index].detach().cpu().numpy().squeeze().astype(np.uint8)

            # Check class IDs
            mask_ids = set(np.unique(ground_truth).tolist()) | set(np.unique(prediction).tolist())
            missing_labels = sorted(
                int(class_id) for class_id in mask_ids if int(class_id) not in class_labels
            )

            if missing_labels:
                print(f"[W&B] Warning: segmentation masks contain class IDs without labels: {missing_labels}")

            # W&B segmentation image
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
                epoch + 1 if epoch is not None else None,
                scenario,
                f"sample_{index}",
                wandb_image,
            )

        # Log table
        wandb.log(
            {table_name: table},
            step=epoch + 1 if epoch is not None else None,
            commit=False,
        )

    # Checkpoint
    def log_checkpoint(
        self,
        checkpoint_path: str,
        name: str = "model",
        aliases: list[str] | None = None,
    ) -> None:
        if not self.is_active:
            return

        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact, aliases=aliases or [])

    # Model information
    def log_model_info(
        self,
        model: torch.nn.Module,
        model_name: str | None = None,
    ) -> None:
        if not self.is_active:
            return

        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )

        metrics: dict[str, Any] = {
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

    # Image preprocessing
    @staticmethod
    def _prepare_image(image: torch.Tensor) -> np.ndarray:
        """Convert tensor image to uint8 HWC format."""
        image = image.detach().cpu().float().numpy()

        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).astype(np.uint8)

        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)

        return image

    # Configuration
    def log_config(self, config: dict[str, Any]) -> None:
        if not self.is_active:
            return

        wandb.config.update(config, allow_val_change=True)

    # Final results
    def log_final_results(
        self,
        train_results: dict[str, Any],
        test_results: dict[str, float],
    ) -> None:
        if not self.is_active:
            return

        summary: dict[str, Any] = {}
        best_val = train_results.get("best_val_metrics", {})

        for key, value in best_val.items():
            summary[f"train/best_val_{key}"] = value

        summary["train/best_epoch"] = train_results.get("best_epoch", -1)

        for key, value in test_results.items():
            summary[f"test/{key}"] = value

        if wandb.run is not None:
            for key, value in summary.items():
                wandb.run.summary[key] = value

    # Run management
    def finish(self) -> None:
        """Finish the current W&B run."""
        if self.is_active:
            wandb.finish()
            self.is_active = False