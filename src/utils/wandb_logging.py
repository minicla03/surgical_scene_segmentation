from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
import wandb

from pathlib import Path

from src.utils.checkpoint import atomic_torch_save

_PER_CLASS_RE = re.compile(r"^(?P<metric>.+?)_class_(?P<cls>.+)$")


def split_per_class_key(key: str) -> tuple[str, str] | None:
    """Split ``dice_class_liver`` into ``("dice", "liver")``; None if not per-class."""
    match = _PER_CLASS_RE.match(key)
    if match is None:
        return None
    return match.group("metric"), match.group("cls")


def per_class_log_key(split: str, key: str) -> str:
    """Namespace for a metric key.

    Per-class metrics go to their own W&B section ``<split>_per_class/<metric>/<class>``
    so the main ``<split>`` section only holds the aggregate curves.
    """
    parts = split_per_class_key(key)
    if parts is None:
        return f"{split}/{key}"
    metric, class_name = parts
    return f"{split}_per_class/{metric}/{class_name}"


def confusion_matrix_image(
    confusion: np.ndarray,
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    """Render a row-normalised confusion matrix (rows = ground truth) as an RGB array."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    confusion = np.asarray(confusion, dtype=np.float64)
    num_classes = confusion.shape[0]
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums > 0)
    names = [str((class_names or {}).get(i, i)) for i in range(num_classes)]

    size = max(5.0, 0.55 * num_classes + 2.0)
    figure = Figure(figsize=(size, size))
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    axis.set_xticks(range(num_classes))
    axis.set_yticks(range(num_classes))
    axis.set_xticklabels(names, rotation=90, fontsize=7)
    axis.set_yticklabels(names, fontsize=7)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")

    if num_classes <= 20:
        for row in range(num_classes):
            for col in range(num_classes):
                if row_sums[row, 0] > 0:
                    value = normalized[row, col]
                    axis.text(
                        col, row, f"{value:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if value > 0.5 else "black",
                    )

    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def _runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {"torch": torch.__version__}
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


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
        run_id: str | None = None,
        resume: str | None = None,
        job_type: str | None = None,
        group: str | None = None,
        notes: str | None = None,
        save_code: bool = True,
    ) -> None:
        self.is_active = False
        self.enabled = enabled
        self.run_id: str | None = None
        self.resumed: bool = False

        if not self.enabled:
            return

        if wandb.run is None:
            init_kwargs: dict[str, Any] = {
                "project": project,
                "entity": entity,
                "name": name,
                "tags": tags,
                "mode": mode,
                "save_code": save_code,
            }
            for key, value in (("job_type", job_type), ("group", group), ("notes", notes)):
                if value is not None:
                    init_kwargs[key] = value

            full_config = dict(config) if config is not None else {}
            full_config.setdefault("runtime", _runtime_info())

            if run_id is not None:
                init_kwargs["id"] = run_id
                init_kwargs["resume"] = resume or "must"
            else:
                init_kwargs["config"] = full_config
                if resume is not None:
                    init_kwargs["resume"] = resume

            self._init_run(init_kwargs)

            if run_id is not None and wandb.run is not None:
                wandb.config.update(full_config, allow_val_change=True)

        self.is_active = wandb.run is not None
        if wandb.run is not None:
            self.run_id = wandb.run.id
            self.resumed = bool(getattr(wandb.run, "resumed", False))
            self._define_default_metrics()

    @staticmethod
    def _init_run(init_kwargs: dict[str, Any]) -> None:
        """Initialise the run; fall back to offline mode if W&B is unreachable."""
        try:
            wandb.init(**init_kwargs)
        except Exception as exc:  # network/auth problems must not kill the training
            if init_kwargs.get("mode") != "online":
                raise
            print(
                f"[W&B] Warning: online init failed ({exc!r}). Falling back to offline mode; "
                "sync later with `wandb sync`. Run resumption is NOT preserved.",
                flush=True,
            )
            init_kwargs = {k: v for k, v in init_kwargs.items() if k not in {"id", "resume"}}
            init_kwargs["mode"] = "offline"
            wandb.init(**init_kwargs)

    # Metric definitions
    def _define_default_metrics(self) -> None:
        """Use ``epoch`` as x-axis for all per-epoch curves, independent of W&B's internal step."""
        try:
            wandb.define_metric("epoch")
            for namespace in ("train", "val", "optimizer", "train_per_class", "val_per_class"):
                wandb.define_metric(f"{namespace}/*", step_metric="epoch")
            wandb.define_metric("train/loss", step_metric="epoch", summary="min")
            wandb.define_metric("val/loss", step_metric="epoch", summary="min")
        except Exception as exc:
            print(f"[W&B] Warning: define_metric failed: {exc!r}", flush=True)

    def set_monitor(self, monitor: str, mode: str = "max") -> None:
        """Track the best value of the monitored validation metric in the run summary."""
        if not self.is_active:
            return
        try:
            wandb.define_metric(f"val/{monitor}", step_metric="epoch", summary=mode)
        except Exception as exc:
            print(f"[W&B] Warning: define_metric failed: {exc!r}", flush=True)

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

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = False,
    ) -> None:
        """Alias for log_metrics."""
        self.log_metrics(metrics, step=step, commit=commit)

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
    def _build_qualitative_table(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        class_labels: dict[int, str],
        scenario: str = "clean",
        epoch: int | None = None,
        max_samples: int | None = None,
        groups: list[str] | None = None,
        scores: list[float | None] | None = None,
        indices: list[int] | None = None,
    ) -> "wandb.Table":
        class_labels = {int(class_id): str(class_name) for class_id, class_name in class_labels.items()}

        # Flatten temporal/batch dimension if necessary
        if images.ndim == 5:
            images = images.flatten(0, 1)
            labels = labels.flatten(0, 1)
            predictions = predictions.flatten(0, 1)

        # Keep at most RGB channels
        if images.shape[1] > 3:
            images = images[:, :3]

        n_samples = images.shape[0] if max_samples is None else min(images.shape[0], max_samples)

        def _meta(values: list | None) -> list | None:
            return values if values is not None and len(values) >= n_samples else None

        groups, scores, indices = _meta(groups), _meta(scores), _meta(indices)

        table = wandb.Table(
            columns=["Epoch", "Scenario", "Group", "Sample", "Index", "Dice", "Segmentation"]
        )
        rank_per_group: dict[str, int] = {}

        for index in range(n_samples):
            image = self._prepare_image(images[index])
            ground_truth = labels[index].detach().cpu().numpy().squeeze().astype(np.uint8)
            prediction = predictions[index].detach().cpu().numpy().squeeze().astype(np.uint8)

            mask_ids = set(np.unique(ground_truth).tolist()) | set(np.unique(prediction).tolist())
            missing_labels = sorted(int(c) for c in mask_ids if int(c) not in class_labels)
            if missing_labels:
                print(f"[W&B] Warning: segmentation masks contain class IDs without labels: {missing_labels}")

            wandb_image = wandb.Image(
                image,
                masks={
                    "ground_truth": {"mask_data": ground_truth, "class_labels": class_labels},
                    "prediction": {"mask_data": prediction, "class_labels": class_labels},
                },
            )

            group = groups[index] if groups is not None else "sample"
            rank = rank_per_group.get(group, 0)
            rank_per_group[group] = rank + 1
            score = scores[index] if scores is not None else None

            table.add_data(
                epoch,
                scenario,
                group,
                f"{group}_{rank}",
                indices[index] if indices is not None else index,
                round(float(score), 4) if score is not None else None,
                wandb_image,
            )

        return table

    def log_qualitative_masks(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        class_labels: dict[int, str],
        scenario: str = "clean",
        epoch: int | None = None,
        max_samples: int | None = None,
        table_name: str = "qualitative/segmentation",
        groups: list[str] | None = None,
        scores: list[float | None] | None = None,
        indices: list[int] | None = None,
    ) -> None:
        """Add a qualitative table to the current W&B step (not committed)."""
        if not self.is_active:
            return

        table = self._build_qualitative_table(
            images, labels, predictions, class_labels, scenario, epoch,
            max_samples, groups, scores, indices,
        )
        wandb.log({table_name: table}, commit=False)

    def log_qualitative(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        predictions: torch.Tensor,
        class_labels: dict[int, str] | None = None,
        epoch: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Alias/wrapper for log_qualitative_masks."""
        self.log_qualitative_masks(
            images=images,
            labels=masks,
            predictions=predictions,
            class_labels=class_labels or {},
            epoch=epoch,
            **kwargs,
        )

    def log_confusion_matrix(
        self,
        confusion: np.ndarray,
        class_labels: dict[int, str] | None = None,
        key: str = "val/confusion_matrix",
        commit: bool = False,
    ) -> None:
        if not self.is_active:
            return
        wandb.log({key: wandb.Image(confusion_matrix_image(confusion, class_labels))}, commit=commit)

    # Checkpoint
    def log_checkpoint(
        self,
        checkpoint: dict,
        checkpoint_path: str | Path,
        name: str = "model",
        aliases: list[str] | None = None,
        upload_to_wandb: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        checkpoint_path = atomic_torch_save(checkpoint, Path(checkpoint_path))

        if not self.is_active or not upload_to_wandb:
            return

        artifact = wandb.Artifact(
            name=name,
            type="model",
            metadata=metadata,
        )
        artifact.add_file(str(checkpoint_path))

        wandb.log_artifact(
            artifact,
            aliases=aliases or ["latest"],
        )

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

        info: dict[str, Any] = {
            "model/parameters": total_parameters,
            "model/trainable_parameters": trainable_parameters,
        }
        if model_name is not None:
            info["model/name"] = model_name

        if not self.resumed:
            wandb.config.update(info, allow_val_change=True)
            for key, value in info.items():
                wandb.run.summary[key] = value

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

        if image.min() < 0.0 or image.max() > 1.0:
            # Normalised (mean/std) or 0-255 input: rescale per image for display only.
            image = (image - image.min()) / max(float(image.max() - image.min()), 1e-8)
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
        test_results: dict[str, Any],
    ) -> None:
        if not self.is_active:
            return

        summary: dict[str, Any] = {}
        best_val = train_results.get("best_metrics", train_results.get("best_val_metrics", {}))

        for key, value in best_val.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                summary[f"train/best_val_{key}"] = float(value)

        if "best_epoch" in train_results:
            summary["train/best_epoch"] = train_results["best_epoch"]

        if "best_score" in train_results:
            summary["train/best_score"] = float(train_results["best_score"])

        test_metrics = test_results.get("metrics", test_results)
        if isinstance(test_metrics, dict):
            for key, value in test_metrics.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    summary[f"test/{key}"] = float(value)

        test_per_class = test_results.get("per_class", {})
        if isinstance(test_per_class, dict):
            for key, value in test_per_class.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    summary[per_class_log_key("test", key)] = float(value)

        if wandb.run is not None:
            for key, value in summary.items():
                wandb.run.summary[key] = value

    # Run management
    def finish(self) -> None:
        """Finish the current W&B run."""
        if self.is_active:
            wandb.finish()
            self.is_active = False


    # Test results
    def log_test_results(
        self,
        metrics: dict[str, float],
        per_class: dict[str, float] | None = None,
        commit: bool = True,
    ) -> None:
        if not self.is_active:
            return

        log_dict: dict[str, Any] = {f"test/{key}": value for key, value in metrics.items()}
        for key, value in (per_class or {}).items():
            log_dict[per_class_log_key("test", key)] = value

        wandb.log(log_dict, commit=commit)

    # Epoch logging
    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        per_class_metrics: dict[str, float] | None = None,
        train_per_class_metrics: dict[str, float] | None = None,
        learning_rate: float | None = None,
        qualitative_batch: dict[str, Any] | None = None,
        class_labels: dict[int, str] | None = None,
        qualitative_frequency: int = 1,
    ) -> None:
        """Log everything for one epoch in a single commit.

        Curves use ``epoch`` as x-axis (see ``_define_default_metrics``); no manual
        W&B step is passed, so later test logs can never collide with it.
        """

        if not self.is_active:
            return

        metrics: dict[str, Any] = {}
        for split, values in (
            ("train", train_metrics),
            ("val", val_metrics),
            ("train", train_per_class_metrics),
            ("val", per_class_metrics),
        ):
            for key, value in (values or {}).items():
                metrics[per_class_log_key(split, key)] = value

        if learning_rate is not None:
            metrics["optimizer/lr"] = learning_rate

        metrics["epoch"] = epoch

        if qualitative_batch is not None and epoch % qualitative_frequency == 0:
            metrics["qualitative/segmentation"] = self._build_qualitative_table(
                images=qualitative_batch["image"],
                labels=qualitative_batch["mask"],
                predictions=qualitative_batch["prediction"],
                class_labels=class_labels or {},
                scenario="val",
                epoch=epoch,
                groups=qualitative_batch.get("group"),
                scores=qualitative_batch.get("score"),
                indices=qualitative_batch.get("index"),
            )

            confusion = qualitative_batch.get("confusion_matrix")
            if confusion is not None:
                metrics["val/confusion_matrix"] = wandb.Image(
                    confusion_matrix_image(confusion, class_labels)
                )

        wandb.log(metrics, commit=True)
