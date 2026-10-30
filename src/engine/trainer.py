from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_fn: torch.nn.Module,
        scaler: Any,
        metric_manager: Any,
        cfg: Any,
        device: torch.device,
        logger: Any | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.metric_manager = metric_manager
        self.cfg = cfg
        self.device = device
        self.logger = logger

        train_cfg = cfg.train
        self.max_epochs = int(train_cfg.get("max_epochs", 50))
        self.accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
        self.eval_freq = max(1, int(train_cfg.get("eval_freq", 1)))

        logging_cfg = cfg.get("logging", {})
        qualitative_cfg = logging_cfg.get("qualitative", {})
        self.qualitative_enabled = bool(qualitative_cfg.get("enabled", True))
        self.qualitative_freq = max(1, int(qualitative_cfg.get("frequency", 1)))
        self.qualitative_max_samples = max(1, int(qualitative_cfg.get("max_samples", 4)))

        early_stopping_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = bool(early_stopping_cfg.get("enabled", True))
        self.patience = int(early_stopping_cfg.get("patience", 10))
        self.monitor = str(early_stopping_cfg.get("monitor", "dice"))
        self.monitor_mode = str(early_stopping_cfg.get("mode", "max")).lower()

        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("early_stopping.mode must be either 'min' or 'max'")

        self.metric_names = list(self.metric_manager.names)
        available_metrics = set(self.metric_names) | {"loss", "fps"}

        if self.monitor not in available_metrics:
            raise ValueError(
                f"Early stopping monitor '{self.monitor}' is not available. "
                f"Available metrics: {sorted(available_metrics)}"
            )

        self.best_score = float("inf") if self.monitor_mode == "min" else -float("inf")
        self.best_epoch = 0
        self.best_metrics: dict[str, float] = {}
        self.epochs_without_improvement = 0

        self.output_dir = Path(cfg.get("output_dir", "./outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.output_dir / "best.pt"

    # Forward
    def _forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.
        Returns:
            logits/predictions,
            loss
        """
        # Mask2Former
        if getattr(self.model, "is_mask2former", False):
            if training:
                outputs = self.model(pixel_values=images, labels=masks)
                loss = getattr(outputs, "loss", None)

                if loss is None:
                    raise RuntimeError("Mask2Former did not return a training loss.")

                logits = getattr(outputs, "logits", None)
                if logits is None:
                    raise RuntimeError("Mask2Former did not return logits.")

                return logits, loss

            outputs = self.model(pixel_values=images)
            logits = getattr(outputs, "logits", None)

            if logits is None:
                raise RuntimeError("Mask2Former did not return logits.")

            return logits, None

        # Standard segmentation model
        logits = self.model(images)
        loss = None
        if masks is not None:
            loss = self.loss_fn(logits, masks)

        return logits, loss

    # Batch preparation
    def _prepare_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        return images, masks

    # AMP
    def _autocast(self):
        precision = str(self.cfg.train.get("precision", "32"))
        enabled = precision == "16-mixed" and self.device.type == "cuda"
        return torch.amp.autocast(device_type="cuda", enabled=enabled)

    # Training
    def train_epoch(
        self,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch_idx, batch in enumerate(progress):
            images, masks = self._prepare_batch(batch)

            with self._autocast():
                _, loss = self._forward(images, masks, training=True)

                if loss is None:
                    raise RuntimeError("Training loss is None.")

                loss_for_backward = loss / self.accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            should_update = (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader)

            if should_update:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            loss_value = float(loss.detach().item())
            total_loss += loss_value
            num_batches += 1

            progress.set_postfix(loss=f"{loss_value:.4f}")

        return {"loss": total_loss / max(num_batches, 1)}

    # Validation
    def validate(
        self,
        epoch: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, torch.Tensor] | None]:
        self.model.eval()
        self.metric_manager.reset()
        total_loss = 0.0
        num_batches = 0
        total_images = 0
        total_model_time = 0.0
        qualitative_batch = None

        progress = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress):
                images, masks = self._prepare_batch(batch)

                if self.qualitative_enabled and qualitative_batch is None and batch_idx == 0:
                    max_samples = min(self.qualitative_max_samples, images.shape[0])
                    qualitative_batch = {
                        "image": images[:max_samples].detach().cpu(),
                        "mask": masks[:max_samples].detach().cpu(),
                    }

                start_time = time.perf_counter()

                with self._autocast():
                    logits, model_loss = self._forward(images, masks, training=False)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time
                total_model_time += elapsed
                total_images += images.shape[0]

                # Standard models compute validation loss here.
                # Mask2Former may already provide it.
                loss = model_loss
                if loss is None:
                    loss = self.loss_fn(logits, masks)

                total_loss += float(loss.detach().item())
                num_batches += 1

                self.metric_manager.update(logits, masks)

        results = self.metric_manager.compute()
        per_class_results = self.metric_manager.compute_per_class()

        results["loss"] = total_loss / max(num_batches, 1)
        results["fps"] = total_images / max(total_model_time, 1e-8)
        results = {key: float(value) for key, value in results.items()}

        return results, per_class_results, qualitative_batch

    # Checkpoint
    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "metrics": metrics,
            "config": self.cfg,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)

    # Early stopping
    def _is_improvement(self, score: float) -> bool:
        if self.monitor_mode == "min":
            return score < self.best_score
        return score > self.best_score

    def _update_best(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> bool:
        if self.monitor not in metrics:
            raise KeyError(
                f"Monitor metric '{self.monitor}' not found in validation metrics: "
                f"{list(metrics.keys())}"
            )

        score = float(metrics[self.monitor])

        if not np.isfinite(score) or not self._is_improvement(score):
            return False

        self.best_score = score
        self.best_epoch = epoch
        self.best_metrics = dict(metrics)

        self._save_checkpoint(self.best_checkpoint_path, epoch, metrics)
        self.epochs_without_improvement = 0

        return True

    # Logging
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        per_class_results: dict[str, float],
        qualitative_batch: dict[str, torch.Tensor] | None,
    ) -> None:
        if self.logger is None:
            return

        metrics = {}
        for key, value in train_metrics.items():
            metrics[f"train/{key}"] = value

        for key, value in val_metrics.items():
            metrics[f"val/{key}"] = value

        for key, value in per_class_results.items():
            metrics[f"val/{key}"] = value

        metrics["epoch"] = epoch

        if self.scheduler is not None:
            try:
                metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            except Exception:
                pass

        self.logger.log(metrics, step=epoch)

        # Per-class metric names
        num_classes = int(self.cfg.data.num_classes)
        class_labels = {index: f"class_{index}" for index in range(num_classes)}
        configured_labels = self.cfg.data.get("classes", self.cfg.data.get("class_labels", None))

        if configured_labels is not None:
            if isinstance(configured_labels, dict):
                class_labels = {int(key): str(value) for key, value in configured_labels.items()}
            elif isinstance(configured_labels, (list, tuple)):
                class_labels = {index: str(name) for index, name in enumerate(configured_labels)}

        # Qualitative logging
        if not self.qualitative_enabled or qualitative_batch is None or epoch % self.qualitative_freq != 0:
            return

        images = qualitative_batch["image"]
        masks = qualitative_batch["mask"]

        with torch.no_grad():
            with self._autocast():
                logits, _ = self._forward(
                    images.to(self.device),
                    masks.to(self.device),
                    training=False,
                )

        predictions = torch.argmax(logits, dim=1).detach().cpu()
        images = images.detach().cpu()
        masks = masks.detach().cpu()

        self.logger.log_qualitative(
            images=images,
            masks=masks,
            predictions=predictions,
            class_labels=class_labels,
            step=epoch,
        )

    # Training loop
    def fit(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = self.train_epoch(epoch)

            # Scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Validation
            if epoch % self.eval_freq == 0:
                val_metrics, per_class_results, qualitative_batch = self.validate(epoch)

                self._log_epoch(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    per_class_results=per_class_results,
                    qualitative_batch=qualitative_batch,
                )

                improved = self._update_best(epoch, val_metrics)

                if not improved:
                    self.epochs_without_improvement += 1

                epoch_time = time.perf_counter() - epoch_start
                epoch_results = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "per_class": per_class_results,
                    "epoch_time": epoch_time,
                }
                history.append(epoch_results)

                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_loss={train_metrics.get('loss', float('nan')):.4f} "
                    f"val_loss={val_metrics.get('loss', float('nan')):.4f} "
                    f"{self.monitor}={val_metrics.get(self.monitor, float('nan')):.4f}"
                )

                if improved:
                    print(f"  -> New best checkpoint ({self.monitor}={self.best_score:.4f})")

                # Early stopping
                if self.early_stopping_enabled and self.epochs_without_improvement >= self.patience:
                    print("Early stopping triggered.")
                    break

            else:
                epoch_time = time.perf_counter() - epoch_start
                epoch_results = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "epoch_time": epoch_time,
                }
                history.append(epoch_results)

        return {
            "history": history,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "best_checkpoint": str(self.best_checkpoint_path),
        }

    # Utilities
    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        if not values:
            return 0.0

        finite_values = [value for value in values if np.isfinite(value)]

        if not finite_values:
            return 0.0

        return float(np.mean(finite_values))