from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.utils.checkpoint import (
    atomic_torch_save,
    build_checkpoint_paths,
    build_epoch_checkpoint_path,
    capture_rng_state,
    format_lr_str,
    load_checkpoint_file,
    resolve_checkpoint_dir,
    restore_rng_state,
    sanitize_artifact_name,
)


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
        dataloader_generator: torch.Generator | None = None,
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
        self.dataloader_generator = dataloader_generator
        self.class_labels = getattr(self.metric_manager, "class_names", {})

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
        self.start_epoch = 1
        self.history: list[dict[str, Any]] = []
        self.resumed_from: Path | None = None

        ckpt_cfg = cfg.get("checkpoint") or train_cfg.get("checkpoint") or {}
        self.save_last = bool(ckpt_cfg.get("save_last", True))
        save_every = ckpt_cfg.get("save_every_n_epochs", None)
        self.save_every_n_epochs = int(save_every) if save_every is not None else None
        self.upload_best_to_wandb = bool(ckpt_cfg.get("upload_best_to_wandb", True))
        self.upload_last_to_wandb = bool(ckpt_cfg.get("upload_last_to_wandb", False))

        self.output_dir = resolve_checkpoint_dir(cfg)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.initial_lr = float(self.optimizer.param_groups[0]["lr"])
        self.initial_lr_str = format_lr_str(self.initial_lr)

        ckpt_paths = build_checkpoint_paths(
            output_dir=self.output_dir,
            dataset_name=str(self.cfg.data.name),
            model_name=str(self.cfg.model.name),
            lr=self.initial_lr,
        )
        self.best_checkpoint_path = ckpt_paths["best"]
        self.last_checkpoint_path = ckpt_paths["last"]
        self.artifact_base_name = sanitize_artifact_name(
            f"{self.cfg.data.name}_{self.cfg.model.name}_lr_{self.initial_lr_str}"
        )

    # Forward
    def _forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        
        if getattr(self.model, "is_mask2former", False):
            return self.model(images, masks)

        predictions = self.model(images)

        return predictions, None

    # Batch preparation
    def _prepare_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
    
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)

        # Semantic segmentation masks:
        # [B, H, W] -> [B, 1, H, W]
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        if masks.ndim != 4:
            raise ValueError(
                f"Invalid mask shape: {tuple(masks.shape)}. "
                "Expected [B, 1, H, W] for multiclass semantic segmentation."
            )

        if masks.shape[1] != 1:
            raise ValueError(
                f"Invalid mask channel dimension: {tuple(masks.shape)}. "
                "Expected [B, 1, H, W], not one-hot [B, C, H, W]."
            )

        return images, masks

    # AMP
    def _autocast(self):
        precision = str(self.cfg.train.get("precision", "32"))
        enabled = precision == "16-mixed" and self.device.type == "cuda"
        return torch.amp.autocast(device_type="cuda", enabled=enabled)

    def _optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
        )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

    # Training
    def train_epoch(self, epoch: int) -> tuple[float, dict[str, float], dict[str, float]]:
        
        self.model.train()
        self.metric_manager.reset()

        self.optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        num_batches = 0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
        )

        for step, batch in enumerate(progress):
            
            images, masks = self._prepare_batch(batch)

            with self._autocast():
                predictions, loss = self._forward(images, masks)

                if loss is None:
                    if self.loss_fn is None:
                        raise RuntimeError(
                            "Model returned no loss and no loss function is configured."
                        )

                    loss = self.loss_fn(predictions, masks)

            loss_for_backward = loss / self.accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                self._optimizer_step()

            self.metric_manager.update(predictions.detach(), masks.detach())

            running_loss += loss.detach().item()
            num_batches += 1

            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = self.metric_manager.compute()
        loss_value = running_loss / max(num_batches, 1)
        per_class = self.metric_manager.compute_per_class()

        return loss_value, metrics, per_class


    # Validation
    @torch.no_grad()
    def validate(
        self, epoch: int
    ) -> tuple[float, dict[str, float], dict[str, float], dict[str, torch.Tensor] | None]:
        
        self.model.eval()
        self.metric_manager.reset()

        running_loss = 0.0
        num_batches = 0
        qualitative_batch = None

        progress = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch} [Val]",
            leave=False,
        )

        for batch_index, batch in enumerate(progress):
            images, masks = self._prepare_batch(batch)

            with self._autocast():
                predictions, loss = self._forward(images, masks)

                if loss is None:
                    if self.loss_fn is None:
                        raise RuntimeError("Model returned no loss and no loss function is configured.")
                    loss = self.loss_fn(predictions, masks)

            self.metric_manager.update(predictions.detach(), masks.detach())

            running_loss += loss.detach().item()
            num_batches += 1

            if (
                batch_index == 0
                and self.qualitative_enabled
                and epoch % self.qualitative_freq  == 0
            ):
                n_samples = min(self.qualitative_max_samples, images.shape[0])
                
                prediction = predictions[:n_samples]

                if prediction.ndim == 4:
                    prediction = prediction.argmax(dim=1)

                qualitative_batch = {
                    "image": images[:n_samples].detach().cpu(),
                    "mask": masks[:n_samples].detach().cpu(),
                    "prediction": prediction.detach().cpu(),
                }

            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = self.metric_manager.compute()
        loss_value = running_loss / max(num_batches, 1)
        per_class = self.metric_manager.compute_per_class()

        return loss_value, metrics, per_class, qualitative_batch

    # Checkpoint
    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict[str, float],
        kind: str = "best",
        scheduler_stepped: bool = False,
        upload_to_wandb: bool = True,
        aliases: list[str] | None = None,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scheduler_stepped": scheduler_stepped,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "epochs_without_improvement": self.epochs_without_improvement,
            "history": self.history,
            "metrics": metrics,
            "config": self.cfg,
            "rng_state": capture_rng_state(self.dataloader_generator),
            "wandb_run_id": (
                self.logger.run_id
                if self.logger is not None and getattr(self.logger, "is_active", False)
                else None
            ),
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        if self.logger is not None:
            best_score_meta = (
                float(self.best_score) if np.isfinite(self.best_score) else None
            )
            metadata = {
                "epoch": int(epoch),
                "kind": kind,
                "dataset": str(self.cfg.data.name),
                "model": str(self.cfg.model.name),
                "lr": self.initial_lr_str,
                "best_score": best_score_meta,
            }
            self.logger.log_checkpoint(
                checkpoint=checkpoint,
                checkpoint_path=path,
                name=self.artifact_base_name,
                aliases=aliases or [f"epoch-{epoch}", kind, "latest"],
                upload_to_wandb=upload_to_wandb,
                metadata=metadata,
            )
        else:
            atomic_torch_save(checkpoint, path)

    def _save_end_of_epoch_checkpoints(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        if self.save_last:
            self._save_checkpoint(
                path=self.last_checkpoint_path,
                epoch=epoch,
                metrics=metrics,
                kind="last",
                scheduler_stepped=True,
                upload_to_wandb=self.upload_last_to_wandb,
                aliases=[f"epoch-{epoch}", "last", "latest"],
            )

        if (
            self.save_every_n_epochs is not None
            and self.save_every_n_epochs > 0
            and epoch % self.save_every_n_epochs == 0
        ):
            epoch_path = build_epoch_checkpoint_path(
                output_dir=self.output_dir,
                dataset_name=str(self.cfg.data.name),
                model_name=str(self.cfg.model.name),
                lr=self.initial_lr,
                epoch=epoch,
            )
            self._save_checkpoint(
                path=epoch_path,
                epoch=epoch,
                metrics=metrics,
                kind="periodic",
                scheduler_stepped=True,
                upload_to_wandb=self.upload_last_to_wandb,
                aliases=[f"epoch-{epoch}", "periodic"],
            )

    def _align_resumed_scheduler(
        self,
        checkpoint: dict[str, Any],
        resumed_epoch: int,
    ) -> None:
        scheduler_stepped = checkpoint.get("scheduler_stepped")
        is_plateau = isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )

        if is_plateau:
            if scheduler_stepped is False or scheduler_stepped is None:
                metrics = checkpoint.get("metrics") or {}
                if self.monitor in metrics:
                    self.scheduler.step(metrics[self.monitor])
            return

        if scheduler_stepped is None:
            # Legacy checkpoints saved inside _update_best were captured before scheduler.step().
            last_sched_epoch = int(getattr(self.scheduler, "last_epoch", resumed_epoch))
            if last_sched_epoch < resumed_epoch:
                for _ in range(resumed_epoch - last_sched_epoch):
                    self.scheduler.step()
        elif not scheduler_stepped:
            self.scheduler.step()

    def load_resume_checkpoint(
        self,
        checkpoint_path: str | Path,
        preloaded_checkpoint: dict[str, Any] | None = None,
    ) -> int:
        """Restore full training state from a completed epoch checkpoint."""
        ckpt_path = Path(checkpoint_path)
        checkpoint = (
            preloaded_checkpoint
            if preloaded_checkpoint is not None
            else load_checkpoint_file(ckpt_path)
        )

        if "model_state_dict" not in checkpoint or "epoch" not in checkpoint:
            raise KeyError(
                f"Checkpoint at '{ckpt_path}' is missing 'model_state_dict' or 'epoch'."
            )

        resumed_epoch = int(checkpoint["epoch"])
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self._align_resumed_scheduler(checkpoint, resumed_epoch)

        self.best_score = float(checkpoint.get("best_score", self.best_score))
        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        self.best_metrics = dict(checkpoint.get("best_metrics", {}))
        self.epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        self.history = list(checkpoint.get("history", []))
        self.start_epoch = resumed_epoch + 1
        self.resumed_from = ckpt_path

        if (
            self.best_epoch > 0
            and resumed_epoch == self.best_epoch
            and not self.best_checkpoint_path.exists()
        ):
            atomic_torch_save(checkpoint, self.best_checkpoint_path)

        restore_rng_state(checkpoint.get("rng_state"), self.dataloader_generator)

        current_lr = float(self.optimizer.param_groups[0]["lr"])
        print(
            f"[Resume] Loaded '{ckpt_path}' (completed epoch={resumed_epoch}, "
            f"next_epoch={self.start_epoch}, best_epoch={self.best_epoch}, "
            f"best_score={self.best_score:.4f}, "
            f"epochs_without_improvement={self.epochs_without_improvement}, "
            f"lr={current_lr:.6g})",
            flush=True,
        )
        return self.start_epoch

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
        self.epochs_without_improvement = 0

        best_aliases = [f"epoch-{epoch}", "best"]
        if not self.upload_last_to_wandb:
            best_aliases.append("latest")

        self._save_checkpoint(
            path=self.best_checkpoint_path,
            epoch=epoch,
            metrics=metrics,
            kind="best",
            scheduler_stepped=False,
            upload_to_wandb=self.upload_best_to_wandb,
            aliases=best_aliases,
        )

        return True

    # Logging
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        val_per_class: dict[str, float] | None = None,
        train_per_class: dict[str, float] | None = None,
        qualitative_batch: dict[str, torch.Tensor] | None = None,
        per_class: dict[str, float] | None = None,
    ) -> None:

        if self.logger is None:
            return

        val_per_class_metrics = val_per_class if val_per_class is not None else per_class

        self.logger.log_epoch(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            per_class_metrics=val_per_class_metrics,
            train_per_class_metrics=train_per_class,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            qualitative_batch=qualitative_batch,
            class_labels=self.class_labels,
            qualitative_frequency=self.qualitative_freq,
        )

    # Training loop
    def fit(self) -> dict[str, Any]:
        if self.start_epoch > self.max_epochs:
            print(
                f"[Resume] Checkpoint already reached epoch {self.start_epoch - 1} "
                f"(>= max_epochs={self.max_epochs}). Skipping training loop.",
                flush=True,
            )
        else:
            for epoch in range(self.start_epoch, self.max_epochs + 1):

                epoch_start = time.perf_counter()
                should_stop = False

                # Train
                train_loss, train_metrics, train_per_class = self.train_epoch(epoch)
                train_metrics["loss"] = train_loss
                last_epoch_metrics = train_metrics

                # Validation
                if epoch % self.eval_freq == 0:
                    val_loss, val_metrics, per_class_results, qualitative_batch = self.validate(epoch)
                    val_metrics["loss"] = val_loss
                    last_epoch_metrics = val_metrics

                    self._log_epoch(
                        epoch=epoch,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        val_per_class=per_class_results,
                        train_per_class=train_per_class,
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
                        "train_per_class": train_per_class,
                        "per_class": per_class_results,
                        "val_per_class": per_class_results,
                        "epoch_time": epoch_time,
                    }
                    self.history.append(epoch_results)

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
                        should_stop = True

                else:
                    epoch_time = time.perf_counter() - epoch_start
                    epoch_results = {
                        "epoch": epoch,
                        "train": train_metrics,
                        "train_per_class": train_per_class,
                        "epoch_time": epoch_time,
                    }
                    self.history.append(epoch_results)

                if self.scheduler is not None:
                    self.scheduler.step()

                self._save_end_of_epoch_checkpoints(
                    epoch=epoch,
                    metrics=last_epoch_metrics,
                )

                if should_stop:
                    break

        best_ckpt_path = self.best_checkpoint_path
        if not best_ckpt_path.exists():
            if self.last_checkpoint_path.exists():
                best_ckpt_path = self.last_checkpoint_path
            elif self.resumed_from is not None and self.resumed_from.exists():
                best_ckpt_path = self.resumed_from

        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "best_checkpoint": str(best_ckpt_path),
            "last_checkpoint": str(self.last_checkpoint_path),
        }