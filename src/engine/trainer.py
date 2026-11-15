from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from omegaconf import OmegaConf

from src.utils.qualitative import QualitativeSampler
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
        # HD95 is slow: during training it is computed every N validations (0 = never).
        # The final evaluation (start_experiment) always computes it.
        self.hd95_freq = max(0, int((cfg.get("evaluation") or {}).get("hd95_freq", 10)))

        logging_cfg = cfg.get("logging", {})
        qualitative_cfg = logging_cfg.get("qualitative", {})
        self.qualitative_enabled = bool(qualitative_cfg.get("enabled", True))
        self.qualitative_freq = max(1, int(qualitative_cfg.get("frequency", 1)))
        self.qualitative_cfg = qualitative_cfg
        self.num_classes = int(getattr(self.metric_manager, "num_classes", len(self.class_labels)))
        self.seed = int((cfg.get("experiment") or {}).get("seed", 0))

        early_stopping_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = bool(early_stopping_cfg.get("enabled", True))
        self.patience = int(early_stopping_cfg.get("patience", 10))
        self.monitor = str(early_stopping_cfg.get("monitor", "dice_acc"))
        self.monitor_mode = str(early_stopping_cfg.get("mode", "max")).lower()
        # An epoch counts as an improvement only if it beats the best score by MORE than min_delta
        # (same convention as Keras): smaller gains are noise and neither reset the patience counter
        # nor replace the best checkpoint.
        self.min_delta = float(early_stopping_cfg.get("min_delta", 0.0))

        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("early_stopping.mode must be either 'min' or 'max'")
        if self.min_delta < 0:
            raise ValueError("early_stopping.min_delta must be >= 0")

        self.metric_names = list(self.metric_manager.names)
        available_metrics = set(self.metric_names) | {"loss", "fps"}

        if self.monitor not in available_metrics:
            raise ValueError(
                f"Early stopping monitor '{self.monitor}' is not available. "
                f"Available metrics: {sorted(available_metrics)}"
            )

        if self.logger is not None and hasattr(self.logger, "set_monitor"):
            self.logger.set_monitor(self.monitor, self.monitor_mode)

        self.best_score = float("inf") if self.monitor_mode == "min" else -float("inf")
        self.best_epoch = 0
        self.best_metrics: dict[str, float] = {}
        self.best_per_class: dict[str, float] = {}
        self.last_epoch = 0
        self.last_metrics: dict[str, float] = {}
        self.last_per_class: dict[str, float] = {}
        self.stopped_early = False
        # qualitative samples kept in memory as (epoch, batch) to log best/last at the end
        self._best_qualitative: tuple[int, dict[str, Any]] | None = None
        self._last_qualitative: tuple[int, dict[str, Any]] | None = None
        self.epochs_without_improvement = 0
        self.start_epoch = 1
        self.history: list[dict[str, Any]] = []
        self.resumed_from: Path | None = None

        ckpt_cfg = cfg.get("checkpoint") or train_cfg.get("checkpoint") or {}
        self.save_last = bool(ckpt_cfg.get("save_last", True))
        save_every = ckpt_cfg.get("save_every_n_epochs", None)
        self.save_every_n_epochs = int(save_every) if save_every is not None else None
        # W&B: the weights of the best and of the last epoch are uploaded ONCE, at the end of the
        # training (weights only, no optimizer state). Nothing is uploaded during the training.
        self.upload_best_to_wandb = bool(ckpt_cfg.get("upload_best_to_wandb", True))
        self.upload_last_to_wandb = bool(ckpt_cfg.get("upload_last_to_wandb", True))

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
        # weights-only copies (small, for inference); overwritten at the end of every training
        self.best_weights_path = self.best_checkpoint_path.with_name(f"weights_{self.best_checkpoint_path.name}")
        self.last_weights_path = self.last_checkpoint_path.with_name(f"weights_{self.last_checkpoint_path.name}")
        artifact_name = f"{self.cfg.data.name}_{self.cfg.model.name}_lr_{self.initial_lr_str}"
        split_cfg = (cfg.get("data") or {}).get("split") or {}
        if split_cfg.get("strategy") == "kfold":
            artifact_name += f"_fold{int(split_cfg.get('fold_index', 0))}"
        experiment_cfg = cfg.get("experiment") or {}
        if experiment_cfg.get("seed_runs"):
            artifact_name += f"_seed{int(experiment_cfg.get('seed', 0))}"
        self.artifact_base_name = sanitize_artifact_name(artifact_name)

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

            # Average over the real size of the accumulation group: the last group
            # of the epoch can be smaller than accumulation_steps.
            total_batches = len(self.train_loader)
            group_start = (step // self.accumulation_steps) * self.accumulation_steps
            group_size = min(self.accumulation_steps, total_batches - group_start)
            loss_for_backward = loss / group_size

            if self.scaler is not None:
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == total_batches:
                self._optimizer_step()

            # HD95 is too slow to run every training step: validation/test only
            self.metric_manager.update(predictions.detach(), masks.detach(), extra=False)

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
        self,
        epoch: int,
        extra: bool | None = None,
        loader: torch.utils.data.DataLoader | None = None,
    ) -> tuple[float, dict[str, float], dict[str, float], dict[str, torch.Tensor] | None]:
        """Evaluation pass on ``loader`` (default: the validation loader).

        ``extra`` enables the expensive metrics (HD95); by default they run every
        ``hd95_freq`` epochs."""
        loader = self.val_loader if loader is None else loader

        if extra is None:
            extra = self.hd95_freq > 0 and epoch % self.hd95_freq == 0

        self.model.eval()
        self.metric_manager.reset()

        running_loss = 0.0
        num_batches = 0

        sampler = None
        if self.qualitative_enabled:
            sampler = QualitativeSampler.from_config(
                self.qualitative_cfg, self.num_classes, seed=self.seed
            )

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch} [Val]",
            leave=False,
        )

        for batch in progress:
            images, masks = self._prepare_batch(batch)

            with self._autocast():
                predictions, loss = self._forward(images, masks)

                if loss is None:
                    if self.loss_fn is None:
                        raise RuntimeError("Model returned no loss and no loss function is configured.")
                    loss = self.loss_fn(predictions, masks)

            self.metric_manager.update(predictions.detach(), masks.detach(), extra=extra)

            running_loss += loss.detach().item()
            num_batches += 1

            if sampler is not None:
                sampler.update(images, masks, predictions.detach())

            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = self.metric_manager.compute()
        loss_value = running_loss / max(num_batches, 1)
        per_class = self.metric_manager.compute_per_class()

        qualitative_batch = sampler.result() if sampler is not None else None
        if qualitative_batch is not None:
            # same matrix the accumulated Dice/IoU are computed from
            qualitative_batch["confusion_matrix"] = self.metric_manager.confusion_matrix().numpy()

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
                upload_to_wandb=False,
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
                upload_to_wandb=False,
                aliases=[f"epoch-{epoch}", "periodic"],
            )

    def _weights_payload(
        self, kind: str, epoch: int, state_dict: dict[str, Any], metrics: dict[str, float]
    ) -> dict[str, Any]:
        """Weights-only checkpoint: enough to rebuild the model and run inference."""
        try:
            config = OmegaConf.to_container(self.cfg, resolve=True) if OmegaConf.is_config(self.cfg) else None
        except Exception:  # unresolved interpolations outside a Hydra run
            config = OmegaConf.to_container(self.cfg, resolve=False)
        return {
            "kind": kind,
            "epoch": int(epoch),
            "model_state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
            "metrics": dict(metrics or {}),
            "best_epoch": int(self.best_epoch),
            "best_score": float(self.best_score) if np.isfinite(self.best_score) else None,
            "config": config,
        }

    def _save_final_weights(self) -> None:
        """Save, overwriting, the weights of the BEST and of the LAST epoch (weights only).

        Local files: ``weights_best_*.pt`` and ``weights_last_*.pt`` next to the full checkpoints.
        W&B: two model artifacts, ``<name>_best`` and ``<name>_last`` (alias ``best`` / ``last`` and
        ``latest``), uploaded once, only if ``checkpoint.upload_best_to_wandb`` /
        ``upload_last_to_wandb`` are true. A new training of the same run adds a new artifact
        version and moves the aliases; W&B cannot overwrite a version.
        """
        entries: list[tuple[str, int, dict[str, Any], dict[str, float], Path, bool]] = []

        # LAST: the weights the model holds now (the stopping epoch with early stopping)
        last_epoch = self.last_epoch if self.last_epoch > 0 else max(self.start_epoch - 1, 0)
        entries.append(("last", last_epoch, self.model.state_dict(), self.last_metrics,
                        self.last_weights_path, self.upload_last_to_wandb))

        # BEST: from the best checkpoint saved at the best epoch
        if self.best_checkpoint_path.is_file():
            best = load_checkpoint_file(self.best_checkpoint_path)
            entries.append(("best", int(best.get("epoch", self.best_epoch)), best["model_state_dict"],
                            best.get("metrics") or self.best_metrics, self.best_weights_path,
                            self.upload_best_to_wandb))
        else:
            print("[Weights] No best checkpoint found: only the last weights are saved.", flush=True)

        for kind, epoch, state_dict, metrics, path, upload in entries:
            payload = self._weights_payload(kind, epoch, state_dict, metrics)
            if self.logger is not None:
                self.logger.log_checkpoint(
                    checkpoint=payload,
                    checkpoint_path=path,
                    name=f"{self.artifact_base_name}_{kind}",
                    aliases=[kind, "latest", f"epoch-{epoch}"],
                    upload_to_wandb=upload,
                    metadata={
                        "kind": kind,
                        "epoch": int(epoch),
                        "weights_only": True,
                        "dataset": str(self.cfg.data.name),
                        "model": str(self.cfg.model.name),
                        "lr": self.initial_lr_str,
                    },
                )
            else:
                atomic_torch_save(payload, path)
            uploaded = bool(upload and self.logger is not None and getattr(self.logger, "is_active", False))
            print(
                f"[Weights] {kind} (epoch {epoch}) saved to {path}"
                + (f" and uploaded to W&B as '{self.artifact_base_name}_{kind}'" if uploaded else ""),
                flush=True,
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
        # best_score starts at +-inf: the first finite score always counts
        if not np.isfinite(self.best_score):
            return True
        if self.monitor_mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

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

        # Full checkpoint (for the resume), overwritten at every improvement. Local only: the weights
        # are uploaded to W&B once, at the end of the training (see _save_final_weights).
        self._save_checkpoint(
            path=self.best_checkpoint_path,
            epoch=epoch,
            metrics=metrics,
            kind="best",
            scheduler_stepped=False,
            upload_to_wandb=False,
            aliases=[f"epoch-{epoch}", "best"],
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

                    self.last_epoch = epoch
                    self.last_metrics = dict(val_metrics)
                    self.last_per_class = dict(per_class_results)
                    if qualitative_batch is not None:
                        self._last_qualitative = (epoch, qualitative_batch)
                        if improved:
                            self._best_qualitative = (epoch, qualitative_batch)
                    if improved:
                        self.best_per_class = dict(per_class_results)

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
                        self.stopped_early = True

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

        self._save_final_weights()

        if self.logger is not None and hasattr(self.logger, "log_final_qualitative"):
            self.logger.log_final_qualitative(
                best=self._best_qualitative,
                last=self._last_qualitative,
                class_labels=self.class_labels,
            )

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
            "best_per_class": self.best_per_class,
            "last_epoch": self.last_epoch,
            "last_metrics": self.last_metrics,
            "last_per_class": self.last_per_class,
            "stopped_early": self.stopped_early,
            "max_epochs": self.max_epochs,
            "best_checkpoint": str(best_ckpt_path),
            "last_checkpoint": str(self.last_checkpoint_path),
        }