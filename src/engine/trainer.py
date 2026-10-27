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

        # Training configuration
        train_cfg = cfg.train
        self.max_epochs = int(train_cfg.get("max_epochs", 50))
        self.accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
        self.eval_freq = max(1, int(train_cfg.get("eval_freq", 1)))

        # Early stopping
        early_stopping_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = bool(early_stopping_cfg.get("enabled", True))
        self.patience = int(early_stopping_cfg.get("patience", 10))
        self.monitor = str(early_stopping_cfg.get("monitor", "dice"))
        self.monitor_mode = str(early_stopping_cfg.get("mode", "max")).lower()

        if self.monitor_mode not in {"max", "min"}:
            raise ValueError("Early stopping mode must be 'max' or 'min'.")

        self.best_score = -float("inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch = -1
        self.best_checkpoint: str | None = None
        self.epochs_without_improvement = 0

        # Metrics corresponding to the best checkpoint
        self.best_val_metrics: dict[str, float] = {}

        # Metric names
        self.metric_names = list(self.metric_manager.names)
        available_metrics = set(self.metric_names) | {"loss", "fps"}

        if self.monitor not in available_metrics:
            raise ValueError(
                f"Early-stopping monitor '{self.monitor}' is not available. "
                f"Available metrics: {sorted(available_metrics)}"
            )

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_fps": [],
        }

        for metric_name in self.metric_names:
            self.history[f"val_{metric_name}"] = []

    def _forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(x)

        # Models with deep supervision
        if isinstance(output, (list, tuple)):
            losses = [self.loss_fn(logits, y) for logits in output]
            loss = sum(losses) / len(losses)
            # First output is used for evaluation
            logits = output[0]
        else:
            logits = output
            loss = self.loss_fn(logits, y)

        return logits, loss

    def _prepare_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch["image"].to(self.device, non_blocking=True)
        y = batch["mask"].to(self.device, non_blocking=True)

        # [B, 1, H, W] -> [B, H, W]
        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]

        # Multiclass segmentation targets
        y = y.long()
        return x, y

    def _autocast(self):
        enabled = self.scaler is not None and self.device.type == "cuda"
        return torch.cuda.amp.autocast(enabled=enabled)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        
        progress = tqdm(
            self.train_loader,
            desc=f"Train [{epoch + 1}/{self.max_epochs}]",
        )
        num_steps = len(self.train_loader)

        for step, batch in enumerate(progress):
            x, y = self._prepare_batch(batch)

            with self._autocast():
                _, loss = self._forward(x, y)

            real_loss = float(loss.detach().item())
            losses.append(real_loss)
            scaled_loss = loss / self.accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_update_step = (step + 1) % self.accumulation_steps == 0 or (step + 1) == num_steps

            if is_update_step:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{real_loss:.4f}")

        return self._safe_mean(losses)

    @torch.inference_mode()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        total_images = 0
        total_model_time = 0.0
        total_loss = 0.0
        num_batches = 0

        self.metric_manager.reset()
        progress = tqdm(
            self.val_loader,
            desc=f"Val   [{epoch + 1}/{self.max_epochs}]",
        )

        for batch in progress:
            x, y = self._prepare_batch(batch)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            with self._autocast():
                logits, loss = self._forward(x, y)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            total_model_time += time.perf_counter() - start_time
            total_loss += float(loss.detach().item())
            num_batches += 1

            # MetricManager handles specific computations (e.g., one-hot encoding, MONAI)
            self.metric_manager.update(logits, y)
            total_images += int(x.shape[0])
            progress.set_postfix(loss=f"{loss.item():.4f}")

        results = self.metric_manager.compute()
        results["loss"] = total_loss / max(num_batches, 1)
        results["fps"] = total_images / max(total_model_time, 1e-8)

        return {key: float(value) for key, value in results.items()}

    def _save_checkpoint(self, epoch: int) -> str:
        output_dir = Path(self.cfg.paths.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "best.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_val_metrics": self.best_val_metrics,
            "config": self.cfg,
        }

        temporary_path = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)

        if self.logger is not None:
            self.logger.log_checkpoint(
                checkpoint_path=str(checkpoint_path),
                name=f"{self.cfg.data.name}-{self.cfg.model.name}",
                aliases=["best"],
            )

        return str(checkpoint_path)

    def _is_improvement(self, score: float) -> bool:
        if np.isnan(score):
            return False
        if self.monitor_mode == "max":
            return score > self.best_score
        return score < self.best_score

    def _update_best(self, epoch: int, val_metrics: dict[str, float]) -> bool:
        score = val_metrics.get(self.monitor, float("nan"))
        if not self._is_improvement(score):
            return False

        self.best_score = score
        self.best_epoch = epoch
        self.epochs_without_improvement = 0
        self.best_val_metrics = dict(val_metrics)
        self.best_checkpoint = self._save_checkpoint(epoch)
        return True

    def fit(self) -> dict[str, Any]:
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            
            train_loss = self.train_epoch(epoch)

            should_validate = (epoch + 1) % self.eval_freq == 0 or (epoch + 1) == self.max_epochs
            if not should_validate:
                continue

            val_metrics = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_fps"].append(val_metrics["fps"])

            for metric_name in self.metric_names:
                self.history[f"val_{metric_name}"].append(
                    val_metrics.get(metric_name, float("nan"))
                )

            learning_rate = self.optimizer.param_groups[0]["lr"]

            if self.logger is not None:
                log_metrics = {"fps": val_metrics["fps"]}
                for metric_name in self.metric_names:
                    log_metrics[metric_name] = val_metrics.get(metric_name, float("nan"))

                self.logger.log_epoch_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_metrics["loss"],
                    learning_rate=learning_rate,
                    metrics=log_metrics,
                )

            print(f"Train Loss : {train_loss:.4f}")
            print(f"Val Loss   : {val_metrics['loss']:.4f}")
            print(f"Val FPS    : {val_metrics['fps']:.2f}")

            for metric_name in self.metric_names:
                value = val_metrics.get(metric_name, float("nan"))
                label = f"Val {metric_name.upper():<8}"
                if np.isnan(value):
                    print(f"{label}: N/A")
                else:
                    print(f"{label}: {value:.4f}")

            improved = self._update_best(epoch, val_metrics)

            if improved:
                print(f"✓ New best {self.monitor}: {self.best_score:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement ({self.epochs_without_improvement}/{self.patience})")

            if self.early_stopping_enabled and self.epochs_without_improvement >= self.patience:
                print("\nEarly stopping.")
                break

        if self.best_checkpoint is None:
            raise RuntimeError("Training finished without creating a valid checkpoint.")

        print("\n" + "=" * 60)
        print("TRAINING FINISHED")
        print("=" * 60)
        print(f"Best epoch : {self.best_epoch + 1}")
        print(f"Best {self.monitor}: {self.best_score:.4f}")
        print(f"Checkpoint : {self.best_checkpoint}")
        print("=" * 60)

        return {
            "best_checkpoint": self.best_checkpoint,
            "best_epoch": self.best_epoch,
            "best_val_metrics": self.best_val_metrics,
            "history": self.history,
        }

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        if not values:
            return float("nan")
        return float(np.mean(values))