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
        self.qualitative_freq = max(1, int(qualitative_cfg.get("frequency", self.eval_freq)))
        self.qualitative_max_samples = max(1, int(qualitative_cfg.get("max_samples", 4)))

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
        self.best_val_metrics: dict[str, float] = {}

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
            "train_time": [],
            "val_time": [],
            "epoch_time": [],
        }

        for metric_name in self.metric_names:
            self.history[f"val_{metric_name}"] = []

    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        if hasattr(self.model, "is_mask2former") and self.model.is_mask2former:
            output = self.model(x, y)
            return None, output.loss

        output = self.model(x)

        if isinstance(output, (list, tuple)):
            losses = []
            for logits in output:
                target = y.unsqueeze(1) if y.ndim == 3 else y
                losses.append(self.loss_fn(logits, target))
            
            loss = sum(losses) / len(losses)
            logits = output[0]
        else:
            logits = output
            if y.ndim == 3:
                y = y.unsqueeze(1)
            loss = self.loss_fn(logits, y)

        return logits, loss

    def _prepare_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch["image"].to(self.device, non_blocking=True)
        y = batch["mask"].to(self.device, non_blocking=True)

        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]
        y = y.long()

        return x, y

    def _autocast(self):
        enabled = self.scaler is not None and self.device.type == "cuda"
        return torch.amp.autocast("cuda", enabled=enabled)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        progress = tqdm(self.train_loader, desc=f"Train [{epoch + 1}/{self.max_epochs}]")
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
    def validate(self, epoch: int) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
        self.model.eval()
        total_images = 0
        total_model_time = 0.0
        total_loss = 0.0
        num_batches = 0
        self.metric_manager.reset()

        progress = tqdm(self.val_loader, desc=f"Val   [{epoch + 1}/{self.max_epochs}]")
        qualitative_batch = None
        collect_qualitative = self.qualitative_enabled and ((epoch + 1) % self.qualitative_freq == 0)

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

            self.metric_manager.update(logits, y)
            total_images += int(x.shape[0])
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if collect_qualitative and qualitative_batch is None and logits is not None:
                predictions = torch.argmax(logits, dim=1)
                qualitative_batch = {
                    "images": x[: self.qualitative_max_samples].detach().cpu(),
                    "labels": y[: self.qualitative_max_samples].detach().cpu(),
                    "predictions": predictions[: self.qualitative_max_samples].detach().cpu(),
                }

        results = self.metric_manager.compute()
        results["loss"] = total_loss / max(num_batches, 1)
        results["fps"] = total_images / max(total_model_time, 1e-8)
        results = {key: float(value) for key, value in results.items()}

        return results, qualitative_batch

    def _save_checkpoint(self, epoch: int) -> str:
        output_dir = Path(self.cfg.paths.checkpoint)
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

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict[str, float],
        train_time: float,
        val_time: float,
        epoch_time: float,
        qualitative_batch: dict[str, torch.Tensor] | None,
    ) -> None:
        if self.logger is None:
            return

        wandb_step = epoch + 1

        extra_metrics = {
            "epoch/train_batches": len(self.train_loader),
            "epoch/val_batches": len(self.val_loader),
            "time/train_seconds": train_time,
            "time/val_seconds": val_time,
            "time/epoch_seconds": epoch_time,
            "training/best_score": self.best_score if np.isfinite(self.best_score) else float("nan"),
            "training/best_epoch": self.best_epoch + 1 if self.best_epoch >= 0 else 0,
            "training/epochs_without_improvement": self.epochs_without_improvement,
        }

        if self.device.type == "cuda":
            extra_metrics["system/gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            extra_metrics["system/gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            extra_metrics["system/gpu_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2

        self.logger.log_metrics(extra_metrics, step=wandb_step, commit=False)

        if qualitative_batch is not None:
            num_classes = int(self.cfg.data.num_classes)
            class_labels = {index: f"class_{index}" for index in range(num_classes)}
            configured_labels = self.cfg.data.get("class_labels", None)

            if configured_labels is not None:
                if isinstance(configured_labels, dict):
                    class_labels = {int(key): str(value) for key, value in configured_labels.items()}
                elif isinstance(configured_labels, (list, tuple)):
                    class_labels = {index: str(name) for index, name in enumerate(configured_labels)}

            self.logger.log_qualitative_masks(
                images=qualitative_batch["images"],
                labels=qualitative_batch["labels"],
                predictions=qualitative_batch["predictions"],
                class_labels=class_labels,
                scenario="clean",
                epoch=epoch,
                max_samples=self.qualitative_max_samples,
            )

        # 3. Loss, LR e metriche principali (commit=True: chiude lo step)
        log_metrics = {"fps": val_metrics["fps"]}
        for metric_name in self.metric_names:
            log_metrics[metric_name] = val_metrics.get(metric_name, float("nan"))

        self.logger.log_epoch_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_metrics["loss"],
            learning_rate=self.optimizer.param_groups[0]["lr"],
            metrics=log_metrics,
            step=wandb_step,
        )

    def fit(self) -> dict[str, Any]:
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        training_start = time.perf_counter()

        for epoch in range(self.max_epochs):
            epoch_start = time.perf_counter()
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")

            train_start = time.perf_counter()
            train_loss = self.train_epoch(epoch)
            train_time = time.perf_counter() - train_start

            if self.scheduler is not None:
                self.scheduler.step()

            should_validate = (epoch + 1) % self.eval_freq == 0 or (epoch + 1) == self.max_epochs

            if not should_validate:
                print(f"Train Loss : {train_loss:.4f}")
                continue

            val_start = time.perf_counter()
            val_metrics, qualitative_batch = self.validate(epoch)
            val_time = time.perf_counter() - val_start
            epoch_time = time.perf_counter() - epoch_start

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_fps"].append(val_metrics["fps"])
            self.history["train_time"].append(train_time)
            self.history["val_time"].append(val_time)
            self.history["epoch_time"].append(epoch_time)

            for metric_name in self.metric_names:
                self.history[f"val_{metric_name}"].append(val_metrics.get(metric_name, float("nan")))

            improved = self._update_best(epoch, val_metrics)

            if improved:
                print(f"✓ New best {self.monitor}: {self.best_score:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement ({self.epochs_without_improvement}/{self.patience})")

            self._log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_metrics=val_metrics,
                train_time=train_time,
                val_time=val_time,
                epoch_time=epoch_time,
                qualitative_batch=qualitative_batch,
            )

            print(f"Train Loss : {train_loss:.4f}")
            print(f"Val Loss   : {val_metrics['loss']:.4f}")
            print(f"Val FPS    : {val_metrics['fps']:.2f}")
            print(f"Train Time : {train_time:.2f}s")
            print(f"Val Time   : {val_time:.2f}s")
            print(f"Epoch Time : {epoch_time:.2f}s")

            for metric_name in self.metric_names:
                value = val_metrics.get(metric_name, float("nan"))
                label = f"Val {metric_name.upper():<8}"
                if np.isnan(value):
                    print(f"{label}: N/A")
                else:
                    print(f"{label}: {value:.4f}")

            if self.early_stopping_enabled and self.epochs_without_improvement >= self.patience:
                print("\nEarly stopping.")
                break

        total_training_time = time.perf_counter() - training_start

        if self.best_checkpoint is None:
            raise RuntimeError("Training finished without creating a valid checkpoint.")

        if self.logger is not None:
            self.logger.log_metrics(
                {
                    "training/total_time_seconds": total_training_time,
                    "training/completed_epochs": len(self.history["train_loss"]),
                    "training/best_epoch": self.best_epoch + 1,
                    "training/best_score": self.best_score,
                },
                step=self.best_epoch + 1 if self.best_epoch >= 0 else 1,
                commit=True,
            )

        print("\n" + "=" * 60)
        print("TRAINING FINISHED")
        print("=" * 60)
        print(f"Best epoch : {self.best_epoch + 1}")
        print(f"Best {self.monitor}: {self.best_score:.4f}")
        print(f"Checkpoint : {self.best_checkpoint}")
        print(f"Total time : {total_training_time:.2f}s")
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