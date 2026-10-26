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

        self.cfg = cfg
        self.device = device
        self.logger = logger

        # =====================================================
        # Training configuration
        # =====================================================

        train_cfg = cfg.train

        self.max_epochs = train_cfg.max_epochs

        self.accumulation_steps = train_cfg.get(
            "accumulation_steps",
            1,
        )

        self.eval_freq = train_cfg.get(
            "eval_freq",
            1,
        )

        # =====================================================
        # Early stopping
        # =====================================================

        early_stopping_cfg = train_cfg.get(
            "early_stopping",
            {},
        )

        self.early_stopping_enabled = early_stopping_cfg.get(
            "enabled",
            True,
        )

        self.patience = early_stopping_cfg.get(
            "patience",
            10,
        )

        self.monitor = early_stopping_cfg.get(
            "monitor",
            "dice",
        )

        self.monitor_mode = early_stopping_cfg.get(
            "mode",
            "max",
        )

        if self.monitor_mode == "max":
            self.best_score = -float("inf")
        else:
            self.best_score = float("inf")

        self.best_epoch = -1
        self.best_checkpoint: str | None = None
        self.epochs_without_improvement = 0

        # =====================================================
        # Model information
        # =====================================================

        self.num_classes = int(
            cfg.data.num_classes
        )

        # =====================================================
        # History
        # =====================================================

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_fps": [],
        }

    # =========================================================
    # Forward
    # =========================================================

    def _forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        output = self.model(x)

        # -----------------------------------------------------
        # Deep supervision / multiple outputs
        # -----------------------------------------------------

        if isinstance(output, (list, tuple)):

            losses = [
                self.loss_fn(logits, y)
                for logits in output
            ]

            loss = sum(losses) / len(losses)

            logits = output[0]

        else:

            logits = output

            loss = self.loss_fn(
                logits,
                y,
            )

        return logits, loss

    # =========================================================
    # Batch preparation
    # =========================================================

    def _prepare_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x = batch["image"].to(
            self.device,
            non_blocking=True,
        )

        y = batch["mask"].to(
            self.device,
            non_blocking=True,
        )

        # -----------------------------------------------------
        # Semantic segmentation masks
        #
        # [B, 1, H, W] -> [B, H, W]
        # -----------------------------------------------------

        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]

        # CE / multiclass losses expect integer labels
        y = y.long()

        return x, y

    # =========================================================
    # AMP
    # =========================================================

    def _autocast(self):

        return torch.cuda.amp.autocast(
            enabled=(
                self.scaler is not None
                and self.device.type == "cuda"
            )
        )

    # =========================================================
    # Train epoch
    # =========================================================

    def train_epoch(
        self,
        epoch: int,
    ) -> float:

        self.model.train()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        losses: list[float] = []

        progress = tqdm(
            self.train_loader,
            desc=(
                f"Train "
                f"[{epoch + 1}/{self.max_epochs}]"
            ),
        )

        for step, batch in enumerate(progress):

            x, y = self._prepare_batch(batch)

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            with self._autocast():

                _, loss = self._forward(
                    x,
                    y,
                )

            real_loss = loss.detach().item()

            losses.append(real_loss)

            # -------------------------------------------------
            # Gradient accumulation
            # -------------------------------------------------

            scaled_loss = (
                loss / self.accumulation_steps
            )

            if self.scaler is not None:

                self.scaler.scale(
                    scaled_loss
                ).backward()

            else:

                scaled_loss.backward()

            # -------------------------------------------------
            # Optimizer step
            # -------------------------------------------------

            is_update_step = (
                (step + 1)
                % self.accumulation_steps
                == 0
                or
                (step + 1)
                == len(self.train_loader)
            )

            if is_update_step:

                if self.scaler is not None:

                    self.scaler.step(
                        self.optimizer
                    )

                    self.scaler.update()

                else:

                    self.optimizer.step()

                self.optimizer.zero_grad(
                    set_to_none=True
                )

            progress.set_postfix(
                loss=f"{real_loss:.4f}"
            )

        return float(
            np.mean(losses)
        )

    # =========================================================
    # Validation
    # =========================================================

    @torch.inference_mode()
    def validate(
        self,
        epoch: int,
    ) -> dict[str, float]:

        self.model.eval()

        val_losses: list[float] = []

        total_images = 0
        total_model_time = 0.0

        dice_values: list[float] = []
        iou_values: list[float] = []

        progress = tqdm(
            self.val_loader,
            desc=(
                f"Val   "
                f"[{epoch + 1}/{self.max_epochs}]"
            ),
        )

        for batch in progress:

            x, y = self._prepare_batch(batch)

            # -------------------------------------------------
            # Inference timing
            # -------------------------------------------------

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            with self._autocast():

                logits, loss = self._forward(
                    x,
                    y,
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            total_model_time += (
                time.perf_counter()
                - start_time
            )

            val_losses.append(
                loss.detach().item()
            )

            # -------------------------------------------------
            # Multiclass prediction
            # -------------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            batch_metrics = self._compute_metrics(
                predictions,
                y,
            )

            dice_values.append(
                batch_metrics["dice"]
            )

            iou_values.append(
                batch_metrics["iou"]
            )

            total_images += x.shape[0]

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{batch_metrics['dice']:.4f}",
            )

        # =====================================================
        # Aggregate validation results
        # =====================================================

        results = {
            "loss": float(
                np.mean(val_losses)
            ),

            "dice": self._safe_mean(
                dice_values
            ),

            "iou": self._safe_mean(
                iou_values
            ),

            "fps": (
                total_images
                / max(
                    total_model_time,
                    1e-8,
                )
            ),
        }

        return results

    # =========================================================
    # Multiclass metrics
    # =========================================================

    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:

        predictions = predictions.long()
        labels = labels.long()

        dice_values = []
        iou_values = []

        # -----------------------------------------------------
        # Per-class Dice / IoU
        # -----------------------------------------------------

        for class_id in range(
            self.num_classes
        ):

            pred_class = (
                predictions == class_id
            )

            label_class = (
                labels == class_id
            )

            intersection = (
                pred_class & label_class
            ).sum(dim=(1, 2)).float()

            pred_area = pred_class.sum(
                dim=(1, 2)
            ).float()

            label_area = label_class.sum(
                dim=(1, 2)
            ).float()

            union = (
                pred_area
                + label_area
                - intersection
            )

            # -------------------------------------------------
            # Dice
            # -------------------------------------------------

            dice = (
                2.0 * intersection + 1e-8
            ) / (
                pred_area
                + label_area
                + 1e-8
            )

            # -------------------------------------------------
            # IoU
            # -------------------------------------------------

            iou = (
                intersection + 1e-8
            ) / (
                union + 1e-8
            )

            # -------------------------------------------------
            # Ignore absent classes
            #
            # If both prediction and GT are empty,
            # class is not considered for this sample.
            # -------------------------------------------------

            valid = (
                (pred_area + label_area) > 0
            )

            if valid.any():

                dice_values.append(
                    dice[valid].mean().item()
                )

                iou_values.append(
                    iou[valid].mean().item()
                )

        # -----------------------------------------------------
        # Macro average
        # -----------------------------------------------------

        return {
            "dice": (
                float(np.mean(dice_values))
                if dice_values
                else 0.0
            ),

            "iou": (
                float(np.mean(iou_values))
                if iou_values
                else 0.0
            ),
        }

    # =========================================================
    # Checkpoint
    # =========================================================

    def _save_checkpoint(
        self,
        epoch: int,
    ) -> str:

        output_dir = Path(
            self.cfg.paths.output
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = (
            output_dir / "best.pt"
        )

        checkpoint = {
            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),

            "best_score":
                self.best_score,

            "best_epoch":
                self.best_epoch,

            "config":
                self.cfg,
        }

        # -----------------------------------------------------
        # Atomic save
        # -----------------------------------------------------

        temporary_path = (
            checkpoint_path.with_suffix(
                ".tmp"
            )
        )

        torch.save(
            checkpoint,
            temporary_path,
        )

        temporary_path.replace(
            checkpoint_path
        )

        # -----------------------------------------------------
        # W&B
        # -----------------------------------------------------

        if self.logger is not None:

            self.logger.log_checkpoint(
                checkpoint_path=str(
                    checkpoint_path
                ),
                name=(
                    f"{self.cfg.data.name}-"
                    f"{self.cfg.model.name}"
                ),
                aliases=["best"],
            )

        return str(
            checkpoint_path
        )

    # =========================================================
    # Improvement
    # =========================================================

    def _is_improvement(
        self,
        score: float,
    ) -> bool:

        if self.monitor_mode == "max":

            return score > self.best_score

        return score < self.best_score

    # =========================================================
    # Update best
    # =========================================================

    def _update_best(
        self,
        epoch: int,
        val_metrics: dict[str, float],
    ) -> bool:

        score = val_metrics[
            self.monitor
        ]

        if not self._is_improvement(
            score
        ):
            return False

        self.best_score = score

        self.best_epoch = epoch

        self.epochs_without_improvement = 0

        self.best_checkpoint = (
            self._save_checkpoint(epoch)
        )

        return True

    # =========================================================
    # Fit
    # =========================================================

    def fit(self) -> dict[str, Any]:

        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        for epoch in range(
            self.max_epochs
        ):

            print(
                f"\nEpoch "
                f"{epoch + 1}/"
                f"{self.max_epochs}"
            )

            # -------------------------------------------------
            # Train
            # -------------------------------------------------

            train_loss = (
                self.train_epoch(epoch)
            )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            should_validate = (
                (epoch + 1)
                % self.eval_freq
                == 0
                or
                (epoch + 1)
                == self.max_epochs
            )

            if not should_validate:
                continue

            val_metrics = self.validate(
                epoch
            )

            # -------------------------------------------------
            # Scheduler
            # -------------------------------------------------

            if self.scheduler is not None:

                self.scheduler.step()

            # -------------------------------------------------
            # History
            # -------------------------------------------------

            self.history[
                "train_loss"
            ].append(train_loss)

            self.history[
                "val_loss"
            ].append(
                val_metrics["loss"]
            )

            self.history[
                "val_dice"
            ].append(
                val_metrics["dice"]
            )

            self.history[
                "val_iou"
            ].append(
                val_metrics["iou"]
            )

            self.history[
                "val_fps"
            ].append(
                val_metrics["fps"]
            )

            # -------------------------------------------------
            # Learning rate
            # -------------------------------------------------

            learning_rate = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------

            if self.logger is not None:

                self.logger.log_epoch_metrics(
                    epoch=epoch,

                    train_loss=train_loss,

                    val_loss=val_metrics[
                        "loss"
                    ],

                    learning_rate=learning_rate,

                    metrics={
                        "dice":
                            val_metrics["dice"],

                        "iou":
                            val_metrics["iou"],

                        "fps":
                            val_metrics["fps"],
                    },
                )

            # -------------------------------------------------
            # Console
            # -------------------------------------------------

            print(
                f"Train Loss : "
                f"{train_loss:.4f}"
            )

            print(
                f"Val Loss   : "
                f"{val_metrics['loss']:.4f}"
            )

            print(
                f"Val Dice   : "
                f"{val_metrics['dice']:.4f}"
            )

            print(
                f"Val IoU    : "
                f"{val_metrics['iou']:.4f}"
            )

            print(
                f"Val FPS    : "
                f"{val_metrics['fps']:.2f}"
            )

            # -------------------------------------------------
            # Best checkpoint
            # -------------------------------------------------

            improved = self._update_best(
                epoch,
                val_metrics,
            )

            if improved:

                print(
                    f"✓ New best "
                    f"{self.monitor}: "
                    f"{self.best_score:.4f}"
                )

            else:

                self.epochs_without_improvement += 1

                print(
                    "No improvement "
                    f"("
                    f"{self.epochs_without_improvement}"
                    f"/"
                    f"{self.patience}"
                    f")"
                )

            # -------------------------------------------------
            # Early stopping
            # -------------------------------------------------

            if (
                self.early_stopping_enabled
                and
                self.epochs_without_improvement
                >= self.patience
            ):

                print(
                    "\nEarly stopping."
                )

                break

        # =====================================================
        # Training finished
        # =====================================================

        if self.best_checkpoint is None:

            raise RuntimeError(
                "Training finished without "
                "creating a valid checkpoint."
            )

        print("\n" + "=" * 60)
        print("TRAINING FINISHED")
        print("=" * 60)

        print(
            f"Best epoch : "
            f"{self.best_epoch + 1}"
        )

        print(
            f"Best {self.monitor}: "
            f"{self.best_score:.4f}"
        )

        print(
            f"Checkpoint : "
            f"{self.best_checkpoint}"
        )

        print("=" * 60)

        return {
            "best_checkpoint":
                self.best_checkpoint,

            "best_epoch":
                self.best_epoch,

            "best_val_dice":
                self.history[
                    "val_dice"
                ][self.best_epoch],

            "best_val_iou":
                self.history[
                    "val_iou"
                ][self.best_epoch],

            "history":
                self.history,
        }

    # =========================================================
    # Utility
    # =========================================================

    @staticmethod
    def _safe_mean(
        values: list[float],
    ) -> float:

        if not values:
            return float("nan")

        return float(
            np.mean(values)
        )