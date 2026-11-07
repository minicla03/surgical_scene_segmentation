from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


class Tester:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        metric_manager: Any,
        loss_fn: torch.nn.Module,
        cfg: Any,
        device: torch.device,
        logger: Any | None = None,
    ) -> None:
        self.model = model.to(device)

        self.test_loader = test_loader
        self.metric_manager = metric_manager
        self.loss_fn = loss_fn

        self.cfg = cfg
        self.device = device
        self.logger = logger

        logging_cfg = cfg.get("logging", {})
        qualitative_cfg = logging_cfg.get(
            "qualitative",
            {},
        )

        self.qualitative_enabled = bool(
            qualitative_cfg.get("enabled", True)
        )

        self.qualitative_max_samples = max(
            1,
            int(
                qualitative_cfg.get(
                    "max_samples",
                    4,
                )
            ),
        )

        self.metric_names = list(
            self.metric_manager.names
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass used during testing.

        Returns:
            logits/predictions,
            loss
        """

        # Mask2Former
        if getattr(
            self.model,
            "is_mask2former",
            False,
        ):
            outputs = self.model(
                pixel_values=images,
            )

            logits = getattr(
                outputs,
                "logits",
                None,
            )

            if logits is None:
                raise RuntimeError(
                    "Mask2Former did not return logits."
                )

            loss = None

            if masks is not None:
                loss = self.loss_fn(
                    logits,
                    masks,
                )

            return logits, loss

        # Standard segmentation model
        logits = self.model(images)

        loss = None

        if masks is not None:
            loss = self.loss_fn(
                logits,
                masks,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch["image"]
        masks = batch["mask"]

        images = images.to(
            self.device,
            non_blocking=True,
        )

        masks = masks.to(
            self.device,
            non_blocking=True,
        )

        return images, masks

    # ------------------------------------------------------------------
    # AMP
    # ------------------------------------------------------------------

    def _autocast(self):
        precision = str(
            self.cfg.train.get(
                "precision",
                "32",
            )
        )

        enabled = (
            precision == "16-mixed"
            and self.device.type == "cuda"
        )

        return torch.amp.autocast(
            device_type="cuda",
            enabled=enabled,
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def load_checkpoint(
        self,
        checkpoint_path: str | Any,
    ) -> dict[str, Any]:
        checkpoint_path = str(
            checkpoint_path
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict
        )

        print(
            f"Loaded checkpoint: "
            f"{checkpoint_path}"
        )

        if isinstance(checkpoint, dict):
            if "epoch" in checkpoint:
                print(
                    f"Checkpoint epoch: "
                    f"{checkpoint['epoch']}"
                )

            if "best_score" in checkpoint:
                print(
                    f"Best score: "
                    f"{checkpoint['best_score']:.4f}"
                )

        return checkpoint

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(
        self,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate the model on the test set.

        Returns:
            Dictionary containing:
                - metrics
                - per_class
                - loss
                - fps
                - qualitative samples
        """

        if checkpoint_path is not None:
            self.load_checkpoint(
                checkpoint_path
            )

        self.model.eval()

        self.metric_manager.reset()

        total_loss = 0.0
        num_batches = 0

        total_images = 0
        total_model_time = 0.0

        qualitative_batch = None

        progress = tqdm(
            self.test_loader,
            desc="[Test]",
            leave=False,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                progress
            ):
                images, masks = (
                    self._prepare_batch(batch)
                )

                # ------------------------------------------------------
                # Save qualitative samples
                # ------------------------------------------------------

                if (
                    self.qualitative_enabled
                    and qualitative_batch is None
                    and batch_idx == 0
                ):
                    max_samples = min(
                        self.qualitative_max_samples,
                        images.shape[0],
                    )

                    qualitative_batch = {
                        "image": images[
                            :max_samples
                        ].detach().cpu(),
                        "mask": masks[
                            :max_samples
                        ].detach().cpu(),
                    }

                # ------------------------------------------------------
                # Inference
                # ------------------------------------------------------

                start_time = time.perf_counter()

                with self._autocast():
                    logits, model_loss = (
                        self._forward(
                            images,
                            masks,
                        )
                    )

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                total_model_time += elapsed
                total_images += images.shape[0]

                # ------------------------------------------------------
                # Loss
                # ------------------------------------------------------

                loss = model_loss

                if loss is None:
                    loss = self.loss_fn(
                        logits,
                        masks,
                    )

                loss_value = float(
                    loss.detach().item()
                )

                total_loss += loss_value
                num_batches += 1

                # ------------------------------------------------------
                # Metrics
                # ------------------------------------------------------

                self.metric_manager.update(
                    logits,
                    masks,
                )

                progress.set_postfix(
                    loss=f"{loss_value:.4f}"
                )

        # ------------------------------------------------------------------
        # Compute metrics
        # ------------------------------------------------------------------

        metrics = self.metric_manager.compute()

        per_class_results = (
            self.metric_manager.compute_per_class()
        )

        test_loss = (
            total_loss
            / max(num_batches, 1)
        )

        fps = (
            total_images
            / max(total_model_time, 1e-8)
        )

        metrics = {
            key: float(value)
            for key, value in metrics.items()
        }

        metrics["loss"] = test_loss
        metrics["fps"] = fps

        # ------------------------------------------------------------------
        # Results
        # ------------------------------------------------------------------

        results = {
            "metrics": metrics,
            "per_class": per_class_results,
            "qualitative": qualitative_batch,
        }

        self._log_results(
            results
        )

        return results

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_results(
        self,
        results: dict[str, Any],
    ) -> None:
        if self.logger is None:
            return

        metrics = results["metrics"]
        per_class = results["per_class"]

        log_data = {}

        for key, value in metrics.items():
            log_data[f"test/{key}"] = float(
                value
            )

        for key, value in per_class.items():
            log_data[f"test/{key}"] = float(
                value
            )

        self.logger.log(
            log_data
        )

        # --------------------------------------------------------------
        # Class names
        # --------------------------------------------------------------

        num_classes = int(
            self.cfg.data.num_classes
        )

        class_labels = {
            index: f"class_{index}"
            for index in range(num_classes)
        }

        configured_labels = self.cfg.data.get(
            "classes",
            self.cfg.data.get(
                "class_labels",
                None,
            ),
        )

        if configured_labels is not None:
            if isinstance(
                configured_labels,
                dict,
            ):
                class_labels = {
                    int(key): str(value)
                    for key, value
                    in configured_labels.items()
                }

            elif isinstance(
                configured_labels,
                (list, tuple),
            ):
                class_labels = {
                    index: str(name)
                    for index, name
                    in enumerate(configured_labels)
                }

        # --------------------------------------------------------------
        # Qualitative results
        # --------------------------------------------------------------

        qualitative = results.get(
            "qualitative"
        )

        if (
            not self.qualitative_enabled
            or qualitative is None
        ):
            return

        images = qualitative["image"]
        masks = qualitative["mask"]

        with torch.no_grad():
            with self._autocast():
                logits, _ = self._forward(
                    images.to(self.device),
                    masks.to(self.device),
                )

        predictions = torch.argmax(
            logits,
            dim=1,
        ).detach().cpu()

        images = images.detach().cpu()
        masks = masks.detach().cpu()

        self.logger.log_qualitative(
            images=images,
            masks=masks,
            predictions=predictions,
            class_labels=class_labels,
        )