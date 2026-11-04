from __future__ import annotations

import torch


def run_smoke_test(trainer) -> None:
    """
    Esegue un test minimale di train e validation.

    Verifica:
    - shape immagini e mask
    - dtype delle mask
    - classi presenti nelle mask
    - forward
    - loss
    - backward
    - optimizer step
    - validation
    - metriche
    - compatibilità num_classes
    """

    print("\n" + "=" * 70)
    print("SMOKE TEST - TRAIN + VALIDATION")
    print("=" * 70)

    expected_classes = int(trainer.cfg.data.num_classes)

    # ============================================================
    # TRAIN - 1 batch
    # ============================================================
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)

    train_batch = next(iter(trainer.train_loader))
    images, masks = trainer._prepare_batch(train_batch)

    _check_batch(
        images=images,
        masks=masks,
        expected_classes=expected_classes,
        stage="TRAIN",
    )

    with trainer._autocast():
        logits, loss = trainer._forward(
            images,
            masks,
            training=True,
        )

    _check_logits(
        logits=logits,
        expected_classes=expected_classes,
        stage="TRAIN",
    )

    if loss is None:
        raise RuntimeError("Training loss is None.")

    loss_value = float(loss.detach().item())

    print(f"Logits shape : {tuple(logits.shape)}")
    print(f"Loss         : {loss_value:.6f}")

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Training loss is not finite: {loss_value}"
        )

    # Backward
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(),
        max_norm=1.0,
    )

    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)

    print("Backward     : OK")
    print("Optimizer    : OK")

    # ============================================================
    # VALIDATION - 1 batch
    # ============================================================
    trainer.model.eval()
    trainer.metric_manager.reset()

    val_batch = next(iter(trainer.val_loader))
    images, masks = trainer._prepare_batch(val_batch)

    _check_batch(
        images=images,
        masks=masks,
        expected_classes=expected_classes,
        stage="VALIDATION",
    )

    with torch.no_grad():
        with trainer._autocast():
            logits, model_loss = trainer._forward(
                images,
                masks,
                training=False,
            )

            loss = model_loss

            if loss is None:
                loss = trainer.loss_fn(logits, masks)

    _check_logits(
        logits=logits,
        expected_classes=expected_classes,
        stage="VALIDATION",
    )

    if loss is None:
        raise RuntimeError("Validation loss is None.")

    val_loss = float(loss.detach().item())

    print(f"Logits shape : {tuple(logits.shape)}")
    print(f"Loss         : {val_loss:.6f}")

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Validation loss is not finite: {val_loss}"
        )

    # Metrics
    trainer.metric_manager.update(logits, masks)
    metrics = trainer.metric_manager.compute()

    print("\n[METRICS]")

    for name, value in metrics.items():
        print(f"{name:20s}: {float(value):.6f}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)


def _check_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    expected_classes: int,
    stage: str,
) -> None:
    print(f"\n[{stage}]")

    print(f"Images shape : {tuple(images.shape)}")
    print(f"Images dtype : {images.dtype}")

    print(f"Masks shape  : {tuple(masks.shape)}")
    print(f"Masks dtype  : {masks.dtype}")

    print(f"Mask min     : {masks.min().item()}")
    print(f"Mask max     : {masks.max().item()}")

    unique_classes = torch.unique(masks).cpu().tolist()

    print(f"Mask classes : {unique_classes}")

    # ------------------------------------------------------------
    # Images
    # ------------------------------------------------------------
    if images.ndim != 4:
        raise RuntimeError(
            f"{stage}: expected images [B, C, H, W], "
            f"got {tuple(images.shape)}"
        )

    # ------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------
    if masks.ndim != 4:
        raise RuntimeError(
            f"{stage}: expected masks [B, 1, H, W], "
            f"got {tuple(masks.shape)}"
        )

    if masks.shape[1] != 1:
        raise RuntimeError(
            f"{stage}: expected mask channel dimension = 1, "
            f"got shape {tuple(masks.shape)}"
        )

    if masks.dtype != torch.long:
        raise RuntimeError(
            f"{stage}: expected masks dtype torch.long, "
            f"got {masks.dtype}"
        )

    # ------------------------------------------------------------
    # Class range
    # ------------------------------------------------------------
    min_class = int(masks.min().item())
    max_class = int(masks.max().item())

    if min_class < 0:
        raise RuntimeError(
            f"{stage}: mask contains negative class label: "
            f"{min_class}"
        )

    if max_class >= expected_classes:
        raise RuntimeError(
            f"{stage}: mask contains class {max_class}, "
            f"but num_classes={expected_classes}."
        )


def _check_logits(
    logits: torch.Tensor,
    expected_classes: int,
    stage: str,
) -> None:
    if logits.ndim != 4:
        raise RuntimeError(
            f"{stage}: expected logits [B, C, H, W], "
            f"got {tuple(logits.shape)}"
        )

    actual_classes = logits.shape[1]

    if actual_classes != expected_classes:
        raise RuntimeError(
            f"{stage}: model outputs {actual_classes} classes, "
            f"but config specifies {expected_classes}."
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            f"{stage}: logits contain NaN or Inf."
        )

if __name__=="main":
    run_smoke_test()