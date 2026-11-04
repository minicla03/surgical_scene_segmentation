import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import transformers

from src.models.mask2former_wrapper import Mask2FormerWrapper


def print_environment():
    print("\n" + "=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)

    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")

    print(f"PyTorch           : {torch.__version__}")
    print(f"PyTorch CUDA      : {torch.version.cuda}")
    print(f"Transformers      : {transformers.__version__}")

    print(f"CUDA available    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU               : {torch.cuda.get_device_name(0)}")
        print(f"GPU count         : {torch.cuda.device_count()}")
        print(f"cuDNN             : {torch.backends.cudnn.version()}")

    print("\nDeterminism:")
    print(
        "torch deterministic algorithms:",
        torch.are_deterministic_algorithms_enabled(),
    )
    print(
        "cuDNN deterministic:",
        torch.backends.cudnn.deterministic,
    )
    print(
        "cuDNN benchmark:",
        torch.backends.cudnn.benchmark,
    )

    print("\nCUDA environment:")
    print(f"CUDA_HOME         : {os.environ.get('CUDA_HOME')}")
    print(f"CUDA_PATH         : {os.environ.get('CUDA_PATH')}")
    print(f"LD_LIBRARY_PATH   : {os.environ.get('LD_LIBRARY_PATH')}")

    print("\nCUBLAS:")
    print(
        "CUBLAS_WORKSPACE_CONFIG:",
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )

    print("=" * 70)


def set_test_seed(seed=42):
    """
    Replicate the same deterministic settings used by the real training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def main():
    print_environment()

    print("\nApplying training seed configuration...")
    set_test_seed(42)

    print("\nDeterministic after set_seed:")
    print(
        "torch deterministic algorithms:",
        torch.are_deterministic_algorithms_enabled(),
    )
    print(
        "cuDNN deterministic:",
        torch.backends.cudnn.deterministic,
    )
    print(
        "cuDNN benchmark:",
        torch.backends.cudnn.benchmark,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    CHECKPOINT = "facebook/mask2former-swin-base-ade-semantic"
    NUM_CLASSES = 7
    BATCH_SIZE = 2
    HEIGHT = 256
    WIDTH = 256

    print(f"\nDevice: {device}")
    print("Creating Mask2Former...")

    model = Mask2FormerWrapper(
        checkpoint=CHECKPOINT,
        num_labels=NUM_CLASSES,
    ).to(device)

    print("Mask2Former created successfully.")

    # --------------------------------------------------
    # Synthetic input
    # --------------------------------------------------

    images = torch.randn(
        BATCH_SIZE,
        3,
        HEIGHT,
        WIDTH,
        device=device,
    )

    masks = torch.randint(
        0,
        NUM_CLASSES,
        (BATCH_SIZE, HEIGHT, WIDTH),
        device=device,
    )

    # --------------------------------------------------
    # TRAIN FORWARD
    # --------------------------------------------------

    print("\n========== TRAIN FORWARD ==========")

    model.train()
    model.zero_grad(set_to_none=True)

    outputs = model(
        pixel_values=images,
        labels=masks,
    )

    loss = outputs.loss

    assert loss is not None, "Training loss is None!"
    assert torch.isfinite(loss), (
        f"Invalid training loss: {loss}"
    )

    print(f"Loss: {loss.item():.6f}")

    # --------------------------------------------------
    # BACKWARD
    # --------------------------------------------------

    print("\n========== BACKWARD ==========")

    loss.backward()

    gradients = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]

    assert gradients, "No gradients were computed!"

    print(f"Parameters with gradients: {len(gradients)}")
    print("TRAIN FORWARD + BACKWARD: OK")

    # --------------------------------------------------
    # VALIDATION FORWARD
    # --------------------------------------------------

    print("\n========== VALIDATION FORWARD ==========")

    model.eval()

    with torch.no_grad():
        predictions = model(
            pixel_values=images,
        )

    assert isinstance(predictions, torch.Tensor), (
        f"Expected Tensor, got {type(predictions)}"
    )

    expected_shape = (
        BATCH_SIZE,
        NUM_CLASSES,
        HEIGHT,
        WIDTH,
    )

    assert predictions.shape == expected_shape, (
        f"Wrong shape: {predictions.shape}, "
        f"expected {expected_shape}"
    )

    assert torch.isfinite(predictions).all(), (
        "Validation predictions contain NaN or Inf!"
    )

    print(f"Predictions shape: {tuple(predictions.shape)}")
    print(
        f"Predictions min/max: "
        f"{predictions.min().item():.3f} / "
        f"{predictions.max().item():.3f}"
    )

    print("VALIDATION FORWARD: OK")

    print("\n" + "=" * 80)
    print("AMP TEST")
    print("=" * 80)

    model.train()

    with torch.amp.autocast(
        device_type="cuda",
        enabled=True,
    ):
        outputs = model(
            pixel_values=images,
            labels=masks,
        )

    loss = outputs.loss

    print(f"AMP loss: {loss.item():.6f}")
    print(f"AMP loss finite: {torch.isfinite(loss).item()}")

    loss.backward()

    print("AMP FORWARD + BACKWARD: OK")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()