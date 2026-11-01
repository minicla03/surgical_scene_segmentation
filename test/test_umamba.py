from __future__ import annotations

import sys
import traceback


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_python() -> None:
    section("1. PYTHON")

    print(f"Python version : {sys.version}")
    print(f"Python path    : {sys.executable}")


def test_torch():
    section("2. PYTORCH / CUDA")

    import torch

    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA runtime   : {torch.version.cuda}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU count      : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}          : {torch.cuda.get_device_name(i)}")

        print(f"Current device : {torch.cuda.current_device()}")
    else:
        print("WARNING: CUDA is not available")

    return torch


def test_mamba_import() -> None:
    section("3. MAMBA IMPORT")

    import mamba_ssm

    print("mamba_ssm import: OK")
    print(f"mamba_ssm path   : {mamba_ssm.__file__}")


def test_causal_conv1d() -> None:
    section("4. CAUSAL-CONV1D IMPORT")

    import causal_conv1d

    print("causal_conv1d import: OK")
    print(f"causal_conv1d path  : {causal_conv1d.__file__}")


def test_mamba_forward(torch) -> None:
    section("5. MAMBA FORWARD")

    from mamba_ssm import Mamba

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    model = Mamba(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)

    model.eval()

    batch_size = 2
    sequence_length = 128
    embedding_dim = 64

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        device=device,
    )

    print(f"Input shape : {tuple(x.shape)}")

    with torch.no_grad():
        y = model(x)

    print(f"Output shape: {tuple(y.shape)}")

    assert y.shape == x.shape, (
        f"Unexpected output shape: {y.shape}, "
        f"expected {x.shape}"
    )

    print("Mamba forward: OK")


def test_gpu_computation(torch) -> None:
    section("6. GPU COMPUTATION")

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA is not available")
        return

    device = torch.device("cuda")

    x = torch.randn(2048, 2048, device=device)
    y = torch.randn(2048, 2048, device=device)

    z = x @ y

    torch.cuda.synchronize()

    print(f"Matrix multiplication: OK")
    print(f"Result shape         : {tuple(z.shape)}")
    print(f"Device               : {z.device}")


def test_nnunet() -> None:
    section("7. NNUNET")

    import nnunetv2

    print("nnunetv2 import: OK")
    print(f"nnunetv2 path: {nnunetv2.__file__}")


def run_test(name: str, func):
    try:
        func()
        print(f"\n[PASS] {name}")
        return True

    except Exception as exc:
        print(f"\n[FAIL] {name}")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def test_umamba_bot_2d(torch) -> None:
    section("8. UMAMBA BOT 2D FORWARD")

    from dynamic_network_architectures.building_blocks.helper import (
        get_matching_instancenorm,
    )
    from nnunetv2.utilities.network_initialization import InitWeights_He

    from nnunetv2.nets.UMambaBot_2d import UMambaBot

    device = torch.device("cuda")

    model = UMambaBot(
        input_channels=3,
        n_stages=4,
        features_per_stage=[16, 32, 64, 128],
        conv_op=torch.nn.Conv2d,
        kernel_sizes=[
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
        ],
        strides=[
            [1, 1],
            [2, 2],
            [2, 2],
            [2, 2],
        ],
        n_conv_per_stage=[2, 2, 2, 2],
        num_classes=2,
        n_conv_per_stage_decoder=[2, 2, 2],
        conv_bias=True,
        norm_op=get_matching_instancenorm(torch.nn.Conv2d),
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )

    model.apply(InitWeights_He(1e-2))
    model = model.to(device)
    model.eval()

    x = torch.randn(
        1,
        3,
        128,
        128,
        device=device,
    )

    print(f"Input shape : {tuple(x.shape)}")

    with torch.no_grad():
        y = model(x)

    print(f"Output shape: {tuple(y.shape)}")

    assert y.shape == (1, 2, 128, 128), (
        f"Unexpected output shape: {y.shape}, "
        f"expected {(1, 2, 128, 128)}"
    )

    print("UMambaBot 2D forward: OK")

def main() -> None:
    print("\n")
    print("#" * 70)
    print("# U-MAMBA ENVIRONMENT TEST")
    print("#" * 70)

    results = {}

    results["Python"] = run_test(
        "Python",
        test_python,
    )

    torch = None

    try:
        torch = test_torch()
        results["PyTorch / CUDA"] = True
        print("\n[PASS] PyTorch / CUDA")
    except Exception as exc:
        results["PyTorch / CUDA"] = False
        print("\n[FAIL] PyTorch / CUDA")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    results["mamba_ssm import"] = run_test(
        "mamba_ssm import",
        test_mamba_import,
    )

    results["causal_conv1d import"] = run_test(
        "causal_conv1d import",
        test_causal_conv1d,
    )

    if torch is not None:
        results["Mamba forward"] = run_test(
            "Mamba forward",
            lambda: test_mamba_forward(torch),
        )

        results["GPU computation"] = run_test(
            "GPU computation",
            lambda: test_gpu_computation(torch),
        )

    results["nnU-Net"] = run_test(
        "nnU-Net",
        test_nnunet,
    )

    results["UMambaBot 2D"] = run_test(
        "UMambaBot 2D",
        lambda: test_umamba_bot_2d(torch),
    )

    section("FINAL RESULT")

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status:4}] {name}")

    failed = [name for name, passed in results.items() if not passed]

    print()

    if failed:
        print("Some tests FAILED:")
        for name in failed:
            print(f"  - {name}")

        sys.exit(1)

    print("All tests passed successfully.")
    print("Your U-Mamba environment looks OK.")


if __name__ == "__main__":
    main()