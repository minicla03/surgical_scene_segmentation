"""
Environment & hardware integration tests for U-Mamba and PyTorch stack.
"""
from __future__ import annotations

import sys
import pytest
import torch


@pytest.mark.env
class TestEnvironment:
    def test_python_version(self):
        """Verify Python version is at least 3.10."""
        assert sys.version_info >= (3, 10), f"Python >= 3.10 required, got {sys.version}"

    def test_pytorch_installation(self):
        """Verify PyTorch is installed and accessible."""
        assert hasattr(torch, "__version__")
        assert torch.__version__ is not None

    def test_cuda_availability(self):
        """Verify CUDA availability and device attributes if GPU is present."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available in current environment.")
        assert torch.cuda.device_count() > 0
        device_name = torch.cuda.get_device_name(0)
        assert len(device_name) > 0

    @pytest.mark.gpu
    def test_gpu_computation(self, require_cuda):
        """Verify tensor computations on GPU produce valid finite results."""
        device = torch.device("cuda")
        x = torch.randn(512, 512, device=device)
        y = torch.randn(512, 512, device=device)
        z = x @ y
        torch.cuda.synchronize()
        assert z.shape == (512, 512)
        assert torch.isfinite(z).all()

    def test_mamba_ssm_import(self):
        """Verify mamba_ssm can be imported."""
        try:
            import mamba_ssm
        except ImportError as e:
            pytest.skip(f"mamba_ssm not installed or not supported in this runtime: {e}")
        assert hasattr(mamba_ssm, "__file__")

    def test_causal_conv1d_import(self):
        """Verify causal_conv1d can be imported."""
        try:
            import causal_conv1d
        except ImportError as e:
            pytest.skip(f"causal_conv1d not installed or not supported in this runtime: {e}")
        assert hasattr(causal_conv1d, "__file__")

    def test_nnunetv2_import(self):
        """Verify nnunetv2 package is available."""
        try:
            import nnunetv2
        except ImportError as e:
            pytest.skip(f"nnunetv2 not installed: {e}")
        assert hasattr(nnunetv2, "__file__")

    @pytest.mark.gpu
    def test_mamba_forward(self, require_cuda):
        """Test Mamba 1D forward pass on CUDA."""
        try:
            from mamba_ssm import Mamba
        except ImportError:
            pytest.skip("mamba_ssm is not installed.")

        device = torch.device("cuda")
        model = Mamba(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(device)
        model.eval()

        batch_size, seq_len, dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, dim, device=device)

        with torch.no_grad():
            out = model(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    @pytest.mark.gpu
    def test_umamba_bot_2d_forward(self, require_cuda):
        """Test UMambaBot 2D neural network forward pass."""
        try:
            from dynamic_network_architectures.building_blocks.helper import (
                get_matching_instancenorm,
            )
            from nnunetv2.utilities.network_initialization import InitWeights_He
            from nnunetv2.nets.UMambaBot_2d import UMambaBot
        except ImportError as e:
            pytest.skip(f"UMambaBot dependencies missing: {e}")

        device = torch.device("cuda")
        model = UMambaBot(
            input_channels=3,
            n_stages=4,
            features_per_stage=[16, 32, 64, 128],
            conv_op=torch.nn.Conv2d,
            kernel_sizes=[[3, 3]] * 4,
            strides=[[1, 1], [2, 2], [2, 2], [2, 2]],
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

        x = torch.randn(1, 3, 128, 128, device=device)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 2, 128, 128)
        assert torch.isfinite(out).all()
