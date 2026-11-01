from __future__ import annotations

import torch
import torch.nn as nn

from nnunetv2.nets.UMambaBot_2d import UMambaBot


class UMambaWrapper(nn.Module):
    """
    Adapter between the thesis segmentation framework and U-Mamba.

    Input:
        [B, C, H, W]

    Output:
        [B, num_classes, H, W]

    The wrapper deliberately exposes a standard PyTorch interface so that
    the existing Trainer, losses, metrics and W&B logging can be reused.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int = 6,
        features_per_stage: list[int] | None = None,
        kernel_sizes: list[tuple[int, int]] | None = None,
        strides: list[tuple[int, int]] | None = None,
        n_conv_per_stage: list[int] | None = None,
        n_conv_per_stage_decoder: list[int] | None = None,
        deep_supervision: bool = False,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # Default architecture
        # ------------------------------------------------------------

        if features_per_stage is None:
            features_per_stage = [
                32,
                64,
                128,
                256,
                320,
                320,
            ]

        if kernel_sizes is None:
            kernel_sizes = [
                (3, 3)
                for _ in range(n_stages)
            ]

        if strides is None:
            strides = [
                (1, 1),
                (2, 2),
                (2, 2),
                (2, 2),
                (2, 2),
                (2, 2),
            ]

        if n_conv_per_stage is None:
            n_conv_per_stage = [
                2
                for _ in range(n_stages)
            ]

        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [
                2
                for _ in range(n_stages - 1)
            ]

        # ------------------------------------------------------------
        # Defensive copies
        #
        # U-Mamba modifies some of these lists internally.
        # Therefore they must be mutable lists, not tuples.
        # ------------------------------------------------------------

        features_per_stage = list(features_per_stage)
        kernel_sizes = list(kernel_sizes)
        strides = list(strides)
        n_conv_per_stage = list(n_conv_per_stage)
        n_conv_per_stage_decoder = list(n_conv_per_stage_decoder)

        # ------------------------------------------------------------
        # Sanity checks
        # ------------------------------------------------------------

        if len(features_per_stage) != n_stages:
            raise ValueError(
                "features_per_stage must contain exactly "
                f"{n_stages} values, got {len(features_per_stage)}"
            )

        if len(kernel_sizes) != n_stages:
            raise ValueError(
                "kernel_sizes must contain exactly "
                f"{n_stages} values, got {len(kernel_sizes)}"
            )

        if len(strides) != n_stages:
            raise ValueError(
                "strides must contain exactly "
                f"{n_stages} values, got {len(strides)}"
            )

        if len(n_conv_per_stage) != n_stages:
            raise ValueError(
                "n_conv_per_stage must contain exactly "
                f"{n_stages} values, got {len(n_conv_per_stage)}"
            )

        if len(n_conv_per_stage_decoder) != n_stages - 1:
            raise ValueError(
                "n_conv_per_stage_decoder must contain exactly "
                f"{n_stages - 1} values, got "
                f"{len(n_conv_per_stage_decoder)}"
            )

        # ------------------------------------------------------------
        # U-Mamba
        # ------------------------------------------------------------

        self.model = UMambaBot(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv2d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={
                "eps": 1e-5,
                "affine": True,
            },
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={
                "inplace": True,
            },
            deep_supervision=deep_supervision,
            stem_channels=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns a single segmentation tensor so that the existing Trainer
        can directly compute the configured loss.
        """

        output = self.model(x)

        # Safety handling in case deep supervision is accidentally enabled.
        if isinstance(output, (list, tuple)):
            output = output[0]

        return output
