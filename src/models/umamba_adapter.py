from __future__ import annotations

import torch
import torch.nn as nn

from nnunetv2.nets.UMambaBot_2d import UMambaBot


class UMambaWrapper(nn.Module):
    """
    Adapter between the thesis segmentation framework and U-Mamba.

    All architecture parameters are configurable through Hydra.

    Input:
        [B, C, H, W]

    Output:
        [B, num_classes, H, W]
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int = 6,
        features_per_stage: list[int] | None = None,
        kernel_sizes: list[list[int]] | None = None,
        strides: list[list[int]] | None = None,
        n_conv_per_stage: list[int] | None = None,
        n_conv_per_stage_decoder: list[int] | None = None,
        deep_supervision: bool = False,
    ):
        super().__init__()


        n_stages = int(n_stages)
        input_channels = int(input_channels)
        num_classes = int(num_classes)

        features_per_stage = [
            int(features)
            for features in features_per_stage
        ]

        kernel_sizes = [
            tuple(int(value) for value in kernel_size)
            for kernel_size in kernel_sizes
        ]

        strides = [
            tuple(int(value) for value in stride)
            for stride in strides
        ]

        n_conv_per_stage = [
            int(value)
            for value in n_conv_per_stage
        ]

        n_conv_per_stage_decoder = [
            int(value)
            for value in n_conv_per_stage_decoder
        ]

        deep_supervision = bool(deep_supervision)

        # ------------------------------------------------------------
        # Sanity checks
        # ------------------------------------------------------------

        if len(features_per_stage) != n_stages:
            raise ValueError(
                "features_per_stage must contain exactly "
                f"{n_stages} values, got "
                f"{len(features_per_stage)}"
            )

        if len(kernel_sizes) != n_stages:
            raise ValueError(
                "kernel_sizes must contain exactly "
                f"{n_stages} values, got "
                f"{len(kernel_sizes)}"
            )

        if len(strides) != n_stages:
            raise ValueError(
                "strides must contain exactly "
                f"{n_stages} values, got "
                f"{len(strides)}"
            )

        if len(n_conv_per_stage) != n_stages:
            raise ValueError(
                "n_conv_per_stage must contain exactly "
                f"{n_stages} values, got "
                f"{len(n_conv_per_stage)}"
            )

        if len(n_conv_per_stage_decoder) != n_stages - 1:
            raise ValueError(
                "n_conv_per_stage_decoder must contain exactly "
                f"{n_stages - 1} values, got "
                f"{len(n_conv_per_stage_decoder)}"
            )

        # ------------------------------------------------------------
        # Build U-Mamba
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

        If deep supervision is enabled and U-Mamba returns multiple
        outputs, only the highest-resolution output is returned so
        that the existing Trainer receives a single segmentation tensor.
        """

        output = self.model(x)

        if isinstance(output, (list, tuple)):
            output = output[0]

        return output
