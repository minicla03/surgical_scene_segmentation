
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import RandSpatialCropd, SpatialPadd
from monai.transforms.intensity.dictionary import (
    RandGaussianSmoothd,
    ScaleIntensityd,
)
from monai.transforms.spatial.dictionary import (
    RandFlipd,
    RandAffined,
    Resized,
)
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped


class TransformFactory:
    def __init__(self, config):
        self.config = config

    # =========================================================
    # TRAIN
    # =========================================================

    def build_train_transform(self):
        transforms = []

        # Preprocessing
        transforms.extend(self._build_preprocessing())

        augmentation = self.config.get("augmentation", {}).get(
            "train", {}
        )

        # Scaling
        scaling = augmentation.get("scaling", {})

        if scaling.get("enabled", False):
            scale_range = scaling.get("range", [-0.1, 0.1])
            scale_limit = max(
                abs(scale_range[0]),
                abs(scale_range[1]),
            )

            transforms.append(
                RandAffined(
                    keys=["image", "mask"],
                    prob=scaling.get("probability", 1.0),
                    scale_range=(scale_limit, scale_limit),
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                )
            )

        # Gaussian blur
        blur = augmentation.get("gaussian_blur", {})

        if blur.get("enabled", False):
            transforms.append(
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=blur.get("probability", 1.0),
                    sigma_x=tuple(
                        blur.get("sigma_x", [0.8, 1.4])
                    ),
                    sigma_y=tuple(
                        blur.get("sigma_y", [0.8, 1.4])
                    ),
                )
            )

        # Horizontal flip
        horizontal_flip = augmentation.get(
            "horizontal_flip",
            {},
        )

        if horizontal_flip.get("enabled", False):
            transforms.append(
                RandFlipd(
                    keys=["image", "mask"],
                    prob=horizontal_flip.get(
                        "probability",
                        0.5,
                    ),
                    spatial_axis=1,
                )
            )

        # Random crop
        crop = augmentation.get("random_crop", {})

        if crop.get("enabled", False):
            crop_size = tuple(
                crop.get("size", [256, 256])
            )

            # Image padding
            transforms.append(
                SpatialPadd(
                    keys=["image"],
                    spatial_size=crop_size,
                    mode="edge",
                )
            )

            # Mask padding
            transforms.append(
                SpatialPadd(
                    keys=["mask"],
                    spatial_size=crop_size,
                    mode="constant",
                    constant_values=0,
                )
            )

            # Random crop
            transforms.append(
                RandSpatialCropd(
                    keys=["image", "mask"],
                    roi_size=crop_size,
                    random_size=False,
                )
            )

        return Compose(transforms)

    # =========================================================
    # VALIDATION
    # =========================================================

    def build_val_transform(self):
        return Compose(
            self._build_preprocessing()
        )

    # =========================================================
    # TEST
    # =========================================================

    def build_test_transform(self):
        return Compose(
            self._build_preprocessing()
        )

    # =========================================================
    # PREPROCESSING
    # =========================================================

    def _build_preprocessing(self):
        transforms = []

        preprocessing = self.config.get(
            "preprocessing",
            {},
        )

        # Channel first
        transforms.append(
            EnsureChannelFirstd(
                keys=["image"],
                channel_dim=-1,
            )
        )

        transforms.append(
            EnsureChannelFirstd(
                keys=["mask"],
                channel_dim="no_channel",
            )
        )

        # Resize
        image_cfg = preprocessing.get(
            "image",
            {},
        )

        if "size" in image_cfg:
            height, width = image_cfg["size"]

            transforms.append(
                Resized(
                    keys=["image", "mask"],
                    spatial_size=(height, width),
                    mode=("bilinear", "nearest"),
                )
            )

        padding_cfg = preprocessing.get("padding", {})

        if padding_cfg.get("enabled", False):
            height, width = image_cfg["size"]
            divisor = padding_cfg.get("divisor", 16)

            transforms.append(
                SpatialPadd(
                    keys=["image"],
                    spatial_size=(
                        ((height + divisor - 1) // divisor) * divisor,
                        ((width + divisor - 1) // divisor) * divisor,
                    ),
                    mode="edge",
                )
            )

            transforms.append(
                SpatialPadd(
                    keys=["mask"],
                    spatial_size=(
                        ((height + divisor - 1) // divisor) * divisor,
                        ((width + divisor - 1) // divisor) * divisor,
                    ),
                    mode="constant",
                    constant_values=0,
                )
            )

        # Normalize
        if image_cfg.get("normalize", False):
            transforms.append(
                ScaleIntensityd(
                    keys=["image"]
                )
            )

        transforms.append(
            EnsureTyped(
                keys=["image", "mask"]
            )
        )

        return transforms