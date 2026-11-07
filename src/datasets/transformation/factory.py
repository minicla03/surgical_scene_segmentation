from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import RandSpatialCropd, SpatialPadd
from monai.transforms.intensity.dictionary import RandGaussianSmoothd, ScaleIntensityd, RandShiftIntensityd, RandAdjustContrastd
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, Resized
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped

import math
import torch


class TransformFactory:
    """Build MONAI transforms from the dataset configuration."""

    def __init__(self, config):
        self.config = config

    def build_train_transform(self) -> Compose:
        transforms = self._build_preprocessing()
        transforms.extend(self._build_augmentation())
        return Compose(transforms)

    def build_val_transform(self) -> Compose:
        return Compose(self._build_preprocessing())

    def build_test_transform(self) -> Compose:
        return Compose(self._build_preprocessing())

    # =========================================================
    # PREPROCESSING
    # =========================================================

    def _build_preprocessing(self) -> list:
        transforms = []
        preprocessing = self.config.get("preprocessing", {})
        image_cfg = preprocessing.get("image", {})
        mask_cfg = preprocessing.get("mask", {})

        self._add_channel_first(transforms, image_cfg, mask_cfg)
        self._add_resize(transforms, image_cfg)
        self._add_padding(transforms, preprocessing, image_cfg)
        self._add_normalization(transforms, image_cfg)
        self._add_type_casting(transforms, image_cfg, mask_cfg)

        return transforms

    @staticmethod
    def _add_channel_first(transforms: list, image_cfg: dict, mask_cfg: dict) -> None:
        if image_cfg.get("channel_first", False):
            transforms.append(EnsureChannelFirstd(keys=["image"], channel_dim=-1))

        if mask_cfg.get("channel_first", False):
            transforms.append(EnsureChannelFirstd(keys=["mask"], channel_dim="no_channel"))

    @staticmethod
    def _add_resize(transforms: list, image_cfg: dict) -> None:
        size = image_cfg.get("size")
        if size is None:
            return

        height, width = size
        transforms.append(
            Resized(
                keys=["image", "mask"],
                spatial_size=(height, width),
                mode=("bilinear", "nearest"),
            )
        )

    @staticmethod
    def _add_padding(transforms: list, preprocessing: dict, image_cfg: dict) -> None:
        padding_cfg = preprocessing.get("padding", {})
        if not padding_cfg.get("enabled", False):
            return

        size = image_cfg.get("size")
        if size is None:
            raise ValueError("Padding requires preprocessing.image.size.")

        height, width = size
        divisor = padding_cfg.get("divisor", 16)

        padded_height = ((height + divisor - 1) // divisor) * divisor
        padded_width = ((width + divisor - 1) // divisor) * divisor

        transforms.append(
            SpatialPadd(
                keys=["image"], spatial_size=(padded_height, padded_width), mode="edge"
            )
        )
        transforms.append(
            SpatialPadd(
                keys=["mask"],
                spatial_size=(padded_height, padded_width),
                mode="constant",
                constant_values=0,
            )
        )

    @staticmethod
    def _add_normalization(transforms: list, image_cfg: dict) -> None:
        if image_cfg.get("normalize", False):
            transforms.append(ScaleIntensityd(keys=["image"]))

    @staticmethod
    def _add_type_casting(transforms: list, image_cfg: dict, mask_cfg: dict) -> None:
        image_dtype = image_cfg.get("dtype")
        mask_dtype = mask_cfg.get("dtype")

        if image_dtype is None and mask_dtype is None:
            transforms.append(EnsureTyped(keys=["image", "mask"]))
            return

        transforms.append(
            EnsureTyped(keys=["image", "mask"], dtype=[image_dtype, mask_dtype])
        )

    # =========================================================
    # AUGMENTATION
    # =========================================================

    def _build_augmentation(self) -> list:
        augmentation = self.config.get("augmentation", {}).get("train", {})
        if not augmentation.get("enabled", True):
            return []

        transforms = []
        transforms.extend(self._build_scaling(augmentation))
        transforms.extend(self._build_rotation(augmentation))
        transforms.extend(self._build_translation(augmentation))
        transforms.extend(self._build_horizontal_flip(augmentation))
        transforms.extend(self._build_vertical_flip(augmentation))
        transforms.extend(self._build_brightness(augmentation))
        transforms.extend(self._build_contrast(augmentation))
        transforms.extend(self._build_color_jitter(augmentation))
        transforms.extend(self._build_gaussian_blur(augmentation))
        transforms.extend(self._build_random_crop(augmentation))

        return transforms

    # =========================================================
    # GEOMETRIC AUGMENTATION
    # =========================================================

    @staticmethod
    def _build_scaling(augmentation: dict) -> list:
        config = augmentation.get("scaling", {})
        if not config.get("enabled", False):
            return []

        scale_range = config.get("range", [-0.1, 0.1])
        return [
            RandAffined(
                keys=["image", "mask"],
                prob=config.get("probability", 1.0),
                scale_range=tuple(scale_range),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            )
        ]

    @staticmethod
    def _build_rotation(augmentation: dict) -> list:
        config = augmentation.get("rotation", {})
        if not config.get("enabled", False):
            return []

        rotation_range = config.get("range", [-10, 10])
        return [
            RandAffined(
                keys=["image", "mask"],
                prob=config.get("probability", 1.0),
                rotate_range=tuple(TransformFactory._degrees_to_radians(rotation_range)),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            )
        ]

    @staticmethod
    def _build_translation(augmentation: dict) -> list:
        config = augmentation.get("translation", {})
        if not config.get("enabled", False):
            return []

        translation_range = config.get("range", [-0.05, 0.05])
        return [
            RandAffined(
                keys=["image", "mask"],
                prob=config.get("probability", 1.0),
                translate_range=tuple(translation_range),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            )
        ]

    @staticmethod
    def _build_horizontal_flip(augmentation: dict) -> list:
        config = augmentation.get("horizontal_flip", {})
        if not config.get("enabled", False):
            return []

        return [
            RandFlipd(
                keys=["image", "mask"],
                prob=config.get("probability", 0.5),
                spatial_axis=1,
            )
        ]

    @staticmethod
    def _build_vertical_flip(augmentation: dict) -> list:
        config = augmentation.get("vertical_flip", {})
        if not config.get("enabled", False):
            return []

        return [
            RandFlipd(
                keys=["image", "mask"],
                prob=config.get("probability", 0.5),
                spatial_axis=0,
            )
        ]

    # =========================================================
    # INTENSITY AUGMENTATION
    # =========================================================

    @staticmethod
    def _build_brightness(augmentation: dict) -> list:
        config = augmentation.get("brightness", {})
        if not config.get("enabled", False):
            return []

        return [
            RandShiftIntensityd(
                keys=["image"],
                offsets=config.get("range", [-0.1, 0.1]),
                prob=config.get("probability", 0.5),
            )
        ]

    @staticmethod
    def _build_contrast(augmentation: dict) -> list:
        config = augmentation.get("contrast", {})
        if not config.get("enabled", False):
            return []

        return [
            RandAdjustContrastd(
                keys=["image"],
                gamma=tuple(config.get("range", [0.8, 1.2])),
                prob=config.get("probability", 0.5),
            )
        ]

    @staticmethod
    def _build_color_jitter(augmentation: dict) -> list:
        config = augmentation.get("color_jitter", {})
        if not config.get("enabled", False):
            return []

        return [RandColorJitterd(
            keys=["image"],
            brightness=config.get("brightness", 0.0),
            contrast=config.get("contrast", 0.0),
            saturation=config.get("saturation", 0.0),
            hue=config.get("hue", 0.0),
            prob=config.get("probability", 0.5),
        )]

    @staticmethod
    def _build_gaussian_blur(augmentation: dict) -> list:
        config = augmentation.get("gaussian_blur", {})
        if not config.get("enabled", False):
            return []

        sigmas = config.get("sigmas", [0.8, 1.4])
        return [
            RandGaussianSmoothd(
                keys=["image"],
                prob=config.get("probability", 1.0),
                sigma_x=tuple(sigmas),
                sigma_y=tuple(sigmas),
            )
        ]

    # =========================================================
    # CROPPING
    # =========================================================

    @staticmethod
    def _build_random_crop(augmentation: dict) -> list:
        config = augmentation.get("random_crop", {})
        if not config.get("enabled", False):
            return []

        crop_size = tuple(config.get("size", [256, 256]))
        return [
            SpatialPadd(keys=["image"], spatial_size=crop_size, mode="edge"),
            SpatialPadd(
                keys=["mask"], spatial_size=crop_size, mode="constant", constant_values=0
            ),
            RandSpatialCropd(
                keys=["image", "mask"], roi_size=crop_size, random_size=False
            ),
        ]

    # =========================================================
    # UTILITIES
    # =========================================================

    @staticmethod
    def _degrees_to_radians(values: list | tuple) -> tuple:
        return tuple(math.radians(value) for value in values)


from torchvision.transforms.v2 import ColorJitter
from monai.transforms import MapTransform


class RandColorJitterd(MapTransform):
    """Apply torchvision ColorJitter to selected dictionary keys."""

    def __init__(
        self,
        keys,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        prob=0.5,
    ):
        super().__init__(keys)
        self.prob = prob
        self.transform = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, data):
        d = dict(data)

        if torch.rand(1).item() >= self.prob:
            return d

        for key in self.keys:
            d[key] = self.transform(d[key])

        return d