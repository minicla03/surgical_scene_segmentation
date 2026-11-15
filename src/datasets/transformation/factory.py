from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import RandSpatialCropd, SpatialPadd
from monai.transforms.intensity.dictionary import NormalizeIntensityd, RandGaussianSmoothd, ScaleIntensityRanged, RandShiftIntensityd, RandAdjustContrastd
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, Resized
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped

import math
import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from monai.transforms import MapTransform, RandomizableTransform


class RandColorJitterd(RandomizableTransform, MapTransform):
    """Color jitter (brightness, contrast, saturation, hue) driven by the MONAI random state.

    Every random draw uses ``self.R``, so ``set_random_state`` (called per worker by ``seed_worker``)
    controls it like the other MONAI random transforms. The same factors are applied to every key.
    The four operations are applied in a fixed order (torchvision's ColorJitter shuffles it).
    Expects float images in [0, 1].
    """

    def __init__(self, keys, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, prob=0.5):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob)
        self.brightness, self.contrast = float(brightness), float(contrast)
        self.saturation, self.hue = float(saturation), float(hue)
        self._factors: dict[str, float] = {}

    def randomize(self, data=None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return
        draw = lambda amount, center: float(self.R.uniform(center - amount, center + amount)) if amount > 0 else None
        self._factors = {
            "brightness": draw(self.brightness, 1.0),
            "contrast": draw(self.contrast, 1.0),
            "saturation": draw(self.saturation, 1.0),
            "hue": draw(self.hue, 0.0),
        }

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            image = d[key]
            if self._factors["brightness"] is not None:
                image = TF.adjust_brightness(image, self._factors["brightness"])
            if self._factors["contrast"] is not None:
                image = TF.adjust_contrast(image, self._factors["contrast"])
            if self._factors["saturation"] is not None:
                image = TF.adjust_saturation(image, self._factors["saturation"])
            if self._factors["hue"] is not None:
                image = TF.adjust_hue(image, self._factors["hue"])
            d[key] = image
        return d


class TransformFactory:
    """Build MONAI transforms from the dataset configuration."""

    def __init__(self, config):
        self.config = config

    def build_train_transform(self) -> Compose:
        transforms = self._build_preprocessing()
        transforms.extend(self._build_augmentation())
        transforms.extend(self._build_pretrained_normalization())
        return Compose(transforms)

    def build_val_transform(self) -> Compose:
        transforms = self._build_preprocessing()
        transforms.extend(self._build_pretrained_normalization())
        return Compose(transforms)

    def build_test_transform(self) -> Compose:
        transforms = self._build_preprocessing()
        transforms.extend(self._build_pretrained_normalization())
        return Compose(transforms)

    # =========================================================
    # PRETRAINED NORMALIZATION
    # =========================================================

    def _build_pretrained_normalization(self) -> list:
        """Mean/std normalization, applied last and only when the model uses
        pretrained weights (model.pretrained.enabled) and
        model.pretrained.normalization.enabled is true."""
        model_cfg = self.config.get("model") or {}
        pretrained = model_cfg.get("pretrained") or {}
        norm = pretrained.get("normalization") or {}

        if not (pretrained.get("enabled", False) and norm.get("enabled", False)):
            return []

        mean = np.asarray(list(norm["mean"]), dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(list(norm["std"]), dtype=np.float32).reshape(-1, 1, 1)
        return [
            NormalizeIntensityd(
                keys=["image"], subtrahend=mean, divisor=std, channel_wise=False, dtype=np.float32
            )
        ]

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
        # Fixed rescaling of the 8-bit images to [0, 1] (x / 255). NOT a per-image min-max: the
        # result must not depend on the content of the single frame, and it matches what the
        # ImageNet mean/std normalization of the pretrained models expects.
        if image_cfg.get("normalize", False):
            transforms.append(
                ScaleIntensityRanged(
                    keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=False
                )
            )

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
        image_size = (self.config.get("preprocessing", {}).get("image", {})).get("size")
        transforms.extend(self._build_translation(augmentation, image_size))
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
    def _build_translation(augmentation: dict, image_size=None) -> list:
        config = augmentation.get("translation", {})
        if not config.get("enabled", False):
            return []

        if image_size is None:
            raise ValueError("Translation augmentation requires preprocessing.image.size.")

        # MONAI expects translate_range in PIXELS; the config gives a fraction
        # of the image size, so convert it per axis (H, W), symmetric around 0.
        translation_range = config.get("range", [-0.05, 0.05])
        fraction = max(abs(float(v)) for v in translation_range)
        height, width = image_size
        return [
            RandAffined(
                keys=["image", "mask"],
                prob=config.get("probability", 1.0),
                translate_range=(fraction * height, fraction * width),
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

