from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import RandSpatialCropd, SpatialPadd
from monai.transforms.intensity.dictionary import RandGaussianSmoothd, ScaleIntensityd
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, Resized
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped


class TransformFactory:
    def __init__(self, config):
        self.config = config


    def build_train_transform(self):
        transforms = self._build_preprocessing()
        transforms.extend(self._build_augmentation())
        return Compose(transforms)

    def build_val_transform(self):
        return Compose(self._build_preprocessing())

    def build_test_transform(self):
        return Compose(self._build_preprocessing())

    # =========================================================
    # PREPROCESSING
    # =========================================================

    def _build_preprocessing(self):
        transforms = []
        preprocessing = self.config.get("preprocessing", {})

        image_cfg = preprocessing.get("image", {})
        mask_cfg = preprocessing.get("mask", {})

        # Channel first
        if image_cfg.get("channel_first", False):
            transforms.append(
                EnsureChannelFirstd(keys=["image"], channel_dim=-1)
            )

        if mask_cfg.get("channel_first", False):
            transforms.append(
                EnsureChannelFirstd(keys=["mask"], channel_dim="no_channel")
            )

        # Resize
        if "size" in image_cfg:
            height, width = image_cfg["size"]
            transforms.append(
                Resized(
                    keys=["image", "mask"],
                    spatial_size=(height, width),
                    mode=("bilinear", "nearest"),
                )
            )

        # Padding
        padding_cfg = preprocessing.get("padding", {})
        if padding_cfg.get("enabled", False):
            height, width = image_cfg["size"]
            divisor = padding_cfg.get("divisor", 16)

            pad_height = ((height + divisor - 1) // divisor) * divisor
            pad_width = ((width + divisor - 1) // divisor) * divisor

            transforms.append(
                SpatialPadd(
                    keys=["image"],
                    spatial_size=(pad_height, pad_width),
                    mode="edge",
                )
            )
            transforms.append(
                SpatialPadd(
                    keys=["mask"],
                    spatial_size=(pad_height, pad_width),
                    mode="constant",
                    constant_values=0,
                )
            )

        # Normalization
        if image_cfg.get("normalize", False):
            transforms.append(ScaleIntensityd(keys=["image"]))

        # Type casting
        image_dtype = image_cfg.get("dtype")
        mask_dtype = mask_cfg.get("dtype")
        dtype = {}

        if image_dtype is not None:
            dtype["image"] = image_dtype

        if mask_dtype is not None:
            dtype["mask"] = mask_dtype

        if dtype:
            transforms.append(
                EnsureTyped(
                    keys=["image", "mask"],
                    dtype=[image_dtype, mask_dtype],
                )
            )
        else:
            transforms.append(
                EnsureTyped(keys=["image", "mask"])
            )

        return transforms

    # =========================================================
    # AUGMENTATION
    # =========================================================

    def _build_augmentation(self):
        transforms = []
        augmentation = self.config.get("augmentation", {}).get("train", {})

        if not augmentation.get("enabled", True):
            return transforms

        # Horizontal flip
        horizontal_flip = augmentation.get("horizontal_flip", {})
        if horizontal_flip.get("enabled", False):
            transforms.append(
                RandFlipd(
                    keys=["image", "mask"],
                    prob=horizontal_flip.get("probability", 0.5),
                    spatial_axis=1,
                )
            )

        # Scaling
        scaling = augmentation.get("scaling", {})
        if scaling.get("enabled", False):
            scale_range = scaling.get("range", [-0.1, 0.1])
            transforms.append(
                RandAffined(
                    keys=["image", "mask"],
                    prob=scaling.get("probability", 1.0),
                    scale_range=tuple(scale_range),
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                )
            )

        # Gaussian blur
        gaussian_blur = augmentation.get("gaussian_blur", {})
        if gaussian_blur.get("enabled", False):
            sigma_x = gaussian_blur.get("sigma_x", [0.8, 1.4])
            sigma_y = gaussian_blur.get("sigma_y", [0.8, 1.4])
            transforms.append(
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=gaussian_blur.get("probability", 1.0),
                    sigma_x=tuple(sigma_x),
                    sigma_y=tuple(sigma_y),
                )
            )

        # Random crop
        random_crop = augmentation.get("random_crop", {})
        if random_crop.get("enabled", False):
            crop_size = tuple(random_crop.get("size", [256, 256]))

            transforms.append(
                SpatialPadd(
                    keys=["image"],
                    spatial_size=crop_size,
                    mode="edge",
                )
            )
            transforms.append(
                SpatialPadd(
                    keys=["mask"],
                    spatial_size=crop_size,
                    mode="constant",
                    constant_values=0,
                )
            )
            transforms.append(
                RandSpatialCropd(
                    keys=["image", "mask"],
                    roi_size=crop_size,
                    random_size=False,
                )
            )

        return transforms