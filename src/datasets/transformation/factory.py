from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Resized,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandAdjustContrastd,
    RandGaussianNoised,
    ScaleIntensityd,
    EnsureTyped,
)


class TransformFactory:

    def __init__(self, config):
        self.config = config

    # =========================================================
    # TRAIN
    # =========================================================

    def build_train_transform(self):

        transforms = []

        # -----------------------------------------------------
        # Preprocessing
        # -----------------------------------------------------

        transforms.extend(
            self._build_preprocessing()
        )

        # -----------------------------------------------------
        # Augmentation
        # -----------------------------------------------------

        augmentation = (
            self.config
            .get("augmentation", {})
            .get("train", {})
        )

        # -----------------------------------------------------
        # Horizontal flip
        # -----------------------------------------------------

        horizontal_flip = augmentation.get(
            "horizontal_flip", {}
        )

        if horizontal_flip.get("enabled", False):

            transforms.append(
                RandFlipd(
                    keys=["image", "mask"],
                    prob=horizontal_flip.get(
                        "probability",
                        horizontal_flip.get("p", 0.5)
                    ),
                    spatial_axis=1,
                )
            )

        # -----------------------------------------------------
        # Vertical flip
        # -----------------------------------------------------

        vertical_flip = augmentation.get(
            "vertical_flip", {}
        )

        if vertical_flip.get("enabled", False):

            transforms.append(
                RandFlipd(
                    keys=["image", "mask"],
                    prob=vertical_flip.get(
                        "probability",
                        vertical_flip.get("p", 0.5)
                    ),
                    spatial_axis=0,
                )
            )

        # -----------------------------------------------------
        # Rotation
        # -----------------------------------------------------

        rotation = augmentation.get(
            "rotation",
            augmentation.get("rotate", {})
        )

        if rotation.get("enabled", False):

            # Config in degrees
            range_degrees = rotation.get(
                "range",
                [-10, 10]
            )

            transforms.append(
                RandRotated(
                    keys=["image", "mask"],
                    prob=rotation.get(
                        "probability",
                        rotation.get("p", 0.3)
                    ),
                    range_x=(
                        range_degrees[0] * 3.141592653589793 / 180.0,
                        range_degrees[1] * 3.141592653589793 / 180.0,
                    ),
                    mode=("bilinear", "nearest"),
                )
            )

        # -----------------------------------------------------
        # Zoom
        # -----------------------------------------------------

        zoom = augmentation.get(
            "zoom",
            {}
        )

        if zoom.get("enabled", False):

            zoom_range = zoom.get(
                "range",
                [0.9, 1.1]
            )

            transforms.append(
                RandZoomd(
                    keys=["image", "mask"],
                    prob=zoom.get(
                        "probability",
                        zoom.get("p", 0.3)
                    ),
                    min_zoom=zoom_range[0],
                    max_zoom=zoom_range[1],
                    mode=("bilinear", "nearest"),
                )
            )

        # -----------------------------------------------------
        # Contrast
        # -----------------------------------------------------

        contrast = augmentation.get(
            "contrast",
            {}
        )

        if contrast.get("enabled", False):

            transforms.append(
                RandAdjustContrastd(
                    keys=["image"],
                    prob=contrast.get(
                        "probability",
                        contrast.get("p", 0.3)
                    ),
                    gamma=tuple(
                        contrast.get(
                            "gamma",
                            [0.8, 1.2]
                        )
                    ),
                )
            )

        # -----------------------------------------------------
        # Gaussian noise
        # -----------------------------------------------------

        gaussian_noise = augmentation.get(
            "gaussian_noise",
            {}
        )

        if gaussian_noise.get("enabled", False):

            transforms.append(
                RandGaussianNoised(
                    keys=["image"],
                    prob=gaussian_noise.get(
                        "probability",
                        gaussian_noise.get("p", 0.2)
                    ),
                    mean=gaussian_noise.get(
                        "mean",
                        0.0
                    ),
                    std=gaussian_noise.get(
                        "std",
                        0.01
                    ),
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
            {}
        )

        # -----------------------------------------------------
        # Channel first
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Resize
        # -----------------------------------------------------

        image_cfg = preprocessing.get(
            "image",
            {}
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

        # -----------------------------------------------------
        # Normalization
        # -----------------------------------------------------

        if image_cfg.get(
            "normalize",
            False
        ):

            transforms.append(
                ScaleIntensityd(
                    keys=["image"],
                )
            )

        # -----------------------------------------------------
        # Convert to tensors
        # -----------------------------------------------------

        transforms.append(
            EnsureTyped(
                keys=["image", "mask"],
            )
        )

        return transforms