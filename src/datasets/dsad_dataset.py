from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Subset

from src.datasets.base_dataset import BaseSegmentationDataset


class DSADDataset(BaseSegmentationDataset):
    SUPPORTED_VARIANTS = {"single_organ", "multi_organ"}

    SINGLE_ORGAN_CLASS_NAMES = ["background", "organ"]

    MULTI_ORGAN_COLUMNS = [
        "abdominal_wall", "colon", "liver", "pancreas",
        "small_intestine", "spleen", "stomach",
    ]

    MULTI_ORGAN_CLASS_NAMES = ["background", *MULTI_ORGAN_COLUMNS]

    SINGLE_ORGAN_COLUMNS = [
        "abdominal_wall", "colon", "inferior_mesenteric_artery",
        "intestinal_veins", "liver", "pancreas", "small_intestine",
        "spleen", "stomach", "ureter", "uterus", "vesicular_glands",
    ]

    OFFICIAL_SPLITS = {
        "train": "train.parquet",
        "validation": "validation.parquet",
        "test": "test.parquet",
    }

    def __init__(self, root: str, variant: str = "single_organ", transform=None):
        super().__init__(root=root, transform=transform)

        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported DSAD variant: {variant}. "
                f"Available: {self.SUPPORTED_VARIANTS}"
            )

        self.variant = variant
        self.root = Path(root) / variant

        if self.variant == "single_organ":
            self.CLASS_NAMES = self.SINGLE_ORGAN_CLASS_NAMES.copy()
        else:
            self.CLASS_NAMES = self.MULTI_ORGAN_CLASS_NAMES.copy()

        self.samples = self._load_samples()
        self._validate_dataset()

    def _load_samples(self):
        """Load all official DSAD parquet splits."""
        samples = []

        for split_name, filename in self.OFFICIAL_SPLITS.items():
            path = self.root / filename

            if not path.exists():
                raise FileNotFoundError(f"DSAD split not found: {path}")

            dataframe = pd.read_parquet(path)

            for _, row in dataframe.iterrows():
                sample = row.to_dict()
                sample["split"] = split_name
                samples.append(sample)

        return samples

    @staticmethod
    def _unwrap_value(value):
        """
        Extract the actual image data from common serialized formats.
        Supports: bytes, dict with "bytes", dict with "path", PIL images, NumPy arrays
        """
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return value["bytes"]
            if value.get("path") is not None:
                return value["path"]
        return value

    @classmethod
    def _decode_image(cls, value):
        """Decode an RGB image from the DSAD sample."""
        value = cls._unwrap_value(value)

        if isinstance(value, bytes):
            image = Image.open(BytesIO(value))
            if "transparency" in image.info:
                image = image.convert("RGBA")
            return np.array(image.convert("RGB"))
        if isinstance(value, (str, Path)):
            image = Image.open(value)
            if "transparency" in image.info:
                image = image.convert("RGBA")
            return np.array(image.convert("RGB"))
        if isinstance(value, Image.Image):
            return np.array(value.convert("RGB"))

        array = np.asarray(value)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)

        return array

    @classmethod
    def _decode_mask(cls, value):
        """Decode a grayscale mask from the DSAD sample."""
        value = cls._unwrap_value(value)

        if isinstance(value, bytes):
            image = Image.open(BytesIO(value))
            if "transparency" in image.info:
                image = image.convert("RGBA")
            return np.array(image.convert("L"))
        if isinstance(value, (str, Path)):
            image = Image.open(value)
            if "transparency" in image.info:
                image = image.convert("RGBA")
            return np.array(image.convert("L"))
        if isinstance(value, Image.Image):
            return np.array(value.convert("L"))

        return np.asarray(value)

    def _load_single_organ_sample(self, sample):
        """Load a single-organ DSAD sample."""
        image = self._decode_image(sample["image"])
        mask = self._decode_mask(sample["mask"])
        mask = (mask > 0).astype(np.int64)

        return {
            "image": image,
            "mask": mask,
            "surgery_id": str(sample["surgery_id"]),
            "frame_id": sample["frame_id"],
            "organ": sample["organ"],
        }

    def _load_multi_organ_sample(self, sample):
        """Load a multi-organ DSAD sample."""
        image = self._decode_image(sample["image"])
        masks = []

        for column in self.MULTI_ORGAN_COLUMNS:
            organ_mask = self._decode_mask(sample[f"mask_{column}"])
            masks.append(organ_mask > 0)

        masks = np.stack(masks, axis=0)
        mask = np.zeros(masks.shape[1:], dtype=np.int64)

        for class_id, organ_mask in enumerate(masks, start=1):
            mask[organ_mask] = class_id

        return {
            "image": image,
            "mask": mask,
            "surgery_id": str(sample["surgery_id"]),
            "frame_id": sample["frame_id"],
        }

    def _load_sample(self, index):
        """Load one DSAD sample."""
        sample = self.samples[index]
        if self.variant == "single_organ":
            return self._load_single_organ_sample(sample)
        return self._load_multi_organ_sample(sample)

    def __len__(self):
        return len(self.samples)

    def _validate_dataset(self):
        """Validate the official DSAD split."""
        required_splits = set(self.OFFICIAL_SPLITS)
        available_splits = {sample["split"] for sample in self.samples}
        missing_splits = required_splits - available_splits

        if missing_splits:
            raise ValueError(f"Missing DSAD splits: {missing_splits}")

        split_surgeries = {
            split_name: {sample["surgery_id"] for sample in self.samples if sample["split"] == split_name}
            for split_name in required_splits
        }

        for split_a, surgeries_a in split_surgeries.items():
            for split_b, surgeries_b in split_surgeries.items():
                if split_a >= split_b:
                    continue

                overlap = surgeries_a & surgeries_b
                if overlap:
                    raise ValueError(f"Surgery leakage between {split_a} and {split_b}: {sorted(overlap)}")

    def create_split(self, strategy: str, group_by: str = "surgery", **kwargs):
        """
        Create the official DSAD train/validation/test split.
        DSAD provides official parquet splits, so no new
        percentage split or stratification is performed.
        """
        if strategy != "official":
            raise ValueError("DSAD only supports the official split.")
        if group_by != "surgery":
            raise ValueError("DSAD official split must use group_by='surgery'.")
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise ValueError(f"Unexpected arguments for DSAD official split: {unexpected}")

        train_indices = [idx for idx, sample in enumerate(self.samples) if sample["split"] == "train"]
        val_indices = [idx for idx, sample in enumerate(self.samples) if sample["split"] == "validation"]
        test_indices = [idx for idx, sample in enumerate(self.samples) if sample["split"] == "test"]

        train_surgeries = self._get_surgeries(train_indices)
        val_surgeries = self._get_surgeries(val_indices)
        test_surgeries = self._get_surgeries(test_indices)

        self.log_split(
            train_groups=train_surgeries,
            val_groups=val_surgeries,
            test_groups=test_surgeries,
            group_key="surgery_id",
        )
        

        return (
            Subset(self, train_indices),
            Subset(self, val_indices),
            Subset(self, test_indices),
        )

    def _get_surgeries(self, indices):
        """Return unique surgery IDs for a set of samples."""

        return sorted({self.samples[index]["surgery_id"] for index in indices})