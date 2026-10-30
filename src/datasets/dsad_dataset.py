from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Subset

from src.datasets.base_dataset import BaseSegmentationDataset


class DSADDataset(BaseSegmentationDataset):

    """
    DSAD dataset supporting both:

        - single_organ
        - multi_organ

    Parameters
    ----------
    root:
        Root directory containing the DSAD variants.

    variant:
        Dataset variant:
            "single_organ"
            "multi_organ"

    transform:
        Optional MONAI transform.

    ---------------------------------------------------------------
    SINGLE ORGAN
    ---------------------------------------------------------------

    Each row represents one image/organ pair:

        image
        mask
        organ

    Target:

        0 = background
        1 = target organ

    ---------------------------------------------------------------
    MULTI ORGAN
    ---------------------------------------------------------------

    Each row represents one image with multiple organ masks:

        image
        mask_abdominal_wall
        mask_colon
        ...
        mask_stomach

    Target:

        0 = background
        1 = abdominal_wall
        2 = colon
        3 = liver
        4 = pancreas
        5 = small_intestine
        6 = spleen
        7 = stomach
    """


    SUPPORTED_VARIANTS = {"single_organ", "multi_organ"}

    SINGLE_ORGAN_CLASS_NAMES = [
        "background",
        "organ",
    ]

    MULTI_ORGAN_COLUMNS = [
        "abdominal_wall",
        "colon",
        "liver",
        "pancreas",
        "small_intestine",
        "spleen",
        "stomach",
    ]

    MULTI_ORGAN_CLASS_NAMES = [
        "background",
        *MULTI_ORGAN_COLUMNS,
    ]

    SINGLE_ORGAN_COLUMNS = [
        "abdominal_wall",
        "colon",
        "inferior_mesenteric_artery",
        "intestinal_veins",
        "liver",
        "pancreas",
        "small_intestine",
        "spleen",
        "stomach",
        "ureter",
        "uterus",
        "vesicular_glands",
    ]

    OFFICIAL_SPLITS = {
        "train": "train.parquet",
        "validation": "validation.parquet",
        "test": "test.parquet",
    }

    def __init__(
        self,
        root: str,
        variant: str = "single_organ",
        transform=None,
    ):
        super().__init__(
            root=root,
            transform=transform,
        )

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
    def _decode_image(value):
        """Decode an image stored as bytes or return it if already decoded."""

        if isinstance(value, bytes):
            return np.array(Image.open(BytesIO(value)).convert("RGB"))

        if isinstance(value, Image.Image):
            return np.array(value.convert("RGB"))

        return np.asarray(value)

    @staticmethod
    def _decode_mask(value):
        """Decode a binary mask stored as bytes or return it if already decoded."""

        if isinstance(value, bytes):
            return np.array(Image.open(BytesIO(value)).convert("L"))

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
            masks.append(self._decode_mask(sample[f"mask_{column}"]) > 0)

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
        """Load one sample."""
        
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

        split_surgeries = {}

        for split_name in required_splits:
            split_surgeries[split_name] = {
                sample["surgery_id"]
                for sample in self.samples
                if sample["split"] == split_name
            }

        for split_a, surgeries_a in split_surgeries.items():
            for split_b, surgeries_b in split_surgeries.items():
                if split_a >= split_b:
                    continue

                overlap = surgeries_a & surgeries_b

                if overlap:
                    raise ValueError(
                        f"Surgery leakage between {split_a} and {split_b}: {sorted(overlap)}"
                    )

    def create_split(
        self,
        strategy: str,
        group_by: str = "surgery",
        **kwargs,
    ):
        """
        Create the DSAD train/validation/test split.

        DSAD uses its official split, so no new percentage split
        or stratification is performed.
        """

        if strategy != "official":
            raise ValueError("DSAD only supports the official split.")

        if group_by != "surgery":
            raise ValueError("DSAD official split must use group_by='surgery'.")

        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise ValueError(
                f"Unexpected arguments for DSAD official split: {unexpected}"
            )

        train_indices = [
            index
            for index, sample in enumerate(self.samples)
            if sample["split"] == "train"
        ]

        val_indices = [
            index
            for index, sample in enumerate(self.samples)
            if sample["split"] == "validation"
        ]

        test_indices = [
            index
            for index, sample in enumerate(self.samples)
            if sample["split"] == "test"
        ]

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