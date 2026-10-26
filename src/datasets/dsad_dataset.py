from pathlib import Path
from io import BytesIO
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Subset

from src.datasets.base_dataset import BaseSegmentationDataset


class DSADDataset(BaseSegmentationDataset):
    """
    DSAD single_organ dataset.

    Each row represents one segmentation sample:
        - surgery_id
        - frame_id
        - organ
        - image
        - mask

    The official split is defined at surgery level:
        - train: 21 surgeries
        - validation: 3 surgeries
        - test: 8 surgeries

    The target is binary:
        0 = background
        1 = target organ
    """

    VARIANT = "single_organ"

    ORGAN_COLUMNS = [
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

    REQUIRED_COLUMNS = [
        "surgery_id",
        "frame_id",
        "organ",
        "image",
        "mask",
    ]

    CLASS_NAMES = [
        "background",
        "organ",
    ]

    NUM_CLASSES = 2

    def __init__(
        self,
        root: str,
        transform=None,
        variant: str = "single_organ",
    ):
        super().__init__(root=root, transform=transform)

        if variant != self.VARIANT:
            raise ValueError(
                f"Only variant='{self.VARIANT}' is currently supported. "
                f"Got: '{variant}'"
            )

        self.root = Path(root)
        self.data_root = self.root / variant

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"DSAD variant directory not found: {self.data_root}"
            )

        self.samples = self._load_parquet_splits()

        self._validate_dataset()

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_parquet_splits(self) -> pd.DataFrame:
        """
        Load train/validation/test parquet files and concatenate them.

        An additional `_official_split` column keeps track of the
        original DSAD split.
        """

        dataframes = []

        for split in ["train", "validation", "test"]:
            path = self.data_root / f"{split}.parquet"

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing DSAD split file: {path}"
                )

            df = pd.read_parquet(path).copy()
            df["_official_split"] = split

            dataframes.append(df)

        samples = pd.concat(
            dataframes,
            ignore_index=True,
        )

        return samples

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dataset(self):
        """Validate the structure and official split."""

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.samples.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # Normalize identifiers to strings.
        self.samples["surgery_id"] = (
            self.samples["surgery_id"]
            .astype(str)
            .str.zfill(2)
        )

        self.samples["frame_id"] = (
            self.samples["frame_id"]
            .astype(str)
        )

        # Check organs.
        unknown_organs = set(
            self.samples["organ"].unique()
        ) - set(self.ORGAN_COLUMNS)

        if unknown_organs:
            raise ValueError(
                f"Unknown organs found in DSAD: {unknown_organs}"
            )

        # Check official split leakage.
        split_surgeries = {
            split: set(
                self.samples.loc[
                    self.samples["_official_split"] == split,
                    "surgery_id",
                ].unique()
            )
            for split in ["train", "validation", "test"]
        }

        train_ids = split_surgeries["train"]
        val_ids = split_surgeries["validation"]
        test_ids = split_surgeries["test"]

        if train_ids & val_ids:
            raise ValueError(
                f"Surgery leakage between train and validation: "
                f"{train_ids & val_ids}"
            )

        if train_ids & test_ids:
            raise ValueError(
                f"Surgery leakage between train and test: "
                f"{train_ids & test_ids}"
            )

        if val_ids & test_ids:
            raise ValueError(
                f"Surgery leakage between validation and test: "
                f"{val_ids & test_ids}"
            )

    # ------------------------------------------------------------------
    # Basic dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, index):
        row = self.samples.iloc[index]

        # --------------------------------------------------------------
        # Image
        # --------------------------------------------------------------

        image_bytes = row["image"]["bytes"]

        with Image.open(BytesIO(image_bytes)) as image:
            image = np.array(
                image.convert("RGB")
            )

        # --------------------------------------------------------------
        # Mask
        # --------------------------------------------------------------

        mask_bytes = row["mask"]["bytes"]

        with Image.open(BytesIO(mask_bytes)) as mask:
            mask = np.array(mask)

        # DSAD single_organ:
        # 0   -> background
        # 255 -> target organ
        mask = (mask > 0).astype(np.uint8)

        # --------------------------------------------------------------
        # Sanity check
        # --------------------------------------------------------------

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch at index {index}: "
                f"image={image.shape}, mask={mask.shape}"
            )

        # --------------------------------------------------------------
        # Organ presence metadata
        # --------------------------------------------------------------

        organ_presence = {
            organ: bool(row[organ])
            for organ in self.ORGAN_COLUMNS
        }

        return {
            "image": image,
            "mask": mask,

            # Metadata
            "organ": str(row["organ"]),
            "surgery_id": str(row["surgery_id"]),
            "frame_id": str(row["frame_id"]),
            "official_split": str(row["_official_split"]),

            "organ_presence": organ_presence,
        }

    # ------------------------------------------------------------------
    # Official split
    # ------------------------------------------------------------------

    def create_official_split(self):
        """
        Return the official DSAD train/validation/test splits.

        The split is based on the surgery IDs already provided by DSAD.
        """

        train_indices = self.samples.index[
            self.samples["_official_split"] == "train"
        ].tolist()

        val_indices = self.samples.index[
            self.samples["_official_split"] == "validation"
        ].tolist()

        test_indices = self.samples.index[
            self.samples["_official_split"] == "test"
        ].tolist()

        return (
            Subset(self, train_indices),
            Subset(self, val_indices),
            Subset(self, test_indices),
        )

    # ------------------------------------------------------------------
    # Custom surgery-level split
    # ------------------------------------------------------------------

    def create_split(
        self,
        train_surgeries: Sequence[str],
        val_surgeries: Sequence[str],
        test_surgeries: Sequence[str],
    ):
        """
        Create custom train/validation/test splits at surgery level.

        This is useful when reproducing a specific paper that uses
        a different DSAD split.

        No surgery can appear in more than one split.
        """

        train_surgeries = {
            str(x).zfill(2)
            for x in train_surgeries
        }

        val_surgeries = {
            str(x).zfill(2)
            for x in val_surgeries
        }

        test_surgeries = {
            str(x).zfill(2)
            for x in test_surgeries
        }

        # --------------------------------------------------------------
        # Check overlap
        # --------------------------------------------------------------

        train_val = train_surgeries & val_surgeries
        train_test = train_surgeries & test_surgeries
        val_test = val_surgeries & test_surgeries

        if train_val:
            raise ValueError(
                f"Train/validation surgery overlap: {train_val}"
            )

        if train_test:
            raise ValueError(
                f"Train/test surgery overlap: {train_test}"
            )

        if val_test:
            raise ValueError(
                f"Validation/test surgery overlap: {val_test}"
            )

        # --------------------------------------------------------------
        # Check unknown surgery IDs
        # --------------------------------------------------------------

        available_surgeries = set(
            self.samples["surgery_id"].unique()
        )

        requested_surgeries = (
            train_surgeries
            | val_surgeries
            | test_surgeries
        )

        unknown = requested_surgeries - available_surgeries

        if unknown:
            raise ValueError(
                f"Unknown surgery IDs: {sorted(unknown)}"
            )

        # --------------------------------------------------------------
        # Build indices
        # --------------------------------------------------------------

        train_indices = self.samples.index[
            self.samples["surgery_id"].isin(train_surgeries)
        ].tolist()

        val_indices = self.samples.index[
            self.samples["surgery_id"].isin(val_surgeries)
        ].tolist()

        test_indices = self.samples.index[
            self.samples["surgery_id"].isin(test_surgeries)
        ].tolist()

        return (
            Subset(self, train_indices),
            Subset(self, val_indices),
            Subset(self, test_indices),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_surgery_ids(self):
        """Return all surgery IDs in the dataset."""

        return sorted(
            self.samples["surgery_id"].unique()
        )

    def get_official_surgery_ids(self):
        """Return surgery IDs grouped by official split."""

        return {
            split: sorted(
                self.samples.loc[
                    self.samples["_official_split"] == split,
                    "surgery_id",
                ].unique()
            )
            for split in [
                "train",
                "validation",
                "test",
            ]
        }

    def summary(self):
        """Print a compact dataset summary."""

        print("=" * 70)
        print("DSAD - SINGLE ORGAN")
        print("=" * 70)

        print(f"Total samples: {len(self)}")
        print(
            f"Number of surgeries: "
            f"{self.samples['surgery_id'].nunique()}"
        )

        print("\nOfficial splits:")

        for split, surgery_ids in self.get_official_surgery_ids().items():

            num_samples = int(
                (
                    self.samples["_official_split"] == split
                ).sum()
            )

            print(
                f"  {split:12s}: "
                f"{num_samples:5d} samples | "
                f"{len(surgery_ids):2d} surgeries"
            )

            print(
                f"               {surgery_ids}"
            )

        print("\nOrgans:")

        organ_counts = (
            self.samples["organ"]
            .value_counts()
            .sort_index()
        )

        for organ, count in organ_counts.items():
            print(
                f"  {organ:30s}: {count:5d}"
            )