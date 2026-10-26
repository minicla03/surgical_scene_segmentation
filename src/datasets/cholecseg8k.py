from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.datasets.base_dataset import BaseSegmentationDataset


class CholecSeg8kDataset(BaseSegmentationDataset):

    NUM_CLASSES = 13

    CLASS_NAMES = [
        "Background",
        "Abdominal Wall",
        "Liver",
        "Gastrointestinal Tract",
        "Fat",
        "Grasper",
        "Connective Tissue",
        "Blood",
        "Cystic Duct",
        "L-hook Electrocautery",
        "Gallbladder",
        "Hepatic Vein",
        "Liver Ligament",
    ]

    # Original watershed mask value -> contiguous class ID
    # https://constantinpape.github.io/torch-em/torch_em/data/datasets/medical/cholecseg8k.html
    LABEL_MAP = {
        255: 0,   # Background
        50: 0,    # Background
        11: 1,    # Abdominal Wall
        21: 2,    # Liver
        13: 3,    # Gastrointestinal Tract
        12: 4,    # Fat
        31: 5,    # Grasper
        23: 6,    # Connective Tissue
        24: 7,    # Blood
        25: 8,    # Cystic Duct
        32: 9,    # L-hook Electrocautery
        22: 10,   # Gallbladder
        33: 11,   # Hepatic Vein
        5: 12,    # Liver Ligament
    }

    def __init__(
        self,
        root,
        transform=None,
    ):
        super().__init__(
            root=root,
            transform=transform,
        )

        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist: {self.root}"
            )

        video_dirs = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith("video")
        )

        if not video_dirs:
            raise RuntimeError(
                f"No video directories found in {self.root}"
            )

        self.video_ids = sorted(
            int(p.name.replace("video", ""))
            for p in video_dirs
        )

        # --------------------------------------------------
        # Build frame-level index
        # --------------------------------------------------

        self.samples = []

        self._build_index()

    # ======================================================
    # Dataset index
    # ======================================================

    def _build_index(self):

        for video_id in self.video_ids:

            video_name = f"video{video_id:02d}"
            video_dir = self.root / video_name

            clip_dirs = sorted(
                p for p in video_dir.iterdir()
                if p.is_dir()
            )

            for clip_dir in clip_dirs:

                image_paths = sorted(
                    clip_dir.glob("*_endo.png")
                )

                for image_path in image_paths:

                    mask_path = image_path.with_name(
                        image_path.name.replace(
                            "_endo.png",
                            "_endo_watershed_mask.png",
                        )
                    )

                    if not mask_path.exists():
                        raise FileNotFoundError(
                            f"Missing mask:\n{mask_path}"
                        )

                    frame_name = image_path.stem.replace(
                        "_endo",
                        "",
                    )

                    self.samples.append(
                        {
                            "video": video_name,
                            "video_id": video_id,
                            "clip": clip_dir.name,
                            "frame": frame_name,
                            "image_path": image_path,
                            "mask_path": mask_path,
                        }
                    )

    # ======================================================
    # Dataset interface
    # ======================================================

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, index):

        info = self.samples[index]

        # --------------------------------------------------
        # Image
        # --------------------------------------------------

        image = cv2.imread(
            str(info["image_path"]),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image:\n"
                f"{info['image_path']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # --------------------------------------------------
        # Mask
        # --------------------------------------------------

        mask_rgb = cv2.imread(
            str(info["mask_path"]),
            cv2.IMREAD_COLOR,
        )

        if mask_rgb is None:
            raise RuntimeError(
                f"Could not read mask:\n"
                f"{info['mask_path']}"
            )

        # The mask is stored as RGB/BGR,
        # but all channels contain the same values.

        if not (
            np.array_equal(
                mask_rgb[:, :, 0],
                mask_rgb[:, :, 1],
            )
            and
            np.array_equal(
                mask_rgb[:, :, 1],
                mask_rgb[:, :, 2],
            )
        ):
            raise ValueError(
                f"Mask channels are not identical:\n"
                f"{info['mask_path']}"
            )

        mask_original = mask_rgb[:, :, 0]

        # --------------------------------------------------
        # Check values
        # --------------------------------------------------

        unique_values = np.unique(mask_original)

        unknown_values = (
            set(unique_values.tolist())
            - set(self.LABEL_MAP.keys())
        )

        if unknown_values:
            raise ValueError(
                f"Unknown mask values "
                f"{sorted(unknown_values)} in:\n"
                f"{info['mask_path']}"
            )

        # --------------------------------------------------
        # Convert dataset labels -> 0 ... NUM_CLASSES-1
        # --------------------------------------------------

        mask = np.zeros(
            mask_original.shape,
            dtype=np.uint8,
        )

        for original_value, class_id in self.LABEL_MAP.items():

            mask[
                mask_original == original_value
            ] = class_id

        # --------------------------------------------------
        # Shape check
        # --------------------------------------------------

        if image.shape[:2] != mask.shape:

            raise RuntimeError(
                "Image/mask size mismatch:\n"
                f"Image: {image.shape[:2]}\n"
                f"Mask : {mask.shape}"
            )

        return {
            "image": image,
            "mask": mask,
            "video": info["video"],
            "video_id": info["video_id"],
            "clip": info["clip"],
            "frame": info["frame"],
        }

    # ======================================================
    # Split utilities
    # ======================================================

    def get_video_indices(
        self,
        video_ids: List[int],
    ) -> List[int]:

        video_ids = set(int(v) for v in video_ids)

        available_videos = set(self.video_ids)

        missing = video_ids - available_videos

        if missing:
            raise ValueError(
                f"Unknown video IDs: {sorted(missing)}"
            )

        indices = [
            idx
            for idx, sample in enumerate(self.samples)
            if sample["video_id"] in video_ids
        ]

        return indices

    # ======================================================
    # Create custom split
    # ======================================================

    def create_split(
        self,
        train_videos: List[int],
        val_videos: List[int],
        test_videos: List[int],
    ) -> Dict[str, Subset]:

        train_videos = set(int(v) for v in train_videos)
        val_videos = set(int(v) for v in val_videos)
        test_videos = set(int(v) for v in test_videos)

        # --------------------------------------------------
        # Check overlap
        # --------------------------------------------------

        train_val = train_videos & val_videos
        train_test = train_videos & test_videos
        val_test = val_videos & test_videos

        if train_val:
            raise ValueError(
                f"Videos present in both train and val: "
                f"{sorted(train_val)}"
            )

        if train_test:
            raise ValueError(
                f"Videos present in both train and test: "
                f"{sorted(train_test)}"
            )

        if val_test:
            raise ValueError(
                f"Videos present in both val and test: "
                f"{sorted(val_test)}"
            )

        # --------------------------------------------------
        # Check that all videos exist
        # --------------------------------------------------

        requested_videos = (
            train_videos
            | val_videos
            | test_videos
        )

        unknown_videos = (
            requested_videos
            - set(self.video_ids)
        )

        if unknown_videos:
            raise ValueError(
                f"Unknown video IDs: "
                f"{sorted(unknown_videos)}"
            )

        # --------------------------------------------------
        # Create frame indices
        # --------------------------------------------------

        train_indices = self.get_video_indices(
            train_videos
        )

        val_indices = self.get_video_indices(
            val_videos
        )

        test_indices = self.get_video_indices(
            test_videos
        )

        # --------------------------------------------------
        # Create PyTorch subsets
        # --------------------------------------------------

        return {
            "train": Subset(
                self,
                train_indices,
            ),
            "val": Subset(
                self,
                val_indices,
            ),
            "test": Subset(
                self,
                test_indices,
            ),
        }

    # ======================================================
    # Random split at VIDEO level
    # ======================================================

    def create_random_split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[Dict[str, Subset], Dict[str, List[int]]]:

        if not np.isclose(
            train_ratio
            + val_ratio
            + test_ratio,
            1.0,
        ):
            raise ValueError(
                "Split ratios must sum to 1."
            )

        if (
            train_ratio <= 0
            or val_ratio <= 0
            or test_ratio <= 0
        ):
            raise ValueError(
                "All split ratios must be > 0."
            )

        # --------------------------------------------------
        # Shuffle VIDEO IDs
        # --------------------------------------------------

        rng = np.random.default_rng(seed)

        shuffled_videos = np.array(
            self.video_ids,
            dtype=int,
        )

        rng.shuffle(shuffled_videos)

        # --------------------------------------------------
        # Compute number of videos
        # --------------------------------------------------

        n_videos = len(shuffled_videos)

        n_train = int(
            n_videos * train_ratio
        )

        n_val = int(
            n_videos * val_ratio
        )

        n_test = (
            n_videos
            - n_train
            - n_val
        )

        if (
            n_train == 0
            or n_val == 0
            or n_test == 0
        ):
            raise ValueError(
                "One split contains zero videos."
            )

        # --------------------------------------------------
        # Assign videos
        # --------------------------------------------------

        train_videos = sorted(
            shuffled_videos[:n_train].tolist()
        )

        val_videos = sorted(
            shuffled_videos[
                n_train:n_train + n_val
            ].tolist()
        )

        test_videos = sorted(
            shuffled_videos[
                n_train + n_val:
            ].tolist()
        )

        # --------------------------------------------------
        # Create Subsets
        # --------------------------------------------------

        splits = self.create_split(
            train_videos=train_videos,
            val_videos=val_videos,
            test_videos=test_videos,
        )

        split_videos = {
            "train": train_videos,
            "val": val_videos,
            "test": test_videos,
        }

        return splits, split_videos