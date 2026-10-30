from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import Subset

from src.datasets.base_dataset import BaseSegmentationDataset


class CholecSeg8kDataset(BaseSegmentationDataset):
    
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

    # Original watershed mask value -> contiguous class ID.
    # Reference:
    # https://constantinpape.github.io/torch-em/torch_em/data/datasets/medical/cholecseg8k.html
    LABEL_MAP = {
        255: 0,
        50: 0,
        0: 0,
        11: 1,
        21: 2,
        13: 3,
        12: 4,
        31: 5,
        23: 6,
        24: 7,
        25: 8,
        32: 9,
        22: 10,
        33: 11,
        5: 12,
    }

    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)

        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        video_dirs = sorted(
            p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("video")
        )

        if not video_dirs:
            raise RuntimeError(f"No video directories found in {self.root}")

        self.video_ids = sorted(
            int(path.name.replace("video", "")) for path in video_dirs
        )

        self.samples = []
        self._build_index()
        self.video_class_presence = self._build_video_class_presence()

    def _build_index(self) -> None:
        """Build a frame-level index while preserving video information."""

        for video_id in self.video_ids:
            video_name = f"video{video_id:02d}"
            video_dir = self.root / video_name

            clip_dirs = sorted(path for path in video_dir.iterdir() if path.is_dir())

            for clip_dir in clip_dirs:
                image_paths = sorted(clip_dir.glob("*_endo.png"))

                for image_path in image_paths:
                    mask_path = image_path.with_name(
                        image_path.name.replace("_endo.png", "_endo_watershed_mask.png")
                    )

                    if not mask_path.exists():
                        raise FileNotFoundError(f"Missing mask:\n{mask_path}")

                    frame_name = image_path.stem.replace("_endo", "")

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

    def _build_video_class_presence(self) -> dict[int, set[int]]:
        """Return the set of semantic classes present in each video."""

        video_classes = {video_id: set() for video_id in self.video_ids}

        for sample in self.samples:
            mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                raise RuntimeError(f"Could not read mask:\n{sample['mask_path']}")

            unique_values = np.unique(mask)
            unknown_values = set(unique_values.tolist()) - set(self.LABEL_MAP)

            if unknown_values:
                raise ValueError(
                    f"Unknown mask values {sorted(unknown_values)} in:\n{sample['mask_path']}"
                )

            classes = {self.LABEL_MAP[int(value)] for value in unique_values}
            video_classes[sample["video_id"]].update(classes)

        return video_classes

    def __len__(self) -> int:
        """Return the number of frames."""

        return len(self.samples)

    def _load_sample(self, index: int) -> dict:
        """Load and preprocess one image-mask pair."""

        info = self.samples[index]

        image = cv2.imread(str(info["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image:\n{info['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_rgb = cv2.imread(str(info["mask_path"]), cv2.IMREAD_COLOR)
        if mask_rgb is None:
            raise RuntimeError(f"Could not read mask:\n{info['mask_path']}")

        if not (
            np.array_equal(mask_rgb[:, :, 0], mask_rgb[:, :, 1])
            and np.array_equal(mask_rgb[:, :, 1], mask_rgb[:, :, 2])
        ):
            raise ValueError(f"Mask channels are not identical:\n{info['mask_path']}")

        mask_original = mask_rgb[:, :, 0]
        unique_values = np.unique(mask_original)
        unknown_values = set(unique_values.tolist()) - set(self.LABEL_MAP)

        if unknown_values:
            raise ValueError(
                f"Unknown mask values {sorted(unknown_values)} in:\n{info['mask_path']}"
            )

        mask = np.zeros(mask_original.shape, dtype=np.uint8)

        for original_value, class_id in self.LABEL_MAP.items():
            mask[mask_original == original_value] = class_id

        if image.shape[:2] != mask.shape:
            raise RuntimeError(
                f"Image/mask size mismatch:\n"
                f"Image: {image.shape[:2]}\n"
                f"Mask: {mask.shape}"
            )

        return {
            "image": image,
            "mask": mask,
            "video": info["video"],
            "video_id": info["video_id"],
            "clip": info["clip"],
            "frame": info["frame"],
        }

    def get_video_indices(self, video_ids: list[int]) -> list[int]:
        """Return frame indices belonging to the given videos."""
        video_ids = {int(video_id) for video_id in video_ids}
        available_videos = set(self.video_ids)
        missing_videos = video_ids - available_videos

        if missing_videos:
            raise ValueError(f"Unknown video IDs: {sorted(missing_videos)}")

        return [
            index
            for index, sample in enumerate(self.samples)
            if sample["video_id"] in video_ids
        ]

    def _validate_video_split(
        self,
        train_videos: list[int],
        val_videos: list[int],
        test_videos: list[int],
    ) -> None:
        """Validate that video groups do not overlap."""
        train_videos = set(train_videos)
        val_videos = set(val_videos)
        test_videos = set(test_videos)

        if train_videos & val_videos:
            raise ValueError(
                f"Videos present in both train and validation: {sorted(train_videos & val_videos)}"
            )

        if train_videos & test_videos:
            raise ValueError(
                f"Videos present in both train and test: {sorted(train_videos & test_videos)}"
            )

        if val_videos & test_videos:
            raise ValueError(
                f"Videos present in both validation and test: {sorted(val_videos & test_videos)}"
            )

        requested_videos = train_videos | val_videos | test_videos
        unknown_videos = requested_videos - set(self.video_ids)

        if unknown_videos:
            raise ValueError(f"Unknown video IDs: {sorted(unknown_videos)}")

    def _create_subsets(
        self,
        train_videos: list[int],
        val_videos: list[int],
        test_videos: list[int],
    ) -> tuple[Subset, Subset, Subset | None]:
        """Create frame-level subsets from non-overlapping video groups."""
        self._validate_video_split(train_videos, val_videos, test_videos)

        train_dataset = Subset(self, self.get_video_indices(train_videos))
        val_dataset = Subset(self, self.get_video_indices(val_videos))

        test_dataset = None
        if test_videos:
            test_dataset = Subset(self, self.get_video_indices(test_videos))

        return train_dataset, val_dataset, test_dataset

    def _create_random_video_split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple[Subset, Subset, Subset | None]:
        """Create a reproducible random split at video level."""
        rng = np.random.default_rng(seed)
        shuffled_videos = np.array(self.video_ids, dtype=int)
        rng.shuffle(shuffled_videos)

        num_videos = len(shuffled_videos)
        num_train = int(num_videos * train_ratio)
        num_val = int(num_videos * val_ratio)
        num_test = num_videos - num_train - num_val

        if num_train == 0 or num_val == 0:
            raise ValueError(
                "Train and validation splits must contain at least one video."
            )

        if test_ratio > 0 and num_test == 0:
            raise ValueError("Test split is enabled but contains zero videos.")

        train_videos = sorted(shuffled_videos[:num_train].tolist())
        val_videos = sorted(shuffled_videos[num_train : num_train + num_val].tolist())
        test_videos = sorted(shuffled_videos[num_train + num_val :].tolist())

        return self._create_subsets(train_videos, val_videos, test_videos)

    def _create_stratified_video_split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple[Subset, Subset, Subset | None]:
        """Create a multilabel-stratified split at video level."""

        video_ids = np.array(self.video_ids, dtype=int)

        labels = np.zeros((len(video_ids), self.num_classes), dtype=np.int8)

        for row, video_id in enumerate(video_ids):
            for class_id in self.video_class_presence[video_id]:
                labels[row, class_id] = 1

        if test_ratio > 0:
            temp_ratio = val_ratio + test_ratio

            first_splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=temp_ratio, random_state=seed
            )
            train_indices, temp_indices = next(first_splitter.split(video_ids, labels))

            temp_video_ids = video_ids[temp_indices]
            temp_labels = labels[temp_indices]

            relative_test_ratio = test_ratio / temp_ratio

            second_splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=relative_test_ratio, random_state=seed
            )
            val_indices, test_indices = next(
                second_splitter.split(temp_video_ids, temp_labels)
            )

            train_videos = sorted(video_ids[train_indices].tolist())
            val_videos = sorted(temp_video_ids[val_indices].tolist())
            test_videos = sorted(temp_video_ids[test_indices].tolist())

        else:
            first_splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=val_ratio, random_state=seed
            )
            train_indices, val_indices = next(first_splitter.split(video_ids, labels))

            train_videos = sorted(video_ids[train_indices].tolist())
            val_videos = sorted(video_ids[val_indices].tolist())
            test_videos = []

        return self._create_subsets(train_videos, val_videos, test_videos)

    def create_split(
        self,
        strategy: str,
        group_by: str = "video",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        stratified: bool = True,
    ) -> tuple[Subset, Subset, Subset | None]:
        """Create train, validation and optional test splits."""

        if group_by != "video":
            raise ValueError(
                "CholecSeg8k supports splitting only by video."
            )

        if strategy == "official":
            raise ValueError(
                "CholecSeg8k does not provide an official split. "
                "Use strategy='percentage'."
            )

        if strategy != "percentage":
            raise ValueError(f"Unsupported split strategy: {strategy}")
                

        if train_ratio <= 0 or val_ratio <= 0:
            raise ValueError("Train and validation ratios must be greater than zero.")

        if test_ratio < 0:
            raise ValueError("Test ratio cannot be negative.")

        total_ratio = train_ratio + val_ratio + test_ratio

        if not np.isclose(total_ratio, 1.0):
            raise ValueError("Train, validation and test ratios must sum to 1.0.")

        if stratified:
            train_dataset, val_dataset, test_dataset = (

                self._create_stratified_video_split(
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    seed=seed,
                )
            )
        else:
            train_dataset, val_dataset, test_dataset = (
                self._create_random_video_split(
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    seed=seed,
                )
            )

        train_videos = sorted(
            {
                self.samples[index]["video_id"]
                for index in train_dataset.indices
            }
        )

        val_videos = sorted(
            {
                self.samples[index]["video_id"]
                for index in val_dataset.indices
            }
        )

        test_videos = []

        if test_dataset is not None:
            test_videos = sorted(
                {
                    self.samples[index]["video_id"]
                    for index in test_dataset.indices
                }
            )

        self.log_split(
            train_groups=train_videos,
            val_groups=val_videos,
            test_groups=test_videos,
            group_key="video_id",
        )

        return train_dataset, val_dataset, test_dataset