from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from torch.utils.data import Subset

from src.datasets.base_dataset import BaseSegmentationDataset


class SARRARP50Dataset(BaseSegmentationDataset):
    """
    SAR-RARP50 semantic segmentation dataset.

    Expected directory structure:

        root/
        ├── train/
        │   ├── video_01/
        │   │   ├── video_left.avi
        │   │   └── segmentation/
        │   │       ├── 000001.png
        │   │       └── ...
        │   └── ...
        ├── val/
        │   └── ...
        └── test/
            └── ...

    The dataset is instantiated without a split. The official
    train/validation/test datasets are created through create_split().
    """

    CLASS_NAMES = [
        "Background",
        "Tool clasper",
        "Tool wrist",
        "Tool shaft",
        "Suturing needle",
        "Thread",
        "Suction tool",
        "Needle Holder",
        "Clamps",
        "Catheter",
    ]

    VALID_SPLITS = {
        "train",
        "val",
        "test",
    }

    SPLIT_NAMES = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        transform=None,
        preload_metadata: bool = True,
    ):
        super().__init__(
            root=root,
            transform=transform,
        )

        self.root = Path(root)
        self.split = split

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        if split is None:
            self.split_root = None
            self.samples: List[Dict] = []
            return

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Unsupported split '{split}'. "
                f"Expected one of: {sorted(self.VALID_SPLITS)}."
            )

        self.split_root = self.root / split

        if not self.split_root.exists():
            raise FileNotFoundError(f"SAR-RARP50 split not found: {self.split_root}")

        if not self.split_root.is_dir():
            raise NotADirectoryError(
                f"SAR-RARP50 split path is not a directory: {self.split_root}"
            )

        self.samples: List[Dict] = []

        if preload_metadata:
            self._build_index()

    def _build_index(self) -> None:
        """Build an index containing every annotated frame."""
        if self.split_root is None:
            raise RuntimeError(
                "_build_index() cannot be called on the base dataset without a split."
            )

        video_dirs = sorted(
            path
            for path in self.split_root.iterdir()
            if path.is_dir() and path.name.startswith("video_")
        )

        if not video_dirs:
            raise RuntimeError(f"No video directories found in {self.split_root}")

        for video_dir in video_dirs:
            video_path = video_dir / "video_left.avi"
            segmentation_dir = video_dir / "segmentation"

            if not video_path.exists():
                raise FileNotFoundError(f"Missing video file: {video_path}")

            if not segmentation_dir.exists():
                raise FileNotFoundError(
                    f"Missing segmentation directory: {segmentation_dir}"
                )

            if not segmentation_dir.is_dir():
                raise NotADirectoryError(
                    f"Segmentation path is not a directory: {segmentation_dir}"
                )

            mask_paths = sorted(segmentation_dir.glob("*.png"))

            for mask_path in mask_paths:
                try:
                    frame_id = int(mask_path.stem)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid mask filename: {mask_path.name}. "
                        f"Expected format '<frame_id>.png'."
                    ) from exc

                self.samples.append(
                    {
                        "video": video_dir.name,
                        "video_path": video_path,
                        "mask_path": mask_path,
                        "frame": frame_id,
                    }
                )

        if not self.samples:
            raise RuntimeError(f"No segmentation samples found in {self.split_root}")

    def __len__(self) -> int:
        """Return the number of annotated frames."""
        return len(self.samples)

    def _get_cap(
        self,
        video_path: Path,
    ) -> cv2.VideoCapture:
        """
        Return a cached VideoCapture for the requested video.

        Each DataLoader worker gets its own cache.
        """
        if not hasattr(self, "_cap_cache"):
            self._cap_cache: dict[str, cv2.VideoCapture] = {}

        key = str(video_path)

        if key not in self._cap_cache:
            cap = cv2.VideoCapture(key)

            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video: {video_path}")

            self._cap_cache[key] = cap

        return self._cap_cache[key]

    def __del__(self) -> None:
        """Release cached VideoCapture objects."""
        if hasattr(self, "_cap_cache"):
            for cap in self._cap_cache.values():
                try:
                    cap.release()
                except Exception:
                    pass

            self._cap_cache.clear()

    def _load_sample(self, index: int) -> Dict:
        """Load one RGB frame and its segmentation mask."""
        sample_info = self.samples[index]

        video_path = sample_info["video_path"]
        mask_path = sample_info["mask_path"]
        frame_id = sample_info["frame"]

        cap = self._get_cap(video_path)

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_id,
        )

        success, frame = cap.read()

        if not success:
            raise RuntimeError(f"Unable to read frame {frame_id} from {video_path}")

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(f"Unable to read mask: {mask_path}")

        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(
                f"Image/mask size mismatch for {sample_info['video']} frame {frame_id}: "
                f"image={image.shape}, mask={mask.shape}"
            )

        unique_values = np.unique(mask)

        invalid_values = [
            int(value)
            for value in unique_values
            if int(value) < 0 or int(value) >= self.num_classes
        ]

        if invalid_values:
            raise ValueError(
                f"Invalid class IDs found in mask {mask_path}: {invalid_values}. "
                f"Expected values in [0, {self.num_classes - 1}]."
            )

        return {
            "image": image,
            "mask": mask,
            "video": sample_info["video"],
            "frame": frame_id,
        }

    def _get_split_indices(
        self,
        split: str,
    ) -> List[int]:
        """Return indices belonging to a specific official split."""
        return [
            index for index, sample in enumerate(self.samples) if sample["split"] == split
        ]

    def _create_split_dataset(
        self,
        split: str,
    ) -> Subset:
        """
        Create a dataset containing one official split.

        The base dataset contains samples from all official splits.
        """
        split_indices = [
            index for index, sample in enumerate(self.samples) if sample["split"] == split
        ]

        if not split_indices:
            raise RuntimeError(f"No samples found for split '{split}'.")

        return Subset(
            self,
            split_indices,
        )

    def create_split(
        self,
        strategy: str,
        group_by: str = "video",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        stratified: bool = False,
    ) -> tuple[Subset, Subset, Subset | None]:
        """
        Create the official SAR-RARP50 train/validation/test split.

        SAR-RARP50 already provides the split at video level, so the
        ratio, seed and stratification parameters are not used.
        """
        if strategy != "official":
            raise ValueError(
                "SAR-RARP50 provides an official split. Use strategy='official'."
            )

        if group_by != "video":
            raise ValueError("SAR-RARP50 supports splitting only by video.")


        train_dataset = SARRARP50Dataset(
            root=str(self.root),
            split="train",
            transform=self.transform,
        )

        val_dataset = SARRARP50Dataset(
            root=str(self.root),
            split="val",
            transform=self.transform,
        )

        test_path = self.root / "test"

        test_dataset = None

        if test_path.exists():
            test_dataset = SARRARP50Dataset(
                root=str(self.root),
                split="test",
                transform=self.transform,
            )

        train_videos = sorted({sample["video"] for sample in train_dataset.samples})

        val_videos = sorted({sample["video"] for sample in val_dataset.samples})

        test_videos = []

        if test_dataset is not None:
            test_videos = sorted({sample["video"] for sample in test_dataset.samples})

        print("\n" + "=" * 60)
        print("DATASET SPLIT")
        print("=" * 60)

        print("\nTrain:")
        print(f"  Groups : {train_videos}")
        print(f"  Frames : {len(train_dataset)}")

        print("\nValidation:")
        print(f"  Groups : {val_videos}")
        print(f"  Frames : {len(val_dataset)}")

        if test_dataset is not None:
            print("\nTest:")
            print(f"  Groups : {test_videos}")
            print(f"  Frames : {len(test_dataset)}")

        print("\nTotal:")
        total_frames = (
            len(train_dataset)
            + len(val_dataset)
            + (len(test_dataset) if test_dataset is not None else 0)
        )

        total_groups = len(set(train_videos) | set(val_videos) | set(test_videos))

        print(f"  Groups : {total_groups}")
        print(f"  Frames : {total_frames}")
        print("=" * 60)

        return train_dataset, val_dataset, test_dataset