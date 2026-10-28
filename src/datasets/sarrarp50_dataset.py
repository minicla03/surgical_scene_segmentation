from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

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
        │   │       ├── 000002.png
        │   │       └── ...
        │   └── ...
        │
        ├── val/
        │   └── ...
        │
        └── test/
            └── ...

    Each sample is defined by a segmentation mask:

        video_xx/segmentation/<frame_id>.png

    The corresponding RGB frame is extracted on-demand from:

        video_xx/video_left.avi

    Public interface
    ----------------
    The dataset is instantiated without a split:

        dataset = SARRARP50Dataset(root=...)

    The official train/val/test datasets are then created with:

        train_ds, val_ds, test_ds = dataset.create_official_split()
    """

    NUM_CLASSES = 10

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

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        root,
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
        self.transform = transform

        # ---------------------------------------------------------
        # Base dataset instance
        # ---------------------------------------------------------
        #
        # This is the instance created by Hydra:
        #
        #     dataset = SARRARP50Dataset(root=...)
        #
        # The actual train/val/test datasets are created later
        # by create_official_split().
        #

        if split is None:
            self.split_root = None
            self.samples: List[Dict] = []
            return

        # ---------------------------------------------------------
        # Validate split
        # ---------------------------------------------------------

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Unsupported split '{split}'. "
                f"Expected one of: {sorted(self.VALID_SPLITS)}."
            )

        # ---------------------------------------------------------
        # Split root
        # ---------------------------------------------------------

        self.split_root = self.root / split

        if not self.split_root.exists():
            raise FileNotFoundError(
                f"SAR-RARP50 split not found: {self.split_root}"
            )

        if not self.split_root.is_dir():
            raise NotADirectoryError(
                f"SAR-RARP50 split path is not a directory: "
                f"{self.split_root}"
            )

        # ---------------------------------------------------------
        # Samples
        # ---------------------------------------------------------

        self.samples: List[Dict] = []

        if preload_metadata:
            self._build_index()

    # =============================================================
    # INDEX
    # =============================================================

    def _build_index(self) -> None:
        """
        Build an index containing one entry for every annotated frame.
        """

        if self.split_root is None:
            raise RuntimeError(
                "_build_index() cannot be called on the base dataset "
                "without a split."
            )

        video_dirs = sorted(
            p
            for p in self.split_root.iterdir()
            if p.is_dir() and p.name.startswith("video_")
        )

        if not video_dirs:
            raise RuntimeError(
                f"No video directories found in {self.split_root}"
            )

        for video_dir in video_dirs:

            video_path = video_dir / "video_left.avi"
            segmentation_dir = video_dir / "segmentation"

            # -----------------------------------------------------
            # Video
            # -----------------------------------------------------

            if not video_path.exists():
                raise FileNotFoundError(
                    f"Missing video file: {video_path}"
                )

            # -----------------------------------------------------
            # Segmentation directory
            # -----------------------------------------------------

            if not segmentation_dir.exists():
                raise FileNotFoundError(
                    f"Missing segmentation directory: "
                    f"{segmentation_dir}"
                )

            if not segmentation_dir.is_dir():
                raise NotADirectoryError(
                    f"Segmentation path is not a directory: "
                    f"{segmentation_dir}"
                )

            # -----------------------------------------------------
            # Masks
            # -----------------------------------------------------

            mask_paths = sorted(
                segmentation_dir.glob("*.png")
            )

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
            raise RuntimeError(
                f"No segmentation samples found in {self.split_root}"
            )

    # =============================================================
    # LENGTH
    # =============================================================

    def __len__(self) -> int:
        return len(self.samples)

    # =============================================================
    # VIDEO CACHE
    # =============================================================

    def _get_cap(
        self,
        video_path: Path,
    ) -> cv2.VideoCapture:
        """
        Return a cached VideoCapture for the requested video.

        The cache is initialized lazily so that each DataLoader worker
        owns its own VideoCapture objects.
        """

        if not hasattr(self, "_cap_cache"):
            self._cap_cache: dict[str, cv2.VideoCapture] = {}

        key = str(video_path)

        if key not in self._cap_cache:

            cap = cv2.VideoCapture(key)

            if not cap.isOpened():
                raise RuntimeError(
                    f"Unable to open video: {video_path}"
                )

            self._cap_cache[key] = cap

        return self._cap_cache[key]

    # =============================================================
    # CLEANUP
    # =============================================================

    def __del__(self) -> None:
        """
        Release all cached VideoCapture objects.
        """

        if hasattr(self, "_cap_cache"):

            for cap in self._cap_cache.values():
                try:
                    cap.release()
                except Exception:
                    pass

            self._cap_cache.clear()

    # =============================================================
    # LOAD SAMPLE
    # =============================================================

    def _load_sample(self, index: int) -> Dict:
        """
        Load a single RGB frame and its segmentation mask.
        """

        sample_info = self.samples[index]

        video_path = sample_info["video_path"]
        mask_path = sample_info["mask_path"]
        frame_id = sample_info["frame"]

        # ---------------------------------------------------------
        # Load video frame
        # ---------------------------------------------------------

        cap = self._get_cap(video_path)

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_id,
        )

        success, frame = cap.read()

        if not success:
            raise RuntimeError(
                f"Unable to read frame {frame_id} "
                f"from {video_path}"
            )

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # ---------------------------------------------------------
        # Load segmentation mask
        # ---------------------------------------------------------

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(
                f"Unable to read mask: {mask_path}"
            )

        # ---------------------------------------------------------
        # Image / mask size check
        # ---------------------------------------------------------

        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(
                f"Image/mask size mismatch for "
                f"{sample_info['video']} frame {frame_id}: "
                f"image={image.shape}, "
                f"mask={mask.shape}"
            )

        # ---------------------------------------------------------
        # Validate class IDs
        # ---------------------------------------------------------

        unique_values = np.unique(mask)

        invalid_values = [
            int(value)
            for value in unique_values
            if int(value) < 0
            or int(value) >= self.NUM_CLASSES
        ]

        if invalid_values:
            raise ValueError(
                f"Invalid class ID found in mask {mask_path}: "
                f"{unique_values}. "
                f"Expected values in "
                f"[0, {self.NUM_CLASSES - 1}]."
            )

        # ---------------------------------------------------------
        # Return sample
        # ---------------------------------------------------------

        return {
            "image": image,
            "mask": mask,
            "video": sample_info["video"],
            "frame": frame_id,
        }

    # =============================================================
    # OFFICIAL SPLIT
    # =============================================================

    def create_official_split(self):
        """
        Create the official SAR-RARP50 train/val/test datasets.

        SAR-RARP50 provides train and validation splits.
        If no test split is present, test_ds is returned as None.

        Returns
        -------
        tuple
            train_ds, val_ds, test_ds
        """

        # ---------------------------------------------------------
        # Train
        # ---------------------------------------------------------

        train_path = self.root / "train"

        if not train_path.exists():
            raise FileNotFoundError(
                f"SAR-RARP50 train split not found: {train_path}"
            )

        train_ds = SARRARP50Dataset(
            root=self.root,
            split="train",
            transform=self.transform,
        )

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        val_path = self.root / "val"

        if not val_path.exists():
            raise FileNotFoundError(
                f"SAR-RARP50 validation split not found: {val_path}"
            )

        val_ds = SARRARP50Dataset(
            root=self.root,
            split="val",
            transform=self.transform,
        )

        # ---------------------------------------------------------
        # Test
        # ---------------------------------------------------------

        test_path = self.root / "test"

        if test_path.exists():

            test_ds = SARRARP50Dataset(
                root=self.root,
                split="test",
                transform=self.transform,
            )

        else:
            test_ds = None

        return train_ds, val_ds, test_ds