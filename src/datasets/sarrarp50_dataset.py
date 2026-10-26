from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.datasets.base_dataset import BaseSegmentationDataset


class SARRARP50Dataset(BaseSegmentationDataset):
    """
    SAR-RARP50 semantic segmentation dataset.

    Each sample is defined by a segmentation mask:
        video_xx/segmentation/<frame_id>.png

    The corresponding RGB image is extracted on-demand from:
        video_xx/video_left.avi
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

    def __init__(
        self,
        root,
        split: str,
        transform=None,
        preload_metadata: bool = True,
    ):
        super().__init__(
            root=root,
            transform=transform,
        )

        self.root = Path(root)
        self.split = split
        self.split_root = self.root / split

        if not self.split_root.exists():
            raise FileNotFoundError(
                f"SAR-RARP50 split not found: {self.split_root}"
            )

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Unsupported split '{split}'. "
                f"Expected one of: train, val, test."
            )

        self.samples: List[Dict] = []

        if preload_metadata:
            self._build_index()

    def _build_index(self):
        """
        Build an index containing one entry for every annotated frame.
        """

        video_dirs = sorted(
            p for p in self.split_root.iterdir()
            if p.is_dir() and p.name.startswith("video_")
        )

        if not video_dirs:
            raise RuntimeError(
                f"No video directories found in {self.split_root}"
            )

        for video_dir in video_dirs:

            video_path = video_dir / "video_left.avi"
            segmentation_dir = video_dir / "segmentation"

            if not video_path.exists():
                raise FileNotFoundError(
                    f"Missing video file: {video_path}"
                )

            if not segmentation_dir.exists():
                raise FileNotFoundError(
                    f"Missing segmentation directory: "
                    f"{segmentation_dir}"
                )

            mask_paths = sorted(segmentation_dir.glob("*.png"))

            for mask_path in mask_paths:

                try:
                    frame_id = int(mask_path.stem)
                except ValueError:
                    raise ValueError(
                        f"Invalid mask filename: {mask_path.name}. "
                        f"Expected format '<frame_id>.png'."
                    )

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

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, index):

        sample_info = self.samples[index]

        video_path = sample_info["video_path"]
        mask_path = sample_info["mask_path"]
        frame_id = sample_info["frame"]

        # ---------------------------------------------------------
        # Load video frame
        # ---------------------------------------------------------

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open video: {video_path}"
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        success, frame = cap.read()
        cap.release()

        if not success:
            raise RuntimeError(
                f"Unable to read frame {frame_id} "
                f"from {video_path}"
            )

        # OpenCV -> RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        # Sanity checks
        # ---------------------------------------------------------

        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(
                f"Image/mask size mismatch for "
                f"{sample_info['video']} frame {frame_id}: "
                f"image={image.shape}, mask={mask.shape}"
            )

        unique_values = np.unique(mask)

        if any(int(value) >= self.NUM_CLASSES for value in unique_values):
            raise ValueError(
                f"Invalid class ID found in mask {mask_path}: "
                f"{unique_values}. "
                f"Expected values in [0, {self.NUM_CLASSES - 1}]."
            )

        return {
            "image": image,
            "mask": mask,
            "video": sample_info["video"],
            "frame": frame_id,
        }