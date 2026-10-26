from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
from PIL import Image

from src.datasets.base_dataset import BaseSegmentationDataset


class ATLAS120KDataset(BaseSegmentationDataset):
    """
    ATLAS-120k semantic segmentation dataset.

    Expected directory structure:

        root/
        └── <procedure>/
            └── <video>/
                ├── clip_index.json
                ├── clip_0001/
                │   ├── images/
                │   │   ├── frame_000001.jpg
                │   │   └── ...
                │   └── masks/
                │       ├── frame_000001.png
                │       └── ...
                └── clip_0002/
                    └── ...

    Each sample contains:

        {
            "image": np.ndarray,       # H x W x 3
            "mask": np.ndarray,        # H x W
            "procedure": str,
            "video": str,
            "clip": str,
            "frame": int,
        }

    Masks are converted from the original 47-class ATLAS
    taxonomy to the consolidated 30-class taxonomy.
    """

    NUM_CLASSES = 30
    NUM_ORIGINAL_CLASSES = 47

    CLASS_NAMES = (
        "Background",
        "Tools/camera",
        "Vein",
        "Artery",
        "Nerve",
        "Small intestine",
        "Colon/rectum",
        "Abdominal wall",
        "Diaphragm",
        "Fat",
        "Liver",
        "Bile/lymph duct",
        "Gallbladder",
        "Hepatic ligament",
        "Cystic plate",
        "Stomach",
        "Spleen",
        "Uterus",
        "Ovary",
        "Oviduct",
        "Prostate",
        "Urethra",
        "Ligated plexus",
        "Seminal vesicles",
        "Non anatomical",
        "Bladder",
        "Lung",
        "Airway (bronchus/trachea)",
        "Esophagus",
        "Pericardium",
    )

    # Original ATLAS 47 classes -> consolidated 30 classes
    ORIGINAL_TO_TRAIN_ID = np.array(
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            3, 2, 10, 11, 12, 2, 13, 14, 15, 11,
            9, 11, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 0, 26, 27, 28, 29, 2, 11, 4,
            0, 24, 0, 0, 0, 0, 0
        ],
        dtype=np.uint8,
    )

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        root,
        transform=None,
        validate_clip_index: bool = True,
    ):
        super().__init__(
            root=root,
            transform=transform,
        )

        self.root = Path(root)
        self.validate_clip_index = validate_clip_index

        # List of sample dictionaries
        self.samples: List[Dict] = []

        # Statistics useful for debugging/sanity checks
        self.validation_stats = {
            "videos": 0,
            "clips": 0,
            "images": 0,
            "valid_samples": 0,
            "missing_images": 0,
            "missing_masks": 0,
            "json_mismatches": 0,
        }

        self._build_index()

    # ------------------------------------------------------------------
    # INDEX
    # ------------------------------------------------------------------

    def _build_index(self):
        """Build the image-mask index using clip_index.json."""

        if not self.root.exists():
            raise FileNotFoundError(
                f"ATLAS-120k root does not exist: {self.root}"
            )

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"ATLAS-120k root is not a directory: {self.root}"
            )

        # Procedure directories
        procedure_dirs = sorted(
            p for p in self.root.iterdir()
            if p.is_dir()
        )

        if not procedure_dirs:
            raise RuntimeError(
                f"No procedure directories found in {self.root}"
            )

        for procedure_dir in procedure_dirs:

            procedure = procedure_dir.name

            # Video directories
            video_dirs = sorted(
                p for p in procedure_dir.iterdir()
                if p.is_dir()
            )

            for video_dir in video_dirs:

                self.validation_stats["videos"] += 1

                clip_index_path = video_dir / "clip_index.json"

                if not clip_index_path.exists():
                    raise FileNotFoundError(
                        f"Missing clip_index.json:\n"
                        f"{clip_index_path}"
                    )

                clip_index = self._load_clip_index(
                    clip_index_path
                )

                self._process_video(
                    procedure=procedure,
                    video_dir=video_dir,
                    clip_index=clip_index,
                )

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found in {self.root}"
            )

        self._print_statistics()

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    @staticmethod
    def _load_clip_index(path: Path) -> Dict:
        """Load and validate clip_index.json."""

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON file: {path}\n{e}"
            ) from e

        if "clips" not in data:
            raise ValueError(
                f"'clips' field not found in {path}"
            )

        if not isinstance(data["clips"], dict):
            raise ValueError(
                f"'clips' must be a dictionary in {path}"
            )

        return data

    # ------------------------------------------------------------------
    # VIDEO / CLIP
    # ------------------------------------------------------------------

    def _process_video(
        self,
        procedure: str,
        video_dir: Path,
        clip_index: Dict,
    ):
        """Process all clips belonging to one video."""

        clips_from_json = clip_index["clips"]

        for clip_name, json_frames in clips_from_json.items():

            self.validation_stats["clips"] += 1

            clip_dir = video_dir / clip_name

            if not clip_dir.exists():
                raise FileNotFoundError(
                    f"Clip declared in clip_index.json "
                    f"does not exist:\n{clip_dir}"
                )

            images_dir = clip_dir / "images"
            masks_dir = clip_dir / "masks"

            if not images_dir.exists():
                raise FileNotFoundError(
                    f"Missing images directory:\n{images_dir}"
                )

            if not masks_dir.exists():
                raise FileNotFoundError(
                    f"Missing masks directory:\n{masks_dir}"
                )

            self._process_clip(
                procedure=procedure,
                video=video_dir.name,
                clip=clip_name,
                clip_dir=clip_dir,
                images_dir=images_dir,
                masks_dir=masks_dir,
                json_frames=json_frames,
            )

    # ------------------------------------------------------------------
    # FRAME VALIDATION
    # ------------------------------------------------------------------

    def _process_clip(
        self,
        procedure: str,
        video: str,
        clip: str,
        clip_dir: Path,
        images_dir: Path,
        masks_dir: Path,
        json_frames: List[int],
    ):
        """
        Validate frames against clip_index.json and create samples.
        """

        # --------------------------------------------------------------
        # Frames declared by JSON
        # --------------------------------------------------------------

        json_frame_ids = set()

        for frame in json_frames:

            try:
                frame_id = int(frame)
            except (TypeError, ValueError):

                raise ValueError(
                    f"Invalid frame ID '{frame}' "
                    f"in {clip_dir.parent / 'clip_index.json'}"
                )

            json_frame_ids.add(frame_id)

        # --------------------------------------------------------------
        # Actual image files
        # --------------------------------------------------------------

        image_paths = sorted(
            p for p in images_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in self.IMAGE_EXTENSIONS
        )

        self.validation_stats["images"] += len(image_paths)

        actual_frame_ids = set()

        for image_path in image_paths:

            frame_id = self._frame_id_from_filename(
                image_path
            )

            actual_frame_ids.add(frame_id)

        # --------------------------------------------------------------
        # JSON <-> filesystem consistency
        # --------------------------------------------------------------

        missing_images = json_frame_ids - actual_frame_ids
        extra_images = actual_frame_ids - json_frame_ids

        if missing_images:

            self.validation_stats["missing_images"] += len(
                missing_images
            )

            raise RuntimeError(
                f"Frames declared in clip_index.json but "
                f"missing from images/:\n"
                f"Procedure: {procedure}\n"
                f"Video: {video}\n"
                f"Clip: {clip}\n"
                f"Missing frames: {sorted(missing_images)[:20]}"
            )

        if extra_images:

            self.validation_stats["json_mismatches"] += len(
                extra_images
            )

            raise RuntimeError(
                f"Images found on disk but not declared "
                f"in clip_index.json:\n"
                f"Procedure: {procedure}\n"
                f"Video: {video}\n"
                f"Clip: {clip}\n"
                f"Extra frames: {sorted(extra_images)[:20]}"
            )

        # --------------------------------------------------------------
        # Build image-mask pairs
        # --------------------------------------------------------------

        for image_path in image_paths:

            frame_id = self._frame_id_from_filename(
                image_path
            )

            mask_path = masks_dir / (
                f"{image_path.stem}.png"
            )

            if not mask_path.exists():

                self.validation_stats["missing_masks"] += 1

                raise FileNotFoundError(
                    f"Missing mask for frame {frame_id}:\n"
                    f"Image: {image_path}\n"
                    f"Expected mask: {mask_path}"
                )

            self.samples.append(
                {
                    "image": image_path,
                    "mask": mask_path,
                    "procedure": procedure,
                    "video": video,
                    "clip": clip,
                    "frame": frame_id,
                }
            )

            self.validation_stats["valid_samples"] += 1

    # ------------------------------------------------------------------
    # FRAME ID
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_id_from_filename(path: Path) -> int:
        """
        Convert:

            frame_004031.jpg

        into:

            4031
        """

        stem = path.stem

        if not stem.startswith("frame_"):
            raise ValueError(
                f"Unexpected frame filename: {path.name}"
            )

        frame_string = stem[len("frame_"):]

        try:
            return int(frame_string)
        except ValueError as e:
            raise ValueError(
                f"Could not extract frame ID from {path.name}"
            ) from e

    # ------------------------------------------------------------------
    # PYTORCH DATASET API
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, index):

        sample = self.samples[index]

        image = self._load_image(
            sample["image"]
        )

        mask = self._load_mask(
            sample["mask"]
        )

        return {
            "image": image,
            "mask": mask,

            # Metadata
            "procedure": sample["procedure"],
            "video": sample["video"],
            "clip": sample["clip"],
            "frame": sample["frame"],
        }

    # ------------------------------------------------------------------
    # IMAGE
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:

        image = Image.open(path).convert("RGB")

        return np.asarray(
            image,
            dtype=np.uint8,
        )

    # ------------------------------------------------------------------
    # MASK
    # ------------------------------------------------------------------

    @classmethod
    def _load_mask(cls, path: Path) -> np.ndarray:

        mask = Image.open(path).convert("L")

        mask = np.asarray(
            mask,
            dtype=np.uint8,
        )

        # Original ATLAS labels should be 0..46
        unique_labels = np.unique(mask)

        invalid_labels = unique_labels[
            unique_labels >= cls.NUM_ORIGINAL_CLASSES
        ]

        if len(invalid_labels) > 0:

            raise ValueError(
                f"Invalid labels found in mask:\n"
                f"{path}\n"
                f"Labels: {invalid_labels.tolist()}\n"
                f"Expected range: "
                f"0-{cls.NUM_ORIGINAL_CLASSES - 1}"
            )

        # 47 -> 30 classes
        mask = cls.ORIGINAL_TO_TRAIN_ID[mask]

        return mask.astype(np.uint8)

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------

    def _print_statistics(self):

        stats = self.validation_stats

        print("\n" + "=" * 60)
        print("ATLAS-120K DATASET")
        print("=" * 60)

        print(f"Root:              {self.root}")
        print(f"Videos:            {stats['videos']:,}")
        print(f"Clips:             {stats['clips']:,}")
        print(f"Images:            {stats['images']:,}")
        print(f"Valid samples:     {stats['valid_samples']:,}")
        print(f"Missing images:    {stats['missing_images']:,}")
        print(f"Missing masks:     {stats['missing_masks']:,}")
        print(f"JSON mismatches:   {stats['json_mismatches']:,}")

        print("=" * 60)