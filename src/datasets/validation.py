from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# DATASET METADATA
# ============================================================

def get_samples(dataset: Any) -> list[dict[str, Any]]:
    """Return dataset sample metadata."""
    samples = getattr(dataset, "samples", None)

    if samples is None:
        raise AttributeError(
            f"{type(dataset).__name__} does not expose a 'samples' attribute."
        )

    return list(samples)


def get_class_names(dataset: Any) -> list[str]:
    """Return semantic class names."""
    class_names = getattr(dataset, "class_names", None)

    if class_names is None:
        class_names = getattr(dataset, "CLASS_NAMES", None)

    if class_names is None:
        raise AttributeError(
            f"{type(dataset).__name__} does not expose 'class_names' or 'CLASS_NAMES'."
        )

    return list(class_names)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors or arrays to NumPy."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()

    return np.asarray(value)


def _load_raw_sample(dataset: Any, index: int) -> dict[str, Any]:
    """
    Load the original sample without runtime transformations.
    This prevents random augmentations from affecting the statistics.
    """
    loader = getattr(dataset, "_load_sample", None)

    if callable(loader):
        return loader(index)

    return dataset[index]


# ============================================================
# CONFIGURATION
# ============================================================

def _get_data_config(config: Any) -> Any:
    """Return the data section from a Hydra/dict configuration."""
    if config is None:
        return None

    if isinstance(config, Mapping):
        return config.get("data", config)

    return getattr(config, "data", config)


def _get_split_config(config: Any) -> Any:
    """Return data.split from configuration."""
    data_config = _get_data_config(config)

    if data_config is None:
        return None

    if isinstance(data_config, Mapping):
        return data_config.get("split")

    return getattr(data_config, "split", None)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    """Read a configuration value."""
    if isinstance(config, Mapping):
        return config.get(key, default)

    if config is not None:
        return getattr(config, key, default)

    return default


# ============================================================
# EXISTING DATASET SPLITS
# ============================================================

def _indices_by_split(dataset: Any) -> dict[str, list[int]]:
    """Return indices grouped by split metadata."""
    splits: dict[str, list[int]] = {}

    for index, sample in enumerate(get_samples(dataset)):
        split = sample.get("split")
        if split is not None:
            splits.setdefault(str(split), []).append(index)

    return splits


# ============================================================
# GROUPING
# ============================================================

def _group_indices(samples: list[dict[str, Any]], group_key: str) -> dict[str, list[int]]:
    """Group sample indices by video/group."""
    groups: dict[str, list[int]] = {}

    for index, sample in enumerate(samples):
        if group_key not in sample:
            raise KeyError(f"Sample {index} does not contain '{group_key}'.")

        group_id = sample[group_key]
        if group_id is None:
            raise ValueError(f"Sample {index} has an invalid '{group_key}' value.")

        groups.setdefault(str(group_id), []).append(index)

    return groups


def _group_class_presence(
    dataset: Any, groups: dict[str, list[int]], class_count: int
) -> np.ndarray:
    """
    Compute class presence for every group.
    A class is considered present in a group if it appears
    in at least one mask belonging to that group.
    """
    group_ids = list(groups.keys())
    labels = np.zeros((len(group_ids), class_count), dtype=np.int8)

    for group_index, group_id in enumerate(group_ids):
        for sample_index in groups[group_id]:
            sample = _load_raw_sample(dataset, sample_index)
            mask = _to_numpy(sample["mask"])

            if mask.ndim != 2:
                raise ValueError(f"Expected a 2D mask, got {mask.shape}.")

            present_classes = np.unique(mask)
            present_classes = present_classes[
                (present_classes >= 0) & (present_classes < class_count)
            ].astype(int)

            labels[group_index, present_classes] = 1

    return labels


# ============================================================
# SPLITTING
# ============================================================

def _partition_groups(
    group_ids: np.ndarray,
    labels: np.ndarray | None,
    test_size: float,
    seed: int,
    stratified: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Split groups into two subsets."""
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    if stratified:
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        except ImportError as exc:
            raise ImportError(
                "Install iterative-stratification with:\n"
                "pip install iterative-stratification"
            ) from exc

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        train_idx, test_idx = next(splitter.split(group_ids, labels))
    else:
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            np.arange(len(group_ids)),
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )

    return group_ids[train_idx], group_ids[test_idx]


def generate_percentage_splits(dataset: Any, split_config: Any) -> dict[str, list[int]]:
    """
    Generate train/validation/test splits.
    Splitting is performed at group level.
    If stratified=True, multilabel stratification uses the
    semantic classes present in each group.
    """
    samples = get_samples(dataset)
    
    group_by = _config_get(split_config, "group_by", "video")
    seed = int(_config_get(split_config, "seed", 42))
    stratified = bool(_config_get(split_config, "stratified", False))
    train_ratio = float(_config_get(split_config, "train_ratio", 0.7))
    val_ratio = float(_config_get(split_config, "val_ratio", 0.3))
    test_ratio = float(_config_get(split_config, "test_ratio", 0.0))

    ratios = {"train": train_ratio, "validation": val_ratio, "test": test_ratio}

    if any(ratio < 0 or ratio > 1 for ratio in ratios.values()):
        raise ValueError("Split ratios must be between 0 and 1.")

    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0. Got {sum(ratios.values()):.4f}.")

    groups = _group_indices(samples, str(group_by))
    group_ids = np.asarray(list(groups.keys()), dtype=object)

    if len(group_ids) < 2:
        raise ValueError("At least two groups are required.")

    # --------------------------------------------------------
    # Class presence at group level
    # --------------------------------------------------------
    labels = None
    if stratified:
        labels = _group_class_presence(
            dataset=dataset,
            groups=groups,
            class_count=len(get_class_names(dataset)),
        )

    # --------------------------------------------------------
    # Test split
    # --------------------------------------------------------
    if test_ratio > 0:
        remaining_groups, test_groups = _partition_groups(
            group_ids=group_ids,
            labels=labels,
            test_size=test_ratio,
            seed=seed,
            stratified=stratified,
        )
        if labels is not None:
            group_to_idx = {group_id: idx for idx, group_id in enumerate(group_ids)}
            remaining_labels = np.asarray(
                [labels[group_to_idx[group_id]] for group_id in remaining_groups]
            )
        else:
            remaining_labels = None
    else:
        remaining_groups = group_ids
        test_groups = np.asarray([], dtype=object)
        remaining_labels = labels

    # --------------------------------------------------------
    # Train / validation
    # --------------------------------------------------------
    if val_ratio == 0:
        train_groups = remaining_groups
        val_groups = np.asarray([], dtype=object)
    elif train_ratio == 0:
        train_groups = np.asarray([], dtype=object)
        val_groups = remaining_groups
    else:
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_groups, val_groups = _partition_groups(
            group_ids=remaining_groups,
            labels=remaining_labels,
            test_size=relative_val_ratio,
            seed=seed,
            stratified=stratified,
        )

    # --------------------------------------------------------
    # Convert groups to sample indices
    # --------------------------------------------------------
    def expand_groups(selected_groups: Iterable[Any]) -> list[int]:
        return [
            index
            for group_id in selected_groups
            for index in groups[str(group_id)]
        ]

    return {
        "train": expand_groups(train_groups),
        "validation": expand_groups(val_groups),
        "test": expand_groups(test_groups),
    }


# ============================================================
# SPLIT RESOLUTION
# ============================================================

def resolve_splits(dataset: Any, config: Any | None = None) -> dict[str, list[int]]:
    """
    Resolve dataset splits.
    Priority:
        1. Existing split metadata.
        2. Splits generated from configuration.
    """
    metadata_splits = _indices_by_split(dataset)
    if metadata_splits:
        return metadata_splits

    if config is None:
        raise ValueError(
            f"{type(dataset).__name__} has no split metadata and no configuration was provided."
        )

    split_config = _get_split_config(config)
    if split_config is None:
        raise ValueError("Missing 'data.split' configuration.")

    strategy = _config_get(split_config, "strategy")
    if strategy != "percentage":
        raise ValueError(f"Unsupported split strategy: {strategy}")

    return generate_percentage_splits(dataset=dataset, split_config=split_config)


# ============================================================
# CLASS STATISTICS
# ============================================================

def _resolve_group_key(samples: list[dict[str, Any]], group_key: str) -> str:
    """Return group_key, falling back to 'surgery_id' (e.g. DSAD) if missing."""
    if samples and group_key not in samples[0] and "surgery_id" in samples[0]:
        return "surgery_id"
    return group_key


def compute_sample_counts(
    dataset: Any, indices: Iterable[int]
) -> dict[int, tuple[np.ndarray, int]]:
    """
    Load every requested mask exactly once.

    Returns {sample_index: (pixels_per_class, ignored_pixels)} where
    ignored_pixels are pixels whose label is outside [0, class_count).
    A class is present in a sample iff pixels_per_class[class] > 0.
    """
    class_count = len(get_class_names(dataset))
    results: dict[int, tuple[np.ndarray, int]] = {}

    for index in indices:
        index = int(index)
        if index in results:
            continue

        mask = _to_numpy(_load_raw_sample(dataset, index)["mask"])
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got {mask.shape} at index {index}.")

        flat = mask.ravel().astype(np.int64)
        valid = (flat >= 0) & (flat < class_count)
        counts = np.bincount(flat[valid], minlength=class_count).astype(np.int64)
        results[index] = (counts, int(flat.size - valid.sum()))

    return results


def _statistics_frame(
    class_names: list[str],
    sample_counts: list[tuple[np.ndarray, int]],
    **labels: Any,
) -> pd.DataFrame:
    """
    Aggregate per-sample counts into one row per class.
    pixel_percentage is computed over VALID pixels (labels in range),
    so it sums to 100 per table; ignored pixels are reported separately.
    """
    class_count = len(class_names)
    pixel_counts = np.zeros(class_count, dtype=np.int64)
    sample_presence = np.zeros(class_count, dtype=np.int64)
    ignored_pixels = 0

    for counts, ignored in sample_counts:
        pixel_counts += counts
        sample_presence += counts > 0
        ignored_pixels += ignored

    total_samples = len(sample_counts)
    valid_pixels = int(pixel_counts.sum())

    rows = []
    for class_id, class_name in enumerate(class_names):
        rows.append({
            **labels,
            "class_id": class_id,
            "class": class_name,
            "samples": int(sample_presence[class_id]),
            "sample_percentage": (
                100.0 * sample_presence[class_id] / total_samples if total_samples else 0.0
            ),
            "pixels": int(pixel_counts[class_id]),
            "pixel_percentage": (
                100.0 * pixel_counts[class_id] / valid_pixels if valid_pixels else 0.0
            ),
            "total_samples": total_samples,
            "valid_pixels": valid_pixels,
            "ignored_pixels": int(ignored_pixels),
        })

    return pd.DataFrame(rows)


def class_pixel_statistics(
    dataset: Any,
    indices: Iterable[int],
    split_name: str,
    cache: dict[int, tuple[np.ndarray, int]] | None = None,
) -> pd.DataFrame:
    """Class statistics (sample presence + pixels) over the given samples."""
    indices = [int(i) for i in indices]
    if cache is None:
        cache = compute_sample_counts(dataset, indices)

    return _statistics_frame(
        get_class_names(dataset), [cache[i] for i in indices], split=split_name
    )


def class_statistics_by_video(
    dataset: Any,
    indices: Iterable[int] | None = None,
    group_key: str = "video",
    split_name: str | None = None,
    cache: dict[int, tuple[np.ndarray, int]] | None = None,
) -> pd.DataFrame:
    """Same statistics computed separately for every video (group)."""
    samples = get_samples(dataset)
    group_key = _resolve_group_key(samples, group_key)
    indices = list(range(len(samples))) if indices is None else [int(i) for i in indices]
    if cache is None:
        cache = compute_sample_counts(dataset, indices)

    groups: dict[str, list[int]] = {}
    for index in indices:
        groups.setdefault(str(samples[index][group_key]), []).append(index)

    class_names = get_class_names(dataset)
    frames = []
    for video, video_indices in sorted(groups.items()):
        labels: dict[str, Any] = {}
        if split_name is not None:
            labels["split"] = split_name
        labels["video"] = video
        frames.append(
            _statistics_frame(class_names, [cache[i] for i in video_indices], **labels)
        )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def class_statistics_global(
    dataset: Any,
    indices: Iterable[int] | None = None,
    cache: dict[int, tuple[np.ndarray, int]] | None = None,
) -> pd.DataFrame:
    """Same statistics over the whole dataset (all samples, regardless of split)."""
    indices = (
        list(range(len(get_samples(dataset))))
        if indices is None
        else [int(i) for i in indices]
    )
    if cache is None:
        cache = compute_sample_counts(dataset, indices)

    return _statistics_frame(
        get_class_names(dataset), [cache[i] for i in indices], scope="global"
    )


def class_video_presence(
    dataset: Any,
    indices: Iterable[int] | None = None,
    group_key: str = "video",
    cache: dict[int, tuple[np.ndarray, int]] | None = None,
    **labels: Any,
) -> pd.DataFrame:
    """
    For every class: in how many videos it appears and in which ones.

    Columns:
        videos             number of videos containing the class (>= 1 pixel)
        total_videos       number of videos considered
        video_percentage   videos / total_videos * 100
        video_list         sorted list of the videos containing the class
        frames_per_video   {video: frames containing the class}
        top_video          video holding the largest share of the class pixels
        top_video_pixel_share  % of the class pixels belonging to top_video
                               (100 -> the class lives in a single video)
    """
    samples = get_samples(dataset)
    group_key = _resolve_group_key(samples, group_key)
    indices = (
        list(range(len(samples))) if indices is None else [int(i) for i in indices]
    )
    if cache is None:
        cache = compute_sample_counts(dataset, indices)

    class_names = get_class_names(dataset)
    class_count = len(class_names)

    groups: dict[str, list[int]] = {}
    for index in indices:
        groups.setdefault(str(samples[index][group_key]), []).append(index)

    video_pixels: dict[str, np.ndarray] = {}
    video_frames: dict[str, np.ndarray] = {}
    for video, video_indices in groups.items():
        video_pixels[video] = np.sum([cache[i][0] for i in video_indices], axis=0)
        video_frames[video] = np.sum([cache[i][0] > 0 for i in video_indices], axis=0)

    rows = []
    for class_id, class_name in enumerate(class_names):
        present = sorted(v for v in groups if video_pixels[v][class_id] > 0)
        class_pixels = sum(int(video_pixels[v][class_id]) for v in present)
        if present:
            top = max(present, key=lambda v: video_pixels[v][class_id])
            top_share = 100.0 * video_pixels[top][class_id] / class_pixels
        else:
            top, top_share = None, 0.0

        rows.append({
            **labels,
            "class_id": class_id,
            "class": class_name,
            "videos": len(present),
            "total_videos": len(groups),
            "video_percentage": 100.0 * len(present) / len(groups) if groups else 0.0,
            "video_list": present,
            "frames_per_video": {v: int(video_frames[v][class_id]) for v in present},
            "top_video": top,
            "top_video_pixel_share": top_share,
        })

    return pd.DataFrame(rows)


def class_video_table(
    dataset: Any,
    indices: Iterable[int] | None = None,
    group_key: str = "video",
    cache: dict[int, tuple[np.ndarray, int]] | None = None,
    split_by_video: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Video x class table with frames and pixels for every class.

    Rows    = videos (index "video"; optional first level "split" when
              split_by_video is given, e.g. {"video01": "train", ...}).
    Columns = MultiIndex (class, metric) with metric in {"frames", "pixels"}:
              frames = number of frames of the video containing the class,
              pixels = total pixels of the class in the video.
    """
    samples = get_samples(dataset)
    group_key = _resolve_group_key(samples, group_key)
    indices = (
        list(range(len(samples))) if indices is None else [int(i) for i in indices]
    )
    if cache is None:
        cache = compute_sample_counts(dataset, indices)

    groups: dict[str, list[int]] = {}
    for index in indices:
        groups.setdefault(str(samples[index][group_key]), []).append(index)

    class_names = get_class_names(dataset)
    columns = pd.MultiIndex.from_product(
        [class_names, ["frames", "pixels"]], names=["class", "metric"]
    )

    videos = sorted(groups)
    if split_by_video is not None:
        videos = sorted(videos, key=lambda v: (split_by_video.get(v, ""), v))

    rows = []
    for video in videos:
        counts = np.stack([cache[i][0] for i in groups[video]])  # (frames, classes)
        frames = (counts > 0).sum(axis=0)
        pixels = counts.sum(axis=0)
        row = np.empty(2 * len(class_names), dtype=np.int64)
        row[0::2], row[1::2] = frames, pixels
        rows.append(row)

    data = np.vstack(rows) if rows else np.empty((0, len(columns)), dtype=np.int64)
    if split_by_video is not None:
        index = pd.MultiIndex.from_tuples(
            [(split_by_video.get(v, "?"), v) for v in videos], names=["split", "video"]
        )
    else:
        index = pd.Index(videos, name="video")
    return pd.DataFrame(data, index=index, columns=columns)


def class_video_matrix(
    dataset: Any,
    indices: Iterable[int] | None = None,
    group_key: str = "video",
    cache: dict[int, tuple[np.ndarray, int]] | None = None,
) -> pd.DataFrame:
    """Frames containing each class only (rows = videos, columns = classes)."""
    table = class_video_table(dataset, indices, group_key, cache)
    return table.xs("frames", axis=1, level="metric")


def flag_classes(
    global_video_presence: pd.DataFrame,
    rare_video_threshold: int = 3,
    concentration_threshold: float = 50.0,
) -> list[str]:
    """Classes present in <= rare_video_threshold videos or concentrated in one video."""
    p = global_video_presence
    mask = (p["videos"] <= rare_video_threshold) | (
        (p["top_video_pixel_share"] >= concentration_threshold) & (p["videos"] > 0)
    )
    return p.loc[mask, "class"].tolist()


# ============================================================
# MAIN VALIDATOR
# ============================================================

def validate_dataset(
    dataset: Any, config: Any | None = None, group_key: str = "video"
) -> dict[str, Any]:
    """
    Compute dataset class distribution on the COMPLETE dataset
    (no sampling). Each mask is read only once.

    Returns:
        {
            "splits": {split: [indices]},
            "split_summary": samples per split,
            "class_statistics": per split,
            "video_statistics": per video (and split),
            "global_statistics": whole dataset,
            "video_presence": videos containing each class, per split,
            "global_video_presence": same, over the whole dataset,
            "video_class_table": video x class table (frames + pixels per class)
        }
    """
    splits = resolve_splits(dataset, config)

    all_indices = sorted({i for idx in splits.values() for i in idx})
    cache = compute_sample_counts(dataset, all_indices)

    split_summary = pd.DataFrame([
        {"split": name, "samples": len(idx)} for name, idx in splits.items()
    ])

    class_statistics = pd.concat(
        [
            class_pixel_statistics(dataset, idx, name, cache=cache)
            for name, idx in splits.items()
        ],
        ignore_index=True,
    )

    video_statistics = pd.concat(
        [
            class_statistics_by_video(
                dataset, idx, group_key=group_key, split_name=name, cache=cache
            )
            for name, idx in splits.items()
            if len(idx)
        ],
        ignore_index=True,
    )

    global_statistics = class_statistics_global(dataset, all_indices, cache=cache)

    video_presence = pd.concat(
        [
            class_video_presence(
                dataset, idx, group_key=group_key, cache=cache, split=name
            )
            for name, idx in splits.items()
            if len(idx)
        ],
        ignore_index=True,
    )
    global_video_presence = class_video_presence(
        dataset, all_indices, group_key=group_key, cache=cache, scope="global"
    )

    all_samples = get_samples(dataset)
    video_col = _resolve_group_key(all_samples, group_key)
    split_by_video = {
        str(all_samples[i][video_col]): name
        for name, idx in splits.items()
        for i in idx
    }
    video_class_table = class_video_table(
        dataset, all_indices, group_key=group_key, cache=cache,
        split_by_video=split_by_video,
    )

    return {
        "splits": splits,
        "split_summary": split_summary,
        "class_statistics": class_statistics,
        "video_statistics": video_statistics,
        "global_statistics": global_statistics,
        "video_presence": video_presence,
        "global_video_presence": global_video_presence,
        "video_class_table": video_class_table,
    }
