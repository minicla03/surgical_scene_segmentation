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

def class_pixel_statistics(
    dataset: Any, indices: Iterable[int], split_name: str
) -> pd.DataFrame:
    """
    Compute class statistics using ALL samples in a split.
    For every class:
        - number of samples containing the class;
        - percentage of samples containing the class;
        - number of pixels;
        - percentage of pixels.
    """
    class_names = get_class_names(dataset)
    class_count = len(class_names)

    pixel_counts = np.zeros(class_count, dtype=np.int64)
    sample_presence = np.zeros(class_count, dtype=np.int64)

    indices = list(indices)
    total_samples = len(indices)
    total_pixels = 0

    for index in indices:
        sample = _load_raw_sample(dataset, int(index))
        mask = _to_numpy(sample["mask"])

        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got {mask.shape} at index {index}.")

        total_pixels += mask.size
        labels, counts = np.unique(mask, return_counts=True)

        for label, count in zip(labels, counts):
            label = int(label)
            if not 0 <= label < class_count:
                continue

            pixel_counts[label] += int(count)
            sample_presence[label] += 1

    rows = []
    for class_id, class_name in enumerate(class_names):
        rows.append({
            "split": split_name,
            "class_id": class_id,
            "class": class_name,
            "samples": int(sample_presence[class_id]),
            "sample_percentage": (
                100.0 * sample_presence[class_id] / total_samples if total_samples else 0.0
            ),
            "pixels": int(pixel_counts[class_id]),
            "pixel_percentage": (
                100.0 * pixel_counts[class_id] / total_pixels if total_pixels else 0.0
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN VALIDATOR
# ============================================================

def validate_dataset(dataset: Any, config: Any | None = None) -> dict[str, Any]:
    """
    Compute dataset class distribution.
    
    IMPORTANT:
        Statistics are computed on the COMPLETE dataset split.
    
    No sampling or max_samples limit is applied.
    
    Returns:
        {
            "splits": ...,
            "split_summary": ...,
            "class_statistics": ...
        }
    """
    splits = resolve_splits(dataset, config)

    # --------------------------------------------------------
    # Number of samples per split
    # --------------------------------------------------------
    split_summary = pd.DataFrame([
        {"split": split_name, "samples": len(indices)}
        for split_name, indices in splits.items()
    ])

    # --------------------------------------------------------
    # Class statistics
    # --------------------------------------------------------
    statistics = [
        class_pixel_statistics(
            dataset=dataset, indices=indices, split_name=split_name
        )
        for split_name, indices in splits.items()
    ]

    class_statistics = pd.concat(statistics, ignore_index=True)

    return {
        "splits": splits,
        "split_summary": split_summary,
        "class_statistics": class_statistics,
    }