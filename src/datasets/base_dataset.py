from abc import ABC, abstractmethod

import numpy as np


class BaseSegmentationDataset(ABC):
    CLASS_NAMES: list[str] = []

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    @abstractmethod
    def __len__(self):
        """Return the number of samples."""
        raise NotImplementedError

    @abstractmethod
    def _load_sample(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        sample = self._load_sample(index)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def log_split(self, train_groups: list, val_groups: list, test_groups: list, group_key: str) -> None:
        """Print split groups and number of samples per split."""
        
        def get_sample_count(groups):
            groups = set(groups)
            return sum(1 for sample in self.samples if sample[group_key] in groups)

        train_count = get_sample_count(train_groups)
        val_count = get_sample_count(val_groups)
        test_count = get_sample_count(test_groups)

        print("\n" + "=" * 60)
        print("DATASET SPLIT")
        print("=" * 60)

        print("\nTrain:")
        print(f"  Groups : {sorted(train_groups)}")
        print(f"  Frames : {train_count}")

        print("\nValidation:")
        print(f"  Groups : {sorted(val_groups)}")
        print(f"  Frames : {val_count}")

        if test_groups:
            print("\nTest:")
            print(f"  Groups : {sorted(test_groups)}")
            print(f"  Frames : {test_count}")

        print("\nTotal:")
        print(f"  Groups : {len(set(train_groups + val_groups + test_groups))}")
        print(f"  Frames : {len(self.samples)}")
        print("=" * 60)

    def create_kfold_split(
        self,
        group_by: str = "video",
        n_folds: int = 4,
        fold_index: int = 0,
        test_ratio: float = 0.0,
        seed: int = 42,
        balance: bool = True,
        n_restarts: int = 30,
        balance_min_pixel_pct: float = 1.0,
    ):
        """Group-level K-fold split (see ``src.datasets.kfold``).

        Returns ``(train, validation, test)`` subsets of ``fold_index``; ``test`` is
        None when ``test_ratio`` is 0. The full plan is kept in ``self.kfold_plan``
        (evaluable classes per fold, pinned videos, coverage tables).
        """
        from torch.utils.data import Subset

        from src.datasets.kfold import resolve_kfold_splits

        indices, plan = resolve_kfold_splits(
            self,
            {
                "group_by": group_by,
                "n_folds": n_folds,
                "fold_index": fold_index,
                "test_ratio": test_ratio,
                "seed": seed,
                "balance": balance,
                "n_restarts": n_restarts,
                "balance_min_pixel_pct": balance_min_pixel_pct,
            },
        )
        self.kfold_plan = plan
        self.kfold_fold_index = int(fold_index)
        self._print_kfold_split(plan, int(fold_index))

        test = Subset(self, indices["test"]) if indices["test"] else None
        return Subset(self, indices["train"]), Subset(self, indices["validation"]), test

    @staticmethod
    def _print_kfold_split(plan, fold_index: int) -> None:
        groups = plan.groups_for(fold_index)
        row = plan.fold_table().iloc[fold_index]
        evaluable = plan.evaluable_classes(fold_index)

        print("\n" + "=" * 60)
        print(f"K-FOLD SPLIT (fold {fold_index + 1}/{plan.n_folds}, seed={plan.seed})")
        print("=" * 60)
        for split in ("train", "validation", "test"):
            if split == "test" and not groups["test"]:
                continue
            print(f"\n{split.capitalize()}:")
            print(f"  Groups : {groups[split]}")
            print(f"  Frames : {int(row[f'{split}_frames'])}")
        if plan.pinned:
            pinned = {
                plan.group_ids[g]: [plan.class_names[c] for c in classes]
                for g, classes in plan.pinned_reasons.items()
            }
            print(f"\nPinned to train (hold single-video classes): {pinned}")
        print(
            "\nEvaluated classes   : "
            f"{[n for n, ok in zip(plan.class_names, evaluable) if ok]}"
        )
        print(
            "NaN (not evaluable) : "
            f"{[n for n, ok in zip(plan.class_names, evaluable) if not ok]}"
        )
        print("=" * 60)

    def get_class_distribution(self, indices: list[int]) -> dict[int, dict[str, int | float]]:
        """Compute pixel distribution for each class."""

        class_pixel_counts = {}
        total_pixels = 0

        for index in indices:
            sample = self._load_sample(index)
            mask = np.asarray(sample["mask"])

            classes, counts = np.unique(mask, return_counts=True)

            for class_id, count in zip(classes, counts):
                class_id = int(class_id)
                count = int(count)

                class_pixel_counts[class_id] = class_pixel_counts.get(class_id, 0) + count
                total_pixels += count

        if total_pixels == 0:
            return {}

        return {
            class_id: {
                "pixels": count,
                "percentage": 100.0 * count / total_pixels,
            }
            for class_id, count in sorted(class_pixel_counts.items())
        }

    def log_class_distribution(self, train_indices: list[int], val_indices: list[int], test_indices: list[int]) -> None:
        """Print class pixel distribution for each split."""
        
        splits = {
            "Train": train_indices,
            "Validation": val_indices,
            "Test": test_indices,
        }

        distributions = {}

        for split_name, indices in splits.items():
            if indices:
                distributions[split_name] = self.get_class_distribution(indices)

        if not distributions:
            return

        class_ids = sorted({
            class_id
            for distribution in distributions.values()
            for class_id in distribution
        })

        print("\n" + "=" * 80)
        print("CLASS DISTRIBUTION")
        print("=" * 80)
        
        print(f"\n{'Class':25s}{'Train':>15s}{'Validation':>15s}{'Test':>15s}")
        print("-" * 80)

        for class_id in class_ids:
            if class_id < len(self.CLASS_NAMES):
                class_name = self.CLASS_NAMES[class_id]
            else:
                class_name = f"class_{class_id}"

            values = []

            for split_name in splits:
                if split_name not in distributions:
                    values.append("-")
                    continue

                class_stats = distributions[split_name].get(
                    class_id, {"pixels": 0, "percentage": 0.0}
                )
                values.append(f"{class_stats['percentage']:.2f}%")

            print(f"{class_name[:25]:25s}{values[0]:>15s}{values[1]:>15s}{values[2]:>15s}")

        print("=" * 80)

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_NAMES)

    @property
    def class_names(self) -> list[str]:
        return self.CLASS_NAMES