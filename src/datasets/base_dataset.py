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