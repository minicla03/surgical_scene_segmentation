from abc import ABC, abstractmethod


class BaseSegmentationDataset(ABC):

    CLASS_NAMES: list[str] = []
    def __init__(
        self,
        root,
        transform=None,
    ):
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

    def log_split(
        self,
        train_groups: list,
        val_groups: list,
        test_groups: list,
        group_key: str,
    ) -> None:
        """Print split groups and number of samples per split."""

        def get_sample_count(groups):
            groups = set(groups)

            return sum(
                1
                for sample in self.samples
                if sample[group_key] in groups
            )

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

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_NAMES)

    @property
    def class_names(self) -> list[str]:
        return self.CLASS_NAMES