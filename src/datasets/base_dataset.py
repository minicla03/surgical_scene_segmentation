from abc import ABC, abstractmethod


class BaseSegmentationDataset(ABC):

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
        """
        Return:
            {
                "image": ...,
                "mask": ...,
            }
        """
        raise NotImplementedError

    def __getitem__(self, index):

        sample = self._load_sample(index)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample