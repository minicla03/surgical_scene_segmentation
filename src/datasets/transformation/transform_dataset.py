from torch.utils.data import Dataset


class TransformDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample