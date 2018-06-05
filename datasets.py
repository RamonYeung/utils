from torch.utils.data import Dataset


class DatasetName(Dataset):
    def __init__(self):
        super().__init__()
        self._len = 0

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        return None


