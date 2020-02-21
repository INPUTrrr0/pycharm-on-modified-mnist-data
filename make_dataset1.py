import torch
import numpy as np
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, data, target=None, transform=None):
        self.data = data
        self.transform = transform
        if target is not None:
            self.target = target
        else:
            self.target = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if idx >= self.__len__():
            toRet = (None, None)
            return toRet

        image = np.uint8(self.data[idx])
        image = np.expand_dims(image, -1)
        image = self.transform(image)
        if self.target is not None:
            target = self.target[idx]
        else:
            target = torch.zeros((1,))

        toRet = (image, target)

        return toRet


