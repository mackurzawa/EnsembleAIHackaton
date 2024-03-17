from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .embedding import Embedding


class TransformDataset(Dataset):
    def __init__(self, path: str):
        self.data = torch.tensor(np.load(path)["representations"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class SybilattackDataset(Dataset):
    def __init__(self, source_path):
        self.source = TransformDataset(source_path)
        self.target = TransformDataset("model/defense_4.npz")

        assert len(self.source) == len(self.target)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.source[index], self.target[index]
