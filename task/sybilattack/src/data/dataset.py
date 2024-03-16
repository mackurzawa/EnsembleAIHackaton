from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .embedding import Embedding


class TransformDataset(Dataset):
    def __init__(self, transform: int, transform_type: Literal["affine", "binary"], common: bool = True):
        self.data = pd.read_csv(
            f'data/common_embeddings_{transform_type}.csv' if common else f"data/other_embeddings_{transform_type}.csv")
        self.data = self.data[self.data.transformation == transform].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        embedding = list(map(float, self.data.embedding[index][1:-1].split(", ")))

        return Embedding(embedding, self.data.img_id[index])


class SybilattackDataset(Dataset):
    def __init__(self, source_transform: int, target_transform: int, transform_type: Literal["affine", "binary"]):
        self.source = TransformDataset(source_transform, transform_type)
        self.target = TransformDataset(target_transform, transform_type)

        assert len(self.source) == len(self.target)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.source[index], self.target[index]
