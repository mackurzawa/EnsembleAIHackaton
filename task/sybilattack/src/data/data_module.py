from typing import Literal

import lightning as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

import torch
from torch.utils.data import DataLoader

from .dataset import SybilattackDataset, TransformDataset
from .embedding import Embedding


def _collate_fn(batch):
    data = torch.stack(batch)
    index = torch.Tensor([i.index for i in batch])

    return Embedding(data, index=index)


class DataModule(pl.LightningDataModule):
    def __init__(self, source_transform: int, target_transform: int, batch_size: int,
                 transform_type: Literal["affine", "binary"]):
        super().__init__()
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.transform_type = transform_type

    def setup(self, stage=None):
        if stage == "fit":
            self.train = SybilattackDataset(source_transform=self.source_transform,
                                            target_transform=self.target_transform,
                                            transform_type=self.transform_type)

        if stage == "predict":
            self.predict = TransformDataset(self.source_transform, self.transform_type, common=False)

    def train_dataloader(self):
        return DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.predict, batch_size=self.batch_size, shuffle=False, collate_fn=_collate_fn)
