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
    def __init__(self, source_path: str, batch_size: int  = 512):
        super().__init__()
        self.source_path = source_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit":
            dataset = SybilattackDataset(source_path=self.source_path)
            
            self.train, self.test = torch.utils.data.random_split(dataset, [0.1, 0.9])

    def train_dataloader(self):
        return DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.test, batch_size=self.batch_size)