import os

import lightning as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F


class MappingModel(pl.LightningModule):
    def __init__(self, embedding_size: int = 192, same=False):
        super().__init__()
        
        self.same = same
        self.models = nn.Linear(embedding_size, embedding_size)

    def training_step(self, batch) -> STEP_OUTPUT:
        source, target = batch

        output = self.models(source)

        loss = F.mse_loss(output, target)

        return loss
    
    def test_step(self, batch):
        source, target = batch

        output = source if self.same else self.models(source)

        self.log("test", F.cosine_similarity(output, target).mean())
