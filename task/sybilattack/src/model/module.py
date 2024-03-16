import os
from typing import Literal

import lightning as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F


class MappingModel(pl.LightningModule):
    def __init__(self, predict_prefix, transform_type=Literal["affine", "binary"], embedding_size: int = 384, same=False):
        super().__init__()

        self.same = same
        self.transform_type = transform_type
        self.predict_prefix = predict_prefix
        self.models = nn.Linear(embedding_size, embedding_size)

    def training_step(self, batch) -> STEP_OUTPUT:
        source, target = batch

        output = self.models(source)

        loss = F.mse_loss(output, target) if self.transform_type == "affine" else F.binary_cross_entropy_with_logits(output, target)

        return loss

    def on_predict_start(self) -> None:
        super().on_predict_start()

        self.predictions = {}

    def on_predict_end(self) -> None:
        super().on_predict_end()

        print(f"Saving data to {self.predict_prefix}")

        torch.save(self.predictions, os.path.join("data", f"{self.predict_prefix}.pt"))

    def predict_step(self, source) -> STEP_OUTPUT:
        prediction = source if self.same else self.models(source)

        if self.transform_type == "binary":
            prediction = F.sigmoid(prediction)

        for index, pred in zip(source.index, prediction):
            self.predictions[index.int().item()] = pred.tolist()
