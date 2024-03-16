import os

import numpy as np
import pandas as pd
import torch

from taskdataset import TaskDataset

result = {}

TRANSFORMATION = "binary" # "affine"

for i in range(15):
    if i == 0:
        data = pd.read_csv(f'data/common_embeddings_{TRANSFORMATION}.csv')
        partial_result = {row.img_id: list(map(float, row.embedding[1:-1].split(", "))) for _, row in data.iterrows()}
    else:
        os.system(f"/home/keren/EnsembleAIHackaton/.venv/bin/python -m src.experiment.train --data.source_transform {i} --data.target_transform 1 --data.batch_size 512 --data.transform_type {TRANSFORMATION} --optimizer AdamW --trainer.max_epoch 100")

        partial_result = torch.load(f"data/{i}_{TRANSFORMATION}.pt")
    result.update(partial_result)

dataset = torch.load("data/SybilAttack.pt")

indices = np.array([i[0] for i in dataset if i[0] in result])
predictions = np.array([result[i[0]] for i in dataset if i[0] in result])

print(indices.shape)
print(predictions.shape)
#
with open(os.path.join("data", f"{TRANSFORMATION}.npz"), "wb") as f:
    np.savez(f, ids=indices, representations=predictions)