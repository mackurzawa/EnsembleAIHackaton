import torch
import numpy as np

from .stealing_requests import model_stealing

clusters = {i: np.load(f'data/centr{i}_quering_queue.npy') for i in range(3)}

cluster_order = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
cluster_order = [i[0] for i in cluster_order]

for cluster_index in cluster_order:
    cluster = clusters[cluster_index]

    for i in cluster:
        representations = torch.tensor([model_stealing(f"data/images/{i}.png") for _ in range(10)])

        print(representations)

        torch.save(representations, f'data/representations.pt')

        print(representations.mean(0))
        print(representations.std(0))

        break

    break
