from collections import defaultdict

import torch
from torch import nn


def get_buckets(data: torch.Tensor, proj: torch.Tensor) -> set:
    """This function maps batch of encoder output features to a set of unique buckets via LSH"""
    result = data @ proj

    hashed = list(map(tuple, (result > 0).int().tolist()))

    print(hashed)

    buckets = defaultdict(list)

    for i, row in enumerate(hashed):
        buckets[row].append(i)

    return set([str(k) for k in dict(buckets).keys()])


def get_std(buckets_covered, lam=0.000001, alpha=0.8, beta=80):
    std = lam * (torch.exp(torch.log(torch.tensor(alpha / lam)) * buckets_covered / beta) - 1)

    return max(std.item(), 0)


class ModelDefence(nn.Module):
    def __init__(self, model: nn.Module, embedding_dim: int, proj_count: int, bucket_dim: int):
        super(ModelDefence, self).__init__()

        self.model = model
        self.proj_count = proj_count
        self.bucket_dim = bucket_dim
        self.embedding_dim = embedding_dim
        self.reset()

    def reset(self) -> None:
        self.proj_list = [torch.randn((self.embedding_dim, self.bucket_dim)) for _ in range(self.proj_count)]
        self.covered_buckets = [set() for _ in range(self.proj_count)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        prediction = self.model(x)

        set_of_new_buckets_list = [
            get_buckets(prediction, proj) for proj in self.proj_list
        ]

        for set_of_buckets, set_of_new_buckets in zip(
                self.covered_buckets, set_of_new_buckets_list
        ):
            set_of_buckets.update(set_of_new_buckets)

        buckets_list = torch.tensor([
            100 * len(set_of_buckets) / (2 ** self.proj_count)
            for set_of_buckets in self.covered_buckets
        ])
        std = get_std(buckets_list.mean())

        return torch.squeeze(
            prediction + torch.normal(
                0, std, size=prediction.shape
            )
        )
