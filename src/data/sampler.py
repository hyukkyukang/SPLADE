import math
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class NonPaddingDistributedSampler(Sampler[int]):
    """Distributed sampler that never pads or duplicates indices."""

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = int(torch.distributed.get_world_size())
            else:
                num_replicas = 1
        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = int(torch.distributed.get_rank())
            else:
                rank = 0

        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be within [0, num_replicas).")

        self.dataset: Sized = dataset
        self.num_replicas: int = num_replicas
        self.rank: int = rank
        self.shuffle: bool = bool(shuffle)
        self.seed: int = int(seed)
        self.epoch: int = 0

    def __iter__(self) -> Iterator[int]:
        total_size: int = len(self.dataset)
        indices: list[int] = list(range(total_size))
        if self.shuffle and total_size > 0:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(total_size, generator=generator).tolist()
        return iter(indices[self.rank :: self.num_replicas])

    def __len__(self) -> int:
        total_size: int = len(self.dataset)
        if total_size <= self.rank:
            return 0
        return int(math.ceil((total_size - self.rank) / self.num_replicas))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
