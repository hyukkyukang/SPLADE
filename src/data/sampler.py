import math
from typing import Iterator, List, Sequence, Sized

import numpy as np
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


class SameDatasetBatchSampler(Sampler[List[int]]):
    """Yield batches whose rows all come from the same task (dataset split).

    Mirrors ``Yibin-Lei/LENS/finetune/data.py::SameDatasetTrainDataset``:

    1. At every epoch, shuffle the per-task index arrays and the dataset order.
    2. For each task, slice its shuffled indices into chunks of size
       ``per_task_batch_size[task] * num_replicas`` (drop the tail).
    3. Shuffle the resulting list of batches globally.
    4. Each rank takes ``batch[rank * local_size : (rank+1) * local_size]``.

    Consumers should use this as ``batch_sampler=...`` on DataLoader
    (batch_size=None). Each yielded element is a *list of global indices* equal
    in length to the local per-rank batch size for the task.

    Parameters
    ----------
    row_ranges : Sequence[np.ndarray]
        One int64 array per task — global row indices belonging to that task.
    per_task_batch_size : Sequence[int]
        Local (per-rank) batch size for each task, aligned with ``row_ranges``.
    num_replicas : int | None
        DDP world size. If None, autodetected (1 if not distributed).
    rank : int | None
        DDP rank. Autodetected analogously.
    seed : int
        Base seed for deterministic shuffling.
    drop_last : bool
        Drop the tail batch when it doesn't divide evenly (LENS default: True).
    """

    def __init__(
        self,
        row_ranges: Sequence[np.ndarray],
        per_task_batch_size: Sequence[int],
        *,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if len(row_ranges) == 0:
            raise ValueError("row_ranges must be non-empty.")
        if len(row_ranges) != len(per_task_batch_size):
            raise ValueError(
                "row_ranges and per_task_batch_size must align (got "
                f"{len(row_ranges)} vs {len(per_task_batch_size)})."
            )
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

        self._row_ranges: List[np.ndarray] = [
            np.asarray(r, dtype=np.int64).copy() for r in row_ranges
        ]
        self._per_task_batch_size: List[int] = [int(v) for v in per_task_batch_size]
        self.num_replicas: int = int(num_replicas)
        self.rank: int = int(rank)
        self.seed: int = int(seed)
        self.drop_last: bool = bool(drop_last)
        self.epoch: int = 0
        self._cached_batches: List[np.ndarray] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_batches = None

    @property
    def batch_size(self) -> int:
        """Representative per-batch size, exposed for Lightning's
        ``DeepSpeedStrategy._auto_select_batch_size`` which expects a flat
        ``batch_size`` attribute on the sampler. We have per-task sizes (the
        symmetric_batch_size differs from train_group dispatch), so this
        returns the maximum so DeepSpeed's reported ``train_batch_size``
        can't underestimate. Training behavior is unaffected — DeepSpeed
        uses this number only for logging metadata, not for control flow.
        """
        return max(self._per_task_batch_size) if self._per_task_batch_size else 1

    def _build_batches(self) -> List[np.ndarray]:
        rng: np.random.Generator = np.random.default_rng(self.seed + self.epoch)
        task_order: np.ndarray = np.arange(len(self._row_ranges))
        rng.shuffle(task_order)
        batches: List[np.ndarray] = []
        for task_idx in task_order:
            rows: np.ndarray = self._row_ranges[task_idx].copy()
            rng.shuffle(rows)
            global_batch_size: int = (
                self._per_task_batch_size[task_idx] * self.num_replicas
            )
            if global_batch_size <= 0:
                continue
            n_full: int = len(rows) // global_batch_size
            end: int = n_full * global_batch_size
            if end == 0:
                # Not even one complete global batch — skip if drop_last.
                if not self.drop_last and len(rows) >= self.num_replicas:
                    batches.append(rows[: self.num_replicas])
                continue
            for start in range(0, end, global_batch_size):
                batches.append(rows[start : start + global_batch_size])
            if not self.drop_last and end < len(rows):
                tail: np.ndarray = rows[end:]
                if len(tail) >= self.num_replicas:
                    batches.append(tail)
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        global_batch: np.ndarray
        for global_batch in self._cached_batches:
            local_size: int = len(global_batch) // self.num_replicas
            start: int = local_size * self.rank
            end: int = start + local_size
            yield [int(i) for i in global_batch[start:end]]

    def __len__(self) -> int:
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()
        return len(self._cached_batches)
