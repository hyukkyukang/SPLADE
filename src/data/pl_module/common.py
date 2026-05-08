import math
from typing import Any, Callable

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase

from src.data.lens_formatting import validate_lens_tokenizer
from src.utils.normalize import normalize_optional_str
from src.utils.transformers import build_tokenizer


class ContiguousDistributedSampler(Sampler[int]):
    """Distributed sampler that assigns contiguous index ranges per rank."""

    def __init__(self, dataset: Dataset[Any]) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            raise RuntimeError("torch.distributed must be initialized.")
        self.dataset: Dataset[Any] = dataset
        self.rank: int = int(torch.distributed.get_rank())
        self.num_replicas: int = int(torch.distributed.get_world_size())
        self.dataset_len: int = int(len(dataset))
        self.num_samples: int = int(math.ceil(self.dataset_len / self.num_replicas))
        start: int = (self.dataset_len * self.rank) // self.num_replicas
        end: int = (self.dataset_len * (self.rank + 1)) // self.num_replicas
        self.start_idx: int = start
        self.end_idx: int = end

    def __iter__(self):
        return iter(range(self.start_idx, self.end_idx))

    def __len__(self) -> int:
        return self.end_idx - self.start_idx


class StridedDistributedSampler(Sampler[int]):
    """Distributed sampler that interleaves individual rows across ranks."""

    def __init__(self, dataset: Dataset[Any]) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            raise RuntimeError("torch.distributed must be initialized.")
        self.dataset: Dataset[Any] = dataset
        self.rank: int = int(torch.distributed.get_rank())
        self.num_replicas: int = int(torch.distributed.get_world_size())
        self.dataset_len: int = int(len(dataset))

    def __iter__(self):
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= self.dataset_len:
            return 0
        return int(math.ceil((self.dataset_len - self.rank) / self.num_replicas))


class RowGroupInterleavedDistributedSampler(Sampler[int]):
    """Assign contiguous row groups round-robin across ranks."""

    def __init__(self, dataset: Dataset[Any], *, row_group_entries: list[Any]) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            raise RuntimeError("torch.distributed must be initialized.")
        self.dataset: Dataset[Any] = dataset
        self.rank: int = int(torch.distributed.get_rank())
        self.num_replicas: int = int(torch.distributed.get_world_size())
        self._ranges: list[tuple[int, int]] = []
        entry: Any
        for entry_idx, entry in enumerate(row_group_entries):
            if entry_idx % self.num_replicas != self.rank:
                continue
            start_idx: int = int(entry.start_idx)
            end_idx: int = start_idx + int(entry.num_rows)
            self._ranges.append((start_idx, end_idx))
        self.num_samples: int = sum(end - start for start, end in self._ranges)

    def __iter__(self):
        for start_idx, end_idx in self._ranges:
            yield from range(start_idx, end_idx)

    def __len__(self) -> int:
        return self.num_samples


def build_model_tokenizer(model_cfg: DictConfig) -> PreTrainedTokenizerBase:
    """Build a model tokenizer with shared fast-tokenizer validation.

    For ``strict_official_lens_tokenizer=True`` we delegate to
    :func:`src.utils.lens_official_loader.build_official_lens_tokenizer` so
    training and inference use the SAME tokenizer object; otherwise we fall
    back to the generic :func:`build_tokenizer` path.
    """
    local_files_only_value = model_cfg.get("local_files_only")
    local_files_only: bool | None = (
        None if local_files_only_value is None else bool(local_files_only_value)
    )
    strict_official_lens_tokenizer: bool = bool(
        model_cfg.get("strict_official_lens_tokenizer", False)
    )

    if strict_official_lens_tokenizer:
        from src.utils.lens_official_loader import build_official_lens_tokenizer
        tokenizer: PreTrainedTokenizerBase = build_official_lens_tokenizer(
            local_files_only=bool(
                local_files_only if local_files_only is not None else True
            ),
        )
    else:
        tokenizer_source: str = normalize_optional_str(
            model_cfg.get("tokenizer_name")
        ) or str(model_cfg.huggingface_name)
        tokenizer_revision: str | None = normalize_optional_str(
            model_cfg.get("tokenizer_revision")
        )
        tokenizer = build_tokenizer(
            tokenizer_source,
            use_fast_tokenizer=bool(model_cfg.use_fast_tokenizer),
            trust_remote_code=bool(model_cfg.trust_remote_code),
            require_fast_tokenizer=bool(model_cfg.require_fast_tokenizer),
            local_files_only=local_files_only,
            revision=tokenizer_revision,
            strict_official_lens_tokenizer=False,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    validate_lens_tokenizer(tokenizer, model_cfg)
    return tokenizer


def build_distributed_sampler(
    dataset: Dataset[Any],
    *,
    shuffle: bool,
    strategy: str = "contiguous",
    row_group_entries: list[Any] | None = None,
) -> Sampler[int] | None:
    """Build a distributed sampler when torch.distributed is initialized."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    if not shuffle:
        normalized_strategy: str = str(strategy).strip().lower()
        if normalized_strategy == "contiguous":
            return ContiguousDistributedSampler(dataset)
        if normalized_strategy == "strided":
            return StridedDistributedSampler(dataset)
        if normalized_strategy == "row_group_interleaved":
            if row_group_entries:
                return RowGroupInterleavedDistributedSampler(
                    dataset,
                    row_group_entries=row_group_entries,
                )
            return ContiguousDistributedSampler(dataset)
        raise ValueError(
            "Distributed sampler strategy must be one of: "
            "contiguous, strided, row_group_interleaved."
        )
    return DistributedSampler(dataset, shuffle=shuffle)


def build_inference_dataloader(
    *,
    dataset: Dataset[Any],
    batch_size: int,
    num_workers: int,
    collate_fn: Callable[..., Any] | None,
    use_cpu: bool,
    shuffle: bool,
    drop_last: bool = False,
    distributed_shuffle: bool | None = None,
    prefetch_factor: int | None = None,
    distributed_sampler_strategy: str = "contiguous",
    distributed_sampler_row_groups: list[Any] | None = None,
) -> DataLoader:
    """Build a non-training dataloader with optional distributed sampling."""
    sampler_shuffle: bool = (
        bool(distributed_shuffle) if distributed_shuffle is not None else bool(shuffle)
    )
    sampler: Sampler[int] | None = build_distributed_sampler(
        dataset,
        shuffle=sampler_shuffle,
        strategy=distributed_sampler_strategy,
        row_group_entries=distributed_sampler_row_groups,
    )
    dataloader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "collate_fn": collate_fn,
        "shuffle": bool(shuffle) if sampler is None else False,
        "drop_last": bool(drop_last),
        "pin_memory": not bool(use_cpu),
    }
    if sampler is not None:
        dataloader_kwargs["sampler"] = sampler
    if int(num_workers) > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**dataloader_kwargs)
