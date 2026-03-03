from typing import Any, Callable

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase

from src.utils.transformers import build_tokenizer


def build_model_tokenizer(model_cfg: DictConfig) -> PreTrainedTokenizerBase:
    """Build a model tokenizer with shared fast-tokenizer validation."""
    tokenizer: PreTrainedTokenizerBase = build_tokenizer(
        str(model_cfg.huggingface_name),
        use_fast_tokenizer=bool(model_cfg.use_fast_tokenizer),
        trust_remote_code=bool(model_cfg.trust_remote_code),
        require_fast_tokenizer=bool(model_cfg.require_fast_tokenizer),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    return tokenizer


def build_distributed_sampler(
    dataset: Dataset[Any], *, shuffle: bool
) -> DistributedSampler | None:
    """Build a distributed sampler when torch.distributed is initialized."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
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
) -> DataLoader:
    """Build a non-training dataloader with optional distributed sampling."""
    sampler_shuffle: bool = (
        bool(distributed_shuffle) if distributed_shuffle is not None else bool(shuffle)
    )
    sampler: DistributedSampler | None = build_distributed_sampler(
        dataset, shuffle=sampler_shuffle
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
