import logging
import os

import torch

from src.utils.logging import log_if_rank_zero as log_if_rank_zero_from_logging


def is_rank_zero() -> bool:
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def log_if_rank_zero(logger: logging.Logger, message: str, level: str = "info") -> None:
    """Proxy to src.utils.logging.log_if_rank_zero for backwards compatibility."""
    log_if_rank_zero_from_logging(logger=logger, message=message, level=level)


def get_world_size() -> int:
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def maybe_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def dist_gather_with_local_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather ``tensor`` across ranks, preserving the gradient on the local slot.

    ``dist.all_gather`` does not propagate gradients. We work around this by
    replacing the local rank's buffer with the original ``tensor`` (which still
    carries its autograd history) after the gather completes. Result: gradients
    flow through *our own* contribution to the concatenation, which is exactly
    what a contrastive loss needs.

    Mirrors ``Yibin-Lei/LENS/finetune/modeling.py::_dist_gather_tensor``.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return tensor
    world_size: int = int(torch.distributed.get_world_size())
    if world_size == 1:
        return tensor
    rank: int = int(torch.distributed.get_rank())
    contiguous: torch.Tensor = tensor.contiguous()
    buffers: list[torch.Tensor] = [
        torch.empty_like(contiguous) for _ in range(world_size)
    ]
    torch.distributed.all_gather(buffers, contiguous)
    buffers[rank] = tensor
    return torch.cat(buffers, dim=0)
