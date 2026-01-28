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
