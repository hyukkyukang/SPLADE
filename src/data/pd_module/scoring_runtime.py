from typing import Callable

import torch


def run_prepare_data_on_primary(
    *,
    local_files_only_for_rank: Callable[[bool], bool],
    prepare_fn: Callable[[bool], None],
) -> None:
    """
    Execute prepare_data once on global rank 0 when distributed is initialized.

    `prepare_fn` receives the resolved `local_files_only` flag.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if int(torch.distributed.get_rank()) != 0:
            return
    prepare_fn(local_files_only_for_rank(True))


def run_setup_with_barrier(
    *,
    local_files_only_for_rank: Callable[[bool], bool],
    load_all_fn: Callable[[bool], None],
) -> None:
    """
    Load datasets in distributed mode with a rank-0 priming phase and barrier.

    `load_all_fn` receives the resolved `local_files_only` flag.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_primary: bool = int(torch.distributed.get_rank()) == 0
        if is_primary:
            load_all_fn(local_files_only_for_rank(True))
        torch.distributed.barrier()
        if not is_primary:
            load_all_fn(local_files_only_for_rank(False))
        return
    load_all_fn(local_files_only_for_rank(True))
