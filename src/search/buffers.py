"""Buffer and config helpers for search."""

from typing import Any

import numpy as np
from omegaconf import DictConfig


def resolve_query_sparsify_config(
    cfg: DictConfig,
) -> tuple[list[int], float, int | None]:
    exclude_value: Any = cfg.testing.query_exclude_token_ids
    if exclude_value:
        exclude_ids = [int(token_id) for token_id in exclude_value]
    else:
        exclude_ids = []
    min_weight_value: float = float(cfg.testing.sparse_min_weight)
    top_k_value: int | None = cfg.testing.sparse_top_k
    if top_k_value is None:
        top_k = None
    else:
        top_k_int = int(top_k_value)
        top_k = None if top_k_int <= 0 else top_k_int
    return exclude_ids, min_weight_value, top_k


def prepare_score_buffers(doc_count: int) -> tuple[np.ndarray, np.ndarray]:
    score_buffer = np.zeros(int(doc_count), dtype=np.float32)
    seen_buffer = np.zeros(int(doc_count), dtype=np.uint8)
    return score_buffer, seen_buffer


__all__ = ["prepare_score_buffers", "resolve_query_sparsify_config"]
