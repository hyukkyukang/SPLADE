"""Buffer and config helpers for search."""

from typing import Any

import numpy as np


def resolve_query_sparsify_config(
    metadata: dict[str, Any],
) -> tuple[list[int], float, int | None]:
    exclude_ids: list[int] = [
        int(token_id) for token_id in metadata.get("exclude_token_ids") or []
    ]
    min_weight_value: float = float(metadata.get("min_weight") or 0.0)
    top_k_value: int | None = (
        None if metadata.get("top_k") is None else int(metadata["top_k"])
    )
    return exclude_ids, min_weight_value, top_k_value


def prepare_score_buffers(doc_count: int) -> tuple[np.ndarray, np.ndarray]:
    score_buffer = np.zeros(int(doc_count), dtype=np.float32)
    seen_buffer = np.zeros(int(doc_count), dtype=np.uint8)
    return score_buffer, seen_buffer


__all__ = ["prepare_score_buffers", "resolve_query_sparsify_config"]
