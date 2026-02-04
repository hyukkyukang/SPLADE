import logging
from typing import Any

import pyarrow as pa

from src.utils.logging import log_if_rank_zero

logger: logging.Logger = logging.getLogger("src.data.utils")


def resolve_dataset_column(dataset: Any, column_name: str) -> pa.Array:
    """
    Resolve a column from a HuggingFace Dataset as a PyArrow array.

    Direct access via dataset.data.column() returns the underlying PyArrow table's
    column, which ignores any _indices from filtering. This function resolves the
    correct column values by applying _indices when present.
    """
    column: pa.Array | pa.ChunkedArray = dataset.data.column(column_name)

    # Apply _indices if the dataset was filtered to preserve row alignment.
    if dataset._indices is not None:
        indices: pa.ChunkedArray = dataset._indices.column(0)
        column = column.take(indices)

    return column


def id_to_idx(
    ids: pa.Array | pa.ChunkedArray | list[Any],
    desc: str,
    enable_tqdm: bool,
) -> dict[str, int]:
    """
    Create a mapping from IDs to their indices.

    Uses PyArrow's native to_pylist() for fast batch conversion when available,
    avoiding slow element-by-element iteration that occurs with list().
    """
    # Parameters are kept for API compatibility with older call sites.
    _ = (desc, enable_tqdm)

    ids_list: list[Any]
    if isinstance(ids, (pa.Array, pa.ChunkedArray)):
        ids_list = ids.to_pylist()
    else:
        log_if_rank_zero(
            logger,
            "Using list() for slow element-by-element iteration... "
            "(Consider using PyArrow arrays for faster conversion).",
            level="warning",
        )
        ids_list = list(ids)

    # Normalize IDs to strings for consistent lookup keys.
    normalized_ids: list[str] = [str(value) for value in ids_list]
    return dict(zip(normalized_ids, range(len(normalized_ids))))
