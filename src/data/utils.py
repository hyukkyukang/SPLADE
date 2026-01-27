import logging
from typing import Any, Dict, List, Union

import pyarrow as pa
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger("DatasetUtils")


def resolve_dataset_column(dataset: Any, column_name: str) -> pa.Array:
    """
    Resolve a column from a HuggingFace Dataset as a PyArrow array.

    Direct access via dataset.data.column() returns the underlying PyArrow table's
    column, which ignores any _indices from filtering. This function resolves the
    correct column values by applying _indices when present.

    Args:
        dataset: A HuggingFace Dataset
        column_name: Name of the column to extract

    Returns:
        PyArrow Array with correct values matching the dataset's visible rows
    """
    column: pa.ChunkedArray = dataset.data.column(column_name)

    # Apply _indices if dataset was filtered
    if dataset._indices is not None:
        indices: pa.ChunkedArray = dataset._indices.column(0)
        column = column.take(indices)

    return column


def id_to_idx(
    ids: Union[pa.Array, pa.ChunkedArray, List[str]], desc: str, enable_tqdm: bool
) -> Dict[str, int]:
    """
    Create a mapping from IDs to their indices.

    Uses PyArrow's native to_pylist() for fast batch conversion when available,
    avoiding slow element-by-element iteration that occurs with list().

    Args:
        ids: IDs as PyArrow Array/ChunkedArray (from HuggingFace datasets) or Python list
        desc: Description for logging (kept for API compatibility, not used)
        enable_tqdm: Whether to show progress (kept for API compatibility, not used)

    Returns:
        Dictionary mapping each ID to its index position.
    """
    # Check if ids is a PyArrow array type for fast batch conversion
    if isinstance(ids, (pa.Array, pa.ChunkedArray)):
        ids_list = ids.to_pylist()
    else:
        logger.warning(
            "Using list() for slow element-by-element iteration... (Consider using PyArrow arrays for faster conversion)."
        )
        ids_list = list(ids)
    return dict(zip(ids_list, range(len(ids_list))))
