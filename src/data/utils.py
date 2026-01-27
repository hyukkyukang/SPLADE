import hashlib
import logging
import os
from typing import Any, Dict, List, Union

import pyarrow as pa
from datasets import config as datasets_config
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


def build_integer_id_cache_key(
    hf_name: str,
    hf_subset: str,
    hf_split: str,
    query_id_column: str,
    corpus_id_column: str,
    hf_max_samples: int | None,
) -> str:
    """
    Build a deterministic cache key for integer-id preprocessing artifacts.
    """
    max_samples_value: str = "none" if hf_max_samples is None else str(hf_max_samples)
    raw_key: str = "|".join(
        [
            hf_name,
            hf_subset,
            hf_split,
            query_id_column,
            corpus_id_column,
            max_samples_value,
        ]
    )
    # Hash to keep the on-disk path short and stable.
    digest: str = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"integer_ids_{digest}"


def resolve_integer_id_cache_dir(
    hf_cache_dir: str | None, integer_id_cache_dir: str | None
) -> str:
    """
    Resolve the base directory for integer-id preprocessing artifacts.
    """
    # Prefer the explicit override, then the dataset cache, then HF defaults.
    base_cache_dir_value: str | os.PathLike[str] = (
        integer_id_cache_dir
        if integer_id_cache_dir is not None
        else (
            hf_cache_dir
            if hf_cache_dir is not None
            else datasets_config.HF_DATASETS_CACHE
        )
    )
    base_cache_dir: str = os.fspath(base_cache_dir_value)
    return base_cache_dir
