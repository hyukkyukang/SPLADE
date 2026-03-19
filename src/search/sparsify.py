"""Query sparsification helpers for search."""

import numpy as np
import torch

from src.index.sparse import resolve_torch_dtype


def _sanitize_exclude_output_ids_tensor(
    exclude_output_ids: torch.Tensor | None,
    *,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if exclude_output_ids is None or int(exclude_output_ids.numel()) == 0:
        return None
    exclude_ids: torch.Tensor = exclude_output_ids.to(
        device=device, dtype=torch.long
    ).flatten()
    valid_ids: torch.Tensor = exclude_ids[
        (exclude_ids >= 0) & (exclude_ids < int(vocab_size))
    ]
    if int(valid_ids.numel()) == 0:
        return None
    return torch.unique(valid_ids, sorted=True)


def _sanitize_exclude_output_ids_array(
    exclude_output_ids: list[int],
    *,
    vocab_size: int,
) -> np.ndarray:
    if not exclude_output_ids:
        return np.zeros((0,), dtype=np.int64)
    exclude_array: np.ndarray = np.asarray(exclude_output_ids, dtype=np.int64)
    valid_mask: np.ndarray = (exclude_array >= 0) & (exclude_array < int(vocab_size))
    if not bool(valid_mask.any()):
        return np.zeros((0,), dtype=np.int64)
    return np.unique(exclude_array[valid_mask])


def sparsify_vector_gpu(
    vector: torch.Tensor,
    *,
    exclude_output_ids: torch.Tensor | None,
    min_weight: float,
    top_k: int | None,
    value_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparsify a single dense vector on the current device."""
    local_vec: torch.Tensor = vector
    sanitized_exclude_ids: torch.Tensor | None = _sanitize_exclude_output_ids_tensor(
        exclude_output_ids,
        vocab_size=int(vector.shape[0]),
        device=vector.device,
    )
    if sanitized_exclude_ids is not None:
        local_vec = local_vec.clone()
        local_vec[sanitized_exclude_ids] = 0.0

    if min_weight > 0.0:
        mask: torch.Tensor = local_vec > min_weight
    else:
        mask = local_vec > 0.0

    if not bool(mask.any()):
        empty_indices: np.ndarray = np.zeros((0,), dtype=np.int32)
        empty_values: np.ndarray = np.zeros((0,), dtype=value_dtype)
        return empty_indices, empty_values

    indices: torch.Tensor = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if top_k is not None and int(indices.numel()) > int(top_k):
        values_for_topk: torch.Tensor = local_vec[indices]
        topk_values: torch.Tensor
        topk_positions: torch.Tensor
        topk_values, topk_positions = torch.topk(
            values_for_topk, k=int(top_k), largest=True, sorted=False
        )
        indices = indices[topk_positions]
        values: torch.Tensor = topk_values
    else:
        values = local_vec[indices]

    if int(indices.numel()) > 1:
        order: torch.Tensor = torch.argsort(indices)
        indices = indices[order]
        values = values[order]

    indices_np: np.ndarray = indices.detach().cpu().numpy().astype(np.int32, copy=False)
    values_np: np.ndarray = (
        values.detach().cpu().numpy().astype(value_dtype, copy=False)
    )
    return indices_np, values_np


def _sparsify_batch_gpu_csr_core_topk(
    vectors: torch.Tensor,
    *,
    exclude_output_ids: torch.Tensor | None,
    threshold: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparsify a batch of dense vectors into CSR tensors on GPU (top-k path)."""
    batch_size: int = int(vectors.shape[0])
    masked: torch.Tensor = vectors
    sanitized_exclude_ids: torch.Tensor | None = _sanitize_exclude_output_ids_tensor(
        exclude_output_ids,
        vocab_size=int(vectors.shape[1]),
        device=vectors.device,
    )
    if sanitized_exclude_ids is not None:
        masked = masked.clone()
        masked.index_fill_(1, sanitized_exclude_ids, float("-inf"))
    topk_values: torch.Tensor
    topk_indices: torch.Tensor
    topk_values, topk_indices = torch.topk(
        masked, k=int(top_k), dim=1, largest=True, sorted=False
    )
    valid_mask: torch.Tensor = topk_values > threshold
    order: torch.Tensor = torch.argsort(topk_indices, dim=1)
    sorted_indices: torch.Tensor = torch.gather(topk_indices, 1, order)
    sorted_values: torch.Tensor = torch.gather(topk_values, 1, order)
    sorted_valid: torch.Tensor = torch.gather(valid_mask, 1, order)
    row_counts: torch.Tensor = sorted_valid.sum(dim=1, dtype=torch.int64)
    indptr_gpu: torch.Tensor = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=vectors.device
    )
    indptr_gpu[1:] = torch.cumsum(row_counts, dim=0)
    flat_indices: torch.Tensor = sorted_indices[sorted_valid]
    flat_values: torch.Tensor = sorted_values[sorted_valid]
    return indptr_gpu, flat_indices, flat_values


def _sparsify_batch_gpu_csr_core_threshold(
    vectors: torch.Tensor,
    *,
    exclude_output_ids: torch.Tensor | None,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparsify a batch of dense vectors into CSR tensors on GPU (threshold path)."""
    batch_size: int = int(vectors.shape[0])
    vocab_size: int = int(vectors.shape[1])
    mask: torch.Tensor = vectors > threshold
    sanitized_exclude_ids: torch.Tensor | None = _sanitize_exclude_output_ids_tensor(
        exclude_output_ids,
        vocab_size=vocab_size,
        device=vectors.device,
    )
    if sanitized_exclude_ids is not None:
        mask.index_fill_(1, sanitized_exclude_ids, False)
    row_idx: torch.Tensor
    col_idx: torch.Tensor
    row_idx, col_idx = torch.nonzero(mask, as_tuple=True)
    linear_idx: torch.Tensor = row_idx.to(torch.int64) * int(vocab_size) + col_idx.to(
        torch.int64
    )
    order: torch.Tensor = torch.argsort(linear_idx)
    row_idx = row_idx[order]
    col_idx = col_idx[order]
    flat_values: torch.Tensor = vectors[row_idx, col_idx]
    row_counts: torch.Tensor = torch.bincount(row_idx, minlength=batch_size)
    flat_indices: torch.Tensor = col_idx
    indptr_gpu: torch.Tensor = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=vectors.device
    )
    indptr_gpu[1:] = torch.cumsum(row_counts, dim=0)
    return indptr_gpu, flat_indices, flat_values


def sparsify_batch_gpu_csr(
    vectors: torch.Tensor,
    *,
    exclude_output_ids: torch.Tensor | None,
    min_weight: float,
    top_k: int | None,
    value_dtype: np.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparsify a batch of dense vectors into CSR tensors on GPU."""
    if vectors.ndim != 2:
        raise ValueError("sparsify_batch_gpu_csr expects a 2D tensor.")

    batch_size: int = int(vectors.shape[0])
    vocab_size: int = int(vectors.shape[1])
    if batch_size == 0:
        indptr = torch.zeros((1,), dtype=torch.int64, device="cpu")
        indices = torch.empty((0,), dtype=torch.int32, device="cpu")
        values = torch.empty(
            (0,), dtype=resolve_torch_dtype(value_dtype), device="cpu"
        )
        return indptr, indices, values

    device: torch.device = vectors.device
    threshold: float = float(min_weight) if min_weight > 0.0 else 0.0
    exclude_ids: torch.Tensor | None = _sanitize_exclude_output_ids_tensor(
        exclude_output_ids,
        vocab_size=vocab_size,
        device=device,
    )

    if top_k is not None:
        top_k_int: int = min(int(top_k), vocab_size)
        if top_k_int <= 0:
            indptr = torch.zeros((batch_size + 1,), dtype=torch.int64, device="cpu")
            indices = torch.empty((0,), dtype=torch.int32, device="cpu")
            values = torch.empty(
                (0,), dtype=resolve_torch_dtype(value_dtype), device="cpu"
            )
            return indptr, indices, values

        indptr_gpu, flat_indices, flat_values = _sparsify_batch_gpu_csr_core_topk(
            vectors,
            exclude_output_ids=exclude_ids,
            threshold=threshold,
            top_k=top_k_int,
        )
    else:
        indptr_gpu, flat_indices, flat_values = (
            _sparsify_batch_gpu_csr_core_threshold(
                vectors,
                exclude_output_ids=exclude_ids,
                threshold=threshold,
            )
        )

    torch_value_dtype: torch.dtype = resolve_torch_dtype(value_dtype)
    indptr = indptr_gpu.to(device="cpu")
    indices = flat_indices.to(dtype=torch.int32, device="cpu")
    values = flat_values.to(dtype=torch_value_dtype, device="cpu")
    return indptr, indices, values


def sparsify_query_vector(
    vector: np.ndarray,
    *,
    exclude_output_ids: list[int],
    min_weight: float,
    top_k: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a dense query vector into sparse indices and values."""
    if vector.ndim != 1:
        raise ValueError("Query vector must be 1D.")

    local_vec: np.ndarray = vector
    # Apply a positive-weight mask and optional threshold.
    if min_weight > 0.0:
        mask: np.ndarray = local_vec > float(min_weight)
    else:
        mask = local_vec > 0.0
    exclude_array: np.ndarray = _sanitize_exclude_output_ids_array(
        exclude_output_ids,
        vocab_size=int(vector.shape[0]),
    )
    if int(exclude_array.size) > 0:
        mask[exclude_array] = False

    indices: np.ndarray = np.nonzero(mask)[0].astype(np.int32, copy=False)
    if indices.size == 0:
        empty_indices: np.ndarray = np.zeros((0,), dtype=np.int32)
        empty_values: np.ndarray = np.zeros((0,), dtype=np.float32)
        return empty_indices, empty_values

    values: np.ndarray = local_vec[indices].astype(np.float32, copy=False)
    if top_k is not None and int(indices.size) > int(top_k):
        top_k_int: int = int(top_k)
        # Keep the highest-weight terms only.
        top_positions: np.ndarray = np.argpartition(values, -top_k_int)[-top_k_int:]
        indices = indices[top_positions]
        values = values[top_positions]

    if int(indices.size) > 1:
        order: np.ndarray = np.argsort(indices)
        indices = indices[order]
        values = values[order]
    return indices, values


__all__ = [
    "_sparsify_batch_gpu_csr_core_topk",
    "_sparsify_batch_gpu_csr_core_threshold",
    "sparsify_batch_gpu_csr",
    "sparsify_query_vector",
    "sparsify_vector_gpu",
]
