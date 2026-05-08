"""GPU sparse matmul scoring backend for inverted-index retrieval.

Formulates ranking as a single sparse×dense matmul + top-K on GPU:

    scores[doc, q] = sum_t D[doc, t] * Q[q, t]

where D is the inverted index expressed as a [n_docs, vocab] CSR tensor and
Q is a dense [batch, vocab] tensor built from the sparsified queries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from src.search.index import InvertedIndex


@dataclass
class GpuIndex:
    """Doc-indexed CSR sparse tensor living on the GPU."""

    d_csr: torch.Tensor  # torch.sparse_csr_tensor [n_docs, vocab_size]
    n_docs: int
    vocab_size: int
    values_dtype: torch.dtype
    device: torch.device


def build_gpu_index(
    index: InvertedIndex,
    *,
    values_dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cuda:0",
) -> GpuIndex:
    """Build a doc-indexed CSR sparse tensor on the GPU.

    The disk layout is term-indexed (term_ptr/post_doc_ids/post_weights describe
    a [vocab, n_docs] CSR). torch.sparse.mm(CSR, dense) is the fast path, so we
    transpose once here into a [n_docs, vocab] CSR via scipy and upload.
    """
    target_device: torch.device = torch.device(device)
    vocab_size: int = int(index.term_ptr.shape[0]) - 1
    n_docs: int = len(index.doc_ids)

    term_csr = sp.csr_matrix(
        (
            np.asarray(index.post_weights, dtype=np.float32),
            np.asarray(index.post_doc_ids, dtype=np.int64),
            np.asarray(index.term_ptr, dtype=np.int64),
        ),
        shape=(vocab_size, n_docs),
        copy=False,
    )
    doc_csr = term_csr.T.tocsr()
    del term_csr

    crow = torch.from_numpy(np.ascontiguousarray(doc_csr.indptr, dtype=np.int64)).to(
        target_device
    )
    col = torch.from_numpy(np.ascontiguousarray(doc_csr.indices, dtype=np.int64)).to(
        target_device
    )
    vals = torch.from_numpy(np.ascontiguousarray(doc_csr.data, dtype=np.float32)).to(
        target_device, dtype=values_dtype
    )
    del doc_csr

    d_csr: torch.Tensor = torch.sparse_csr_tensor(
        crow, col, vals, size=(n_docs, vocab_size), device=target_device
    )

    return GpuIndex(
        d_csr=d_csr,
        n_docs=n_docs,
        vocab_size=vocab_size,
        values_dtype=values_dtype,
        device=target_device,
    )


def _build_dense_query_batch(
    q_indices_list: list[np.ndarray],
    q_values_list: list[np.ndarray],
    vocab_size: int,
    values_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a dense [B, vocab] query tensor on the target device."""
    batch_size: int = len(q_indices_list)
    Q: torch.Tensor = torch.zeros(
        (batch_size, vocab_size), dtype=values_dtype, device=device
    )
    if batch_size == 0:
        return Q

    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []
    for i, (indices, values) in enumerate(zip(q_indices_list, q_values_list)):
        if indices.size == 0:
            continue
        row_chunks.append(np.full(indices.shape[0], i, dtype=np.int64))
        col_chunks.append(np.asarray(indices, dtype=np.int64))
        val_chunks.append(np.asarray(values, dtype=np.float32))
    if not row_chunks:
        return Q

    rows = torch.from_numpy(np.concatenate(row_chunks)).to(device)
    cols = torch.from_numpy(np.concatenate(col_chunks)).to(device)
    vals = torch.from_numpy(np.concatenate(val_chunks)).to(device=device, dtype=values_dtype)
    Q.index_put_((rows, cols), vals, accumulate=False)
    return Q


def score_batch_gpu(
    gpu_index: GpuIndex,
    q_indices_list: list[np.ndarray],
    q_values_list: list[np.ndarray],
    top_k: int,
    *,
    query_chunk: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Score a batch of queries via sparse matmul and per-query top-K.

    Returns per-query ``(doc_indices_int32, scores_float32)`` arrays, matching
    the CPU backend contract in ``retrieval.py``.
    """
    batch_size: int = len(q_indices_list)
    if batch_size == 0:
        return []
    effective_k: int = int(min(top_k, gpu_index.n_docs))
    if effective_k <= 0:
        return [
            (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
            for _ in range(batch_size)
        ]

    chunk_size: int = batch_size if query_chunk is None else max(1, int(query_chunk))

    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for start in range(0, batch_size, chunk_size):
        end: int = min(start + chunk_size, batch_size)
        Q = _build_dense_query_batch(
            q_indices_list[start:end],
            q_values_list[start:end],
            vocab_size=gpu_index.vocab_size,
            values_dtype=gpu_index.values_dtype,
            device=gpu_index.device,
        )
        # torch.sparse.mm(sparse_csr [n_docs, vocab], dense [vocab, b]) -> dense [n_docs, b]
        scores = torch.sparse.mm(gpu_index.d_csr, Q.t().contiguous())
        top_scores, top_indices = torch.topk(scores, k=effective_k, dim=0)
        # [k, b] -> [b, k]
        top_scores = top_scores.t().contiguous().to(dtype=torch.float32).cpu().numpy()
        top_indices = top_indices.t().contiguous().to(dtype=torch.int32).cpu().numpy()
        for row in range(end - start):
            all_indices.append(top_indices[row])
            all_scores.append(top_scores[row])
        del scores, Q, top_scores, top_indices

    return list(zip(all_indices, all_scores))


__all__ = ["GpuIndex", "build_gpu_index", "score_batch_gpu"]
