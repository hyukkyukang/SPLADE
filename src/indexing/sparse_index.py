import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numba
import numpy as np
import torch

logger: logging.Logger = logging.getLogger("src.indexing.sparse_index")


def resolve_numpy_dtype(dtype_name: str) -> np.dtype:
    """Resolve a numpy dtype from a string label."""
    normalized: str = str(dtype_name).lower()
    if normalized in {"float16", "fp16"}:
        return np.float16
    if normalized in {"float32", "fp32"}:
        return np.float32
    if normalized in {"float64", "fp64"}:
        return np.float64
    raise ValueError(f"Unsupported numpy dtype: {dtype_name}")


@dataclass(frozen=True)
class ShardInfo:
    """File locations for a single sparse shard."""

    rank: int
    shard_id: int
    doc_count: int
    nnz: int
    indptr_path: Path
    indices_path: Path
    values_path: Path
    doc_ids_path: Path


@dataclass(frozen=True)
class InvertedIndex:
    """Inverted index data loaded from disk."""

    term_ptr: np.ndarray
    post_doc_ids: np.ndarray
    post_weights: np.ndarray
    doc_ids: list[str]
    metadata: dict[str, Any]


class SparseShardWriter:
    """Write sparse doc vectors into CSR shards on disk."""

    # --- Special methods ---
    def __init__(
        self,
        output_dir: Path,
        vocab_size: int,
        rank: int,
        *,
        top_k: int | None,
        min_weight: float,
        exclude_token_ids: Sequence[int],
        shard_max_docs: int,
        value_dtype: str,
    ) -> None:
        self.output_dir: Path = output_dir
        self.vocab_size: int = int(vocab_size)
        self.rank: int = int(rank)
        self.top_k: int | None = None if top_k is None else int(top_k)
        self.min_weight: float = float(min_weight)
        self.exclude_token_ids: list[int] = [
            int(token_id) for token_id in exclude_token_ids
        ]
        self.shard_max_docs: int = max(1, int(shard_max_docs))
        self.value_dtype: np.dtype = resolve_numpy_dtype(value_dtype)

        self._rank_dir: Path = self.output_dir / "shards" / f"rank_{self.rank}"
        self._rank_dir.mkdir(parents=True, exist_ok=True)

        self._exclude_tensor: torch.Tensor | None = None
        if self.exclude_token_ids:
            self._exclude_tensor = torch.tensor(
                self.exclude_token_ids, dtype=torch.long, device="cpu"
            )

        self._manifest: list[dict[str, Any]] = []
        self._shard_idx: int = 0
        self._total_docs: int = 0
        self._total_nnz: int = 0
        self._reset_buffer()

    # --- Protected methods ---
    def _reset_buffer(self) -> None:
        self._buffer_doc_ids: list[str] = []
        self._buffer_indices: list[np.ndarray] = []
        self._buffer_values: list[np.ndarray] = []
        self._buffer_indptr: list[int] = [0]
        self._buffer_nnz: int = 0

    def _sparsify_vector(self, vector: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        local_vec: torch.Tensor = vector
        if self._exclude_tensor is not None:
            local_vec = local_vec.clone()
            local_vec[self._exclude_tensor] = 0.0

        if self.min_weight > 0.0:
            mask: torch.Tensor = local_vec > self.min_weight
        else:
            mask = local_vec > 0.0

        if not bool(mask.any()):
            empty_indices: np.ndarray = np.zeros((0,), dtype=np.int32)
            empty_values: np.ndarray = np.zeros((0,), dtype=self.value_dtype)
            return empty_indices, empty_values

        indices: torch.Tensor = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if self.top_k is not None and int(indices.numel()) > self.top_k:
            values_for_topk: torch.Tensor = local_vec[indices]
            topk_values: torch.Tensor
            topk_positions: torch.Tensor
            topk_values, topk_positions = torch.topk(
                values_for_topk, k=self.top_k, largest=True, sorted=False
            )
            indices = indices[topk_positions]
            values: torch.Tensor = topk_values
        else:
            values: torch.Tensor = local_vec[indices]

        if int(indices.numel()) > 1:
            order: torch.Tensor = torch.argsort(indices)
            indices = indices[order]
            values = values[order]

        indices_np: np.ndarray = indices.cpu().numpy().astype(np.int32, copy=False)
        values_np: np.ndarray = (
            values.cpu().numpy().astype(self.value_dtype, copy=False)
        )
        return indices_np, values_np

    def _append_doc(self, doc_id: str, indices: np.ndarray, values: np.ndarray) -> None:
        self._buffer_doc_ids.append(doc_id)
        self._buffer_indices.append(indices)
        self._buffer_values.append(values)
        next_ptr: int = int(self._buffer_indptr[-1]) + int(indices.size)
        self._buffer_indptr.append(next_ptr)
        self._buffer_nnz += int(indices.size)

    def _flush(self) -> None:
        if not self._buffer_doc_ids:
            return

        shard_prefix: Path = self._rank_dir / f"shard_{self._shard_idx:06d}"
        indptr: np.ndarray = np.array(self._buffer_indptr, dtype=np.int64)
        if self._buffer_nnz > 0:
            indices: np.ndarray = np.concatenate(self._buffer_indices).astype(
                np.int32, copy=False
            )
            values: np.ndarray = np.concatenate(self._buffer_values).astype(
                self.value_dtype, copy=False
            )
        else:
            indices = np.zeros((0,), dtype=np.int32)
            values = np.zeros((0,), dtype=self.value_dtype)

        indptr_path: Path = Path(f"{shard_prefix}_indptr.npy")
        indices_path: Path = Path(f"{shard_prefix}_indices.npy")
        values_path: Path = Path(f"{shard_prefix}_values.npy")
        doc_ids_path: Path = Path(f"{shard_prefix}_doc_ids.json")

        np.save(indptr_path, indptr)
        np.save(indices_path, indices)
        np.save(values_path, values)
        with doc_ids_path.open("w", encoding="utf-8") as doc_file:
            json.dump(self._buffer_doc_ids, doc_file)

        shard_record: dict[str, Any] = {
            "shard_id": self._shard_idx,
            "doc_count": len(self._buffer_doc_ids),
            "nnz": int(indices.size),
            "indptr": indptr_path.name,
            "indices": indices_path.name,
            "values": values_path.name,
            "doc_ids": doc_ids_path.name,
        }
        self._manifest.append(shard_record)
        self._shard_idx += 1
        self._total_docs += len(self._buffer_doc_ids)
        self._total_nnz += int(indices.size)
        self._reset_buffer()

    # --- Public methods ---
    def write_batch(self, doc_ids: Sequence[str], doc_reps: torch.Tensor) -> None:
        if len(doc_ids) == 0:
            return
        if int(doc_reps.shape[0]) != len(doc_ids):
            raise ValueError("doc_ids length does not match doc_reps batch size.")

        doc_reps_cpu: torch.Tensor = doc_reps.detach()
        if doc_reps_cpu.is_cuda:
            doc_reps_cpu = doc_reps_cpu.cpu()
        doc_reps_cpu = doc_reps_cpu.float()

        batch_size: int = int(doc_reps_cpu.shape[0])
        for idx in range(batch_size):
            doc_id: str = str(doc_ids[idx])
            vector: torch.Tensor = doc_reps_cpu[idx]
            indices: np.ndarray
            values: np.ndarray
            indices, values = self._sparsify_vector(vector)
            self._append_doc(doc_id, indices, values)
            if len(self._buffer_doc_ids) >= self.shard_max_docs:
                self._flush()

    def finalize(self) -> None:
        self._flush()
        manifest_path: Path = self._rank_dir / "manifest.json"
        manifest_payload: dict[str, Any] = {
            "rank": self.rank,
            "vocab_size": self.vocab_size,
            "top_k": self.top_k,
            "min_weight": self.min_weight,
            "exclude_token_ids": self.exclude_token_ids,
            "value_dtype": str(self.value_dtype),
            "doc_count": self._total_docs,
            "nnz": self._total_nnz,
            "shards": self._manifest,
        }
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest_payload, manifest_file, indent=2)


def load_shard_manifest(encode_path: Path) -> tuple[list[ShardInfo], dict[str, Any]]:
    """Load shard metadata from an encode output directory."""
    shards_root: Path = encode_path / "shards"
    if not shards_root.exists():
        raise FileNotFoundError(f"Missing shards directory at {shards_root}.")

    shard_infos: list[ShardInfo] = []
    metadata: dict[str, Any] = {}
    rank_dirs: list[Path] = sorted(shards_root.glob("rank_*"))
    if not rank_dirs:
        raise FileNotFoundError("No rank directories found under encode shards.")

    for rank_dir in rank_dirs:
        manifest_path: Path = rank_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest: dict[str, Any] = json.load(manifest_file)

        if not metadata:
            metadata = {
                "vocab_size": manifest.get("vocab_size"),
                "top_k": manifest.get("top_k"),
                "min_weight": manifest.get("min_weight"),
                "exclude_token_ids": manifest.get("exclude_token_ids"),
                "value_dtype": manifest.get("value_dtype"),
            }

        shards: Iterable[dict[str, Any]] = manifest.get("shards", [])
        for shard in shards:
            shard_infos.append(
                ShardInfo(
                    rank=int(manifest.get("rank", 0)),
                    shard_id=int(shard["shard_id"]),
                    doc_count=int(shard["doc_count"]),
                    nnz=int(shard["nnz"]),
                    indptr_path=rank_dir / str(shard["indptr"]),
                    indices_path=rank_dir / str(shard["indices"]),
                    values_path=rank_dir / str(shard["values"]),
                    doc_ids_path=rank_dir / str(shard["doc_ids"]),
                )
            )

    shard_infos.sort(key=lambda info: (info.rank, info.shard_id))
    return shard_infos, metadata


def load_inverted_index(index_path: Path) -> InvertedIndex:
    """Load an inverted index from disk with memory-mapped arrays."""
    term_ptr_path: Path = index_path / "term_ptr.npy"
    post_doc_ids_path: Path = index_path / "post_doc_ids.npy"
    post_weights_path: Path = index_path / "post_weights.npy"
    doc_ids_path: Path = index_path / "doc_ids.json"
    metadata_path: Path = index_path / "metadata.json"

    if not term_ptr_path.exists():
        raise FileNotFoundError(f"Missing term_ptr.npy at {term_ptr_path}")
    if not post_doc_ids_path.exists():
        raise FileNotFoundError(f"Missing post_doc_ids.npy at {post_doc_ids_path}")
    if not post_weights_path.exists():
        raise FileNotFoundError(f"Missing post_weights.npy at {post_weights_path}")
    if not doc_ids_path.exists():
        raise FileNotFoundError(f"Missing doc_ids.json at {doc_ids_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json at {metadata_path}")

    # Memory-map arrays to avoid large RAM spikes for big corpora.
    term_ptr: np.ndarray = np.load(term_ptr_path, mmap_mode="r")
    post_doc_ids: np.ndarray = np.load(post_doc_ids_path, mmap_mode="r")
    post_weights: np.ndarray = np.load(post_weights_path, mmap_mode="r")
    with doc_ids_path.open("r", encoding="utf-8") as doc_file:
        doc_ids: list[str] = json.load(doc_file)
    with metadata_path.open("r", encoding="utf-8") as meta_file:
        metadata: dict[str, Any] = json.load(meta_file)

    return InvertedIndex(
        term_ptr=term_ptr,
        post_doc_ids=post_doc_ids,
        post_weights=post_weights,
        doc_ids=doc_ids,
        metadata=metadata,
    )


def sparsify_query_vector(
    vector: np.ndarray,
    *,
    exclude_token_ids: Sequence[int],
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
    if exclude_token_ids:
        exclude_array: np.ndarray = np.asarray(exclude_token_ids, dtype=np.int64)
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


def _estimate_posting_length(term_ptr: np.ndarray, q_indices: np.ndarray) -> int:
    """Estimate total postings length for the query terms."""
    if q_indices.size == 0:
        return 0
    term_next: np.ndarray = term_ptr[q_indices + 1]
    term_curr: np.ndarray = term_ptr[q_indices]
    total: int = int(np.sum(term_next - term_curr))
    return total


@numba.njit
def _fill_postings(
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    doc_offset: int,
    term_offsets: np.ndarray,
    post_doc_ids: np.ndarray,
    post_weights: np.ndarray,
) -> None:
    doc_count: int = indptr.shape[0] - 1
    for doc_idx in range(doc_count):
        start: int = int(indptr[doc_idx])
        end: int = int(indptr[doc_idx + 1])
        doc_id: int = int(doc_offset + doc_idx)
        for pos in range(start, end):
            term_id: int = int(indices[pos])
            write_pos: int = int(term_offsets[term_id])
            post_doc_ids[write_pos] = doc_id
            post_weights[write_pos] = values[pos]
            term_offsets[term_id] = write_pos + 1


def build_inverted_index_from_shards(
    shard_infos: Sequence[ShardInfo],
    vocab_size: int,
    *,
    value_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build inverted index arrays from sparse shards."""
    term_counts: np.ndarray = np.zeros(int(vocab_size), dtype=np.int64)
    doc_ids: list[str] = []

    for shard in shard_infos:
        indices: np.ndarray = np.load(shard.indices_path)
        if int(indices.size) > 0:
            term_counts += np.bincount(indices, minlength=int(vocab_size)).astype(
                np.int64, copy=False
            )
        with shard.doc_ids_path.open("r", encoding="utf-8") as doc_file:
            shard_doc_ids: list[str] = json.load(doc_file)
        doc_ids.extend(shard_doc_ids)

    term_ptr: np.ndarray = np.zeros(int(vocab_size) + 1, dtype=np.int64)
    term_ptr[1:] = np.cumsum(term_counts)
    total_nnz: int = int(term_ptr[-1])

    post_doc_ids: np.ndarray = np.empty(total_nnz, dtype=np.int32)
    post_weights: np.ndarray = np.empty(total_nnz, dtype=value_dtype)
    term_offsets: np.ndarray = term_ptr[:-1].copy()

    doc_offset: int = 0
    for shard in shard_infos:
        indptr: np.ndarray = np.load(shard.indptr_path)
        indices: np.ndarray = np.load(shard.indices_path)
        values: np.ndarray = np.load(shard.values_path)
        _fill_postings(
            indptr=indptr,
            indices=indices,
            values=values,
            doc_offset=doc_offset,
            term_offsets=term_offsets,
            post_doc_ids=post_doc_ids,
            post_weights=post_weights,
        )
        doc_offset += int(indptr.shape[0]) - 1

    return term_ptr, post_doc_ids, post_weights, doc_ids


@numba.njit
def _accumulate_scores(
    term_ptr: np.ndarray,
    post_doc_ids: np.ndarray,
    post_weights: np.ndarray,
    q_indices: np.ndarray,
    q_values: np.ndarray,
    scores: np.ndarray,
    seen: np.ndarray,
    touched: np.ndarray,
) -> int:
    # Accumulate scores and track which docs were touched.
    touched_count: int = 0
    for idx in range(q_indices.shape[0]):
        term_id: int = int(q_indices[idx])
        q_weight: float = float(q_values[idx])
        start: int = int(term_ptr[term_id])
        end: int = int(term_ptr[term_id + 1])
        for pos in range(start, end):
            doc_id: int = int(post_doc_ids[pos])
            if seen[doc_id] == 0:
                seen[doc_id] = 1
                touched[touched_count] = doc_id
                touched_count += 1
            scores[doc_id] += q_weight * float(post_weights[pos])
    return touched_count


def score_query_postings(
    term_ptr: np.ndarray,
    post_doc_ids: np.ndarray,
    post_weights: np.ndarray,
    q_indices: np.ndarray,
    q_values: np.ndarray,
    *,
    scores: np.ndarray,
    seen: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a query against an inverted index and return top-k results."""
    if top_k <= 0:
        empty_docs: np.ndarray = np.zeros((0,), dtype=np.int32)
        empty_scores: np.ndarray = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores
    if q_indices.size == 0:
        empty_docs = np.zeros((0,), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores

    total_postings: int = _estimate_posting_length(term_ptr, q_indices)
    if total_postings <= 0:
        empty_docs = np.zeros((0,), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores

    touched: np.ndarray = np.empty(int(total_postings), dtype=np.int32)
    touched_count: int = _accumulate_scores(
        term_ptr=term_ptr,
        post_doc_ids=post_doc_ids,
        post_weights=post_weights,
        q_indices=q_indices,
        q_values=q_values,
        scores=scores,
        seen=seen,
        touched=touched,
    )
    if touched_count <= 0:
        empty_docs = np.zeros((0,), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return empty_docs, empty_scores

    touched_docs: np.ndarray = touched[:touched_count]
    touched_scores: np.ndarray = scores[touched_docs]

    if touched_count <= top_k:
        order: np.ndarray = np.argsort(touched_scores)[::-1]
        top_docs: np.ndarray = touched_docs[order]
        top_scores: np.ndarray = touched_scores[order]
    else:
        top_k_int: int = int(top_k)
        top_positions: np.ndarray = np.argpartition(touched_scores, -top_k_int)[
            -top_k_int:
        ]
        top_docs = touched_docs[top_positions]
        top_scores = touched_scores[top_positions]
        order = np.argsort(top_scores)[::-1]
        top_docs = top_docs[order]
        top_scores = top_scores[order]

    # Reset buffers for the next query.
    scores[touched_docs] = 0.0
    seen[touched_docs] = 0
    return top_docs.astype(np.int32, copy=False), top_scores.astype(
        np.float32, copy=False
    )


__all__ = [
    "InvertedIndex",
    "ShardInfo",
    "SparseShardWriter",
    "build_inverted_index_from_shards",
    "load_inverted_index",
    "load_shard_manifest",
    "resolve_numpy_dtype",
    "score_query_postings",
    "sparsify_query_vector",
]
