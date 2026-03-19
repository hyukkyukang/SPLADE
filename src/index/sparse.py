import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numba
import numpy as np
import torch

from src.utils.logging import get_logger
from src.utils.output_space import OutputSpaceSpec

logger = get_logger("src.index.sparse")


def resolve_numpy_dtype(dtype_name: str) -> np.dtype:
    """Resolve a numpy dtype from a string label."""
    if isinstance(dtype_name, np.dtype):
        return dtype_name
    if isinstance(dtype_name, type) and issubclass(dtype_name, np.generic):
        return np.dtype(dtype_name)

    normalized: str = str(dtype_name).strip()
    if normalized.startswith("<class '") and normalized.endswith("'>"):
        normalized = normalized[len("<class '") : -2]
    if normalized.startswith("numpy."):
        normalized = normalized[len("numpy.") :]
    normalized = normalized.lower()
    if normalized in {"float16", "fp16"}:
        return np.dtype("float16")
    if normalized in {"float32", "fp32"}:
        return np.dtype("float32")
    if normalized in {"float64", "fp64"}:
        return np.dtype("float64")
    try:
        return np.dtype(normalized)
    except TypeError as exc:  # pragma: no cover - invalid dtype strings
        raise ValueError(f"Unsupported numpy dtype: {dtype_name}") from exc


def resolve_torch_dtype(value_dtype: np.dtype) -> torch.dtype:
    """Resolve a torch dtype from a numpy dtype."""
    resolved: np.dtype = np.dtype(value_dtype)
    if resolved == np.float16:
        return torch.float16
    if resolved == np.float32:
        return torch.float32
    if resolved == np.float64:
        return torch.float64
    raise ValueError(f"Unsupported torch dtype for numpy dtype: {resolved}")


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
        exclude_output_ids: Sequence[int],
        source_exclude_token_ids: Sequence[int],
        model_family: str,
        compact_head_alignment: str | None = None,
        output_token_aligned: bool | None = None,
        output_space: OutputSpaceSpec | None = None,
        shard_max_docs: int,
        value_dtype: str,
    ) -> None:
        self.output_dir: Path = output_dir
        self.vocab_size: int = int(vocab_size)
        self.rank: int = int(rank)
        self.top_k: int | None = None if top_k is None else int(top_k)
        self.min_weight: float = float(min_weight)
        self.exclude_output_ids: list[int] = [
            int(output_id) for output_id in exclude_output_ids
        ]
        self.source_exclude_token_ids: list[int] = [
            int(token_id) for token_id in source_exclude_token_ids
        ]
        self.model_family: str = str(model_family)
        self.output_space: OutputSpaceSpec = (
            output_space
            if output_space is not None
            else OutputSpaceSpec.from_alignment(
                vocab_size=self.vocab_size,
                compact_head_alignment=compact_head_alignment,
                output_token_aligned=output_token_aligned,
            )
        )
        self.compact_head_alignment: str = self.output_space.compact_head_alignment
        self.output_token_aligned: bool = self.output_space.output_token_aligned
        self.shard_max_docs: int = max(1, int(shard_max_docs))
        self.value_dtype: np.dtype = resolve_numpy_dtype(value_dtype)

        self._rank_dir: Path = self.output_dir / "shards" / f"rank_{self.rank}"
        self._rank_dir.mkdir(parents=True, exist_ok=True)

        self._exclude_tensor: torch.Tensor | None = None
        if self.exclude_output_ids:
            self._exclude_tensor = torch.tensor(
                self.exclude_output_ids, dtype=torch.long, device="cpu"
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
            values = local_vec[indices]

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

    def write_sparse_batch(
        self,
        doc_ids: Sequence[str],
        indices_list: Sequence[np.ndarray],
        values_list: Sequence[np.ndarray],
    ) -> None:
        if len(doc_ids) == 0:
            return
        if len(doc_ids) != len(indices_list) or len(doc_ids) != len(values_list):
            raise ValueError("Sparse batch inputs must align with doc_ids length.")

        for doc_id, indices, values in zip(doc_ids, indices_list, values_list):
            self._append_doc(str(doc_id), indices, values)
            if len(self._buffer_doc_ids) >= self.shard_max_docs:
                self._flush()

    def write_sparse_csr_batch(
        self,
        doc_ids: Sequence[str],
        indptr: np.ndarray | torch.Tensor,
        indices: np.ndarray | torch.Tensor,
        values: np.ndarray | torch.Tensor,
    ) -> None:
        """Write a CSR batch (indptr/indices/values) for provided doc_ids."""
        if len(doc_ids) == 0:
            return

        indptr_np: np.ndarray = (
            indptr.detach().cpu().numpy() if isinstance(indptr, torch.Tensor) else indptr
        )
        indices_np: np.ndarray = (
            indices.detach().cpu().numpy()
            if isinstance(indices, torch.Tensor)
            else indices
        )
        values_np: np.ndarray = (
            values.detach().cpu().numpy()
            if isinstance(values, torch.Tensor)
            else values
        )

        indptr_np = indptr_np.astype(np.int64, copy=False)
        indices_np = indices_np.astype(np.int32, copy=False)
        values_np = values_np.astype(self.value_dtype, copy=False)

        if indptr_np.shape[0] != len(doc_ids) + 1:
            raise ValueError("CSR indptr length must equal doc_ids length + 1.")

        for doc_idx, doc_id in enumerate(doc_ids):
            start: int = int(indptr_np[doc_idx])
            end: int = int(indptr_np[doc_idx + 1])
            self._append_doc(str(doc_id), indices_np[start:end], values_np[start:end])
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
            "exclude_output_ids": self.exclude_output_ids,
            "source_exclude_token_ids": self.source_exclude_token_ids,
            "value_dtype": str(self.value_dtype),
            "model_family": self.model_family,
            "doc_count": self._total_docs,
            "nnz": self._total_nnz,
            "shards": self._manifest,
        }
        manifest_payload.update(self.output_space.to_metadata_dict())
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
            output_space: OutputSpaceSpec = OutputSpaceSpec.from_metadata(manifest)
            metadata = {
                "vocab_size": manifest.get("vocab_size"),
                "top_k": manifest.get("top_k"),
                "min_weight": manifest.get("min_weight"),
                "exclude_output_ids": manifest.get(
                    "exclude_output_ids", manifest.get("exclude_token_ids")
                ),
                "source_exclude_token_ids": manifest.get(
                    "source_exclude_token_ids", manifest.get("exclude_token_ids")
                ),
                "value_dtype": manifest.get("value_dtype"),
                "model_family": manifest.get("model_family"),
            }
            metadata.update(output_space.to_metadata_dict())

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
        if values.dtype != value_dtype:
            values = values.astype(value_dtype, copy=False)
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
def _compute_term_max(term_ptr: np.ndarray, post_weights: np.ndarray) -> np.ndarray:
    vocab_size: int = term_ptr.shape[0] - 1
    term_max: np.ndarray = np.zeros(vocab_size, dtype=post_weights.dtype)
    for term_id in range(vocab_size):
        start: int = int(term_ptr[term_id])
        end: int = int(term_ptr[term_id + 1])
        if start < end:
            max_val = post_weights[start]
            for pos in range(start + 1, end):
                value = post_weights[pos]
                if value > max_val:
                    max_val = value
            term_max[term_id] = max_val
    return term_max


def compute_term_max(term_ptr: np.ndarray, post_weights: np.ndarray) -> np.ndarray:
    """Compute the max posting weight per term."""
    if term_ptr.ndim != 1:
        raise ValueError("term_ptr must be a 1D array.")
    return _compute_term_max(term_ptr, post_weights)


@numba.njit
def _fill_block_max(
    term_ptr: np.ndarray,
    post_weights: np.ndarray,
    block_ptr: np.ndarray,
    block_size: int,
    block_max: np.ndarray,
) -> None:
    vocab_size: int = term_ptr.shape[0] - 1
    for term_id in range(vocab_size):
        start: int = int(term_ptr[term_id])
        end: int = int(term_ptr[term_id + 1])
        if start >= end:
            continue
        block_base: int = int(block_ptr[term_id])
        block_idx: int = 0
        pos: int = start
        while pos < end:
            block_end: int = pos + block_size
            if block_end > end:
                block_end = end
            max_val = post_weights[pos]
            for offset in range(pos + 1, block_end):
                value = post_weights[offset]
                if value > max_val:
                    max_val = value
            block_max[block_base + block_idx] = max_val
            block_idx += 1
            pos = block_end


def compute_block_max(
    term_ptr: np.ndarray, post_weights: np.ndarray, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-block max weights and block pointers for postings."""
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if term_ptr.ndim != 1:
        raise ValueError("term_ptr must be a 1D array.")

    vocab_size: int = term_ptr.shape[0] - 1
    term_lengths: np.ndarray = term_ptr[1:] - term_ptr[:-1]
    block_counts: np.ndarray = (term_lengths + int(block_size) - 1) // int(block_size)
    block_ptr: np.ndarray = np.zeros(vocab_size + 1, dtype=np.int64)
    if block_counts.size > 0:
        block_ptr[1:] = np.cumsum(block_counts, dtype=np.int64)
    total_blocks: int = int(block_ptr[-1])
    block_max: np.ndarray = np.zeros(total_blocks, dtype=post_weights.dtype)
    if total_blocks > 0:
        _fill_block_max(term_ptr, post_weights, block_ptr, int(block_size), block_max)
    return block_max, block_ptr


def compute_term_and_block_max(
    term_ptr: np.ndarray, post_weights: np.ndarray, block_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute term max plus block max and pointers."""
    term_max: np.ndarray = compute_term_max(term_ptr, post_weights)
    block_max, block_ptr = compute_block_max(term_ptr, post_weights, block_size)
    return term_max, block_max, block_ptr


__all__ = [
    "ShardInfo",
    "SparseShardWriter",
    "build_inverted_index_from_shards",
    "compute_block_max",
    "compute_term_and_block_max",
    "compute_term_max",
    "load_shard_manifest",
    "resolve_numpy_dtype",
    "resolve_torch_dtype",
]
