import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import torch

from src.index.sparse import resolve_numpy_dtype


@dataclass(frozen=True)
class DenseShardInfo:
    """File locations for a single dense shard."""

    rank: int
    shard_id: int
    doc_count: int
    dim: int
    vectors_path: Path
    doc_ids_path: Path
    group_ids_path: Path | None = None


class DenseShardWriter:
    """Write dense document vectors into shard files on disk."""

    def __init__(
        self,
        output_dir: Path,
        dim: int,
        rank: int,
        *,
        model_family: str,
        similarity: str,
        normalized: bool,
        shard_max_docs: int,
        value_dtype: str,
    ) -> None:
        self.output_dir: Path = output_dir
        self.dim: int = int(dim)
        self.rank: int = int(rank)
        self.model_family: str = str(model_family)
        self.similarity: str = str(similarity).strip().lower()
        self.normalized: bool = bool(normalized)
        self.shard_max_docs: int = max(1, int(shard_max_docs))
        self.value_dtype: np.dtype = resolve_numpy_dtype(value_dtype)

        self._rank_dir: Path = self.output_dir / "shards" / f"rank_{self.rank}"
        self._rank_dir.mkdir(parents=True, exist_ok=True)

        self._manifest: list[dict[str, Any]] = []
        self._shard_idx: int = 0
        self._total_docs: int = 0
        self._reset_buffer()

    def _reset_buffer(self) -> None:
        self._buffer_doc_ids: list[str] = []
        self._buffer_group_ids: list[str] = []
        self._buffer_vectors: list[np.ndarray] = []

    def _flush(self) -> None:
        if not self._buffer_doc_ids:
            return

        shard_prefix: Path = self._rank_dir / f"shard_{self._shard_idx:06d}"
        vectors: np.ndarray = np.stack(self._buffer_vectors, axis=0).astype(
            self.value_dtype, copy=False
        )
        vectors_path: Path = Path(f"{shard_prefix}_vectors.npy")
        doc_ids_path: Path = Path(f"{shard_prefix}_doc_ids.json")
        group_ids_path: Path | None = None

        np.save(vectors_path, vectors)
        with doc_ids_path.open("w", encoding="utf-8") as doc_file:
            json.dump(self._buffer_doc_ids, doc_file)
        if self._buffer_group_ids:
            group_ids_path = Path(f"{shard_prefix}_group_ids.json")
            with group_ids_path.open("w", encoding="utf-8") as group_file:
                json.dump(self._buffer_group_ids, group_file)

        manifest_entry: dict[str, Any] = {
            "shard_id": self._shard_idx,
            "doc_count": len(self._buffer_doc_ids),
            "dim": self.dim,
            "vectors": vectors_path.name,
            "doc_ids": doc_ids_path.name,
        }
        if group_ids_path is not None:
            manifest_entry["group_ids"] = group_ids_path.name
        self._manifest.append(manifest_entry)
        self._shard_idx += 1
        self._total_docs += len(self._buffer_doc_ids)
        self._reset_buffer()

    def write_batch(
        self,
        doc_ids: Sequence[str],
        doc_reps: torch.Tensor,
        *,
        doc_group_ids: Sequence[str | None] | None = None,
    ) -> None:
        if len(doc_ids) == 0:
            return
        if int(doc_reps.shape[0]) != len(doc_ids):
            raise ValueError("doc_ids length does not match doc_reps batch size.")
        if doc_group_ids is not None and len(doc_group_ids) != len(doc_ids):
            raise ValueError("doc_group_ids length does not match doc_ids batch size.")
        doc_reps_cpu: torch.Tensor = doc_reps.detach()
        if doc_reps_cpu.is_cuda:
            doc_reps_cpu = doc_reps_cpu.cpu()
        doc_reps_cpu = doc_reps_cpu.float()
        if int(doc_reps_cpu.shape[1]) != self.dim:
            raise ValueError("doc_reps dimension does not match configured dim.")
        for row_idx, (doc_id, vector) in enumerate(zip(doc_ids, doc_reps_cpu)):
            resolved_doc_id: str = str(doc_id)
            self._buffer_doc_ids.append(resolved_doc_id)
            if doc_group_ids is not None:
                raw_group_id: str | None = doc_group_ids[row_idx]
                resolved_group_id: str = (
                    resolved_doc_id
                    if raw_group_id is None or not str(raw_group_id).strip()
                    else str(raw_group_id)
                )
                self._buffer_group_ids.append(resolved_group_id)
            self._buffer_vectors.append(
                vector.numpy().astype(self.value_dtype, copy=False)
            )
            if len(self._buffer_doc_ids) >= self.shard_max_docs:
                self._flush()

    def finalize(self) -> None:
        self._flush()
        manifest_path: Path = self._rank_dir / "manifest.json"
        payload: dict[str, Any] = {
            "index_kind": "dense",
            "rank": self.rank,
            "dim": self.dim,
            "doc_count": self._total_docs,
            "value_dtype": str(self.value_dtype),
            "model_family": self.model_family,
            "similarity": self.similarity,
            "normalized": self.normalized,
            "has_group_ids": any("group_ids" in shard for shard in self._manifest),
            "shards": self._manifest,
        }
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(payload, manifest_file, indent=2)


def load_dense_shard_manifest(encode_path: Path) -> tuple[list[DenseShardInfo], dict[str, Any]]:
    shards_root: Path = encode_path / "shards"
    if not shards_root.exists():
        raise FileNotFoundError(f"Missing shards directory at {shards_root}.")

    shard_infos: list[DenseShardInfo] = []
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
                "index_kind": "dense",
                "dim": int(manifest["dim"]),
                "value_dtype": str(manifest["value_dtype"]),
                "model_family": manifest.get("model_family"),
                "similarity": manifest.get("similarity", "dot"),
                "normalized": bool(manifest.get("normalized", False)),
                "has_group_ids": bool(manifest.get("has_group_ids", False)),
            }
        for shard_payload in manifest.get("shards", []):
            shard_infos.append(
                DenseShardInfo(
                    rank=int(manifest["rank"]),
                    shard_id=int(shard_payload["shard_id"]),
                    doc_count=int(shard_payload["doc_count"]),
                    dim=int(shard_payload["dim"]),
                    vectors_path=rank_dir / str(shard_payload["vectors"]),
                    doc_ids_path=rank_dir / str(shard_payload["doc_ids"]),
                    group_ids_path=(
                        None
                        if shard_payload.get("group_ids") is None
                        else rank_dir / str(shard_payload["group_ids"])
                    ),
                )
            )
    shard_infos.sort(key=lambda info: (info.rank, info.shard_id))
    return shard_infos, metadata


def _resolve_faiss_metric(similarity: str) -> int:
    normalized_similarity: str = str(similarity).strip().lower()
    if normalized_similarity in {"dot", "ip", "cosine"}:
        return faiss.METRIC_INNER_PRODUCT
    if normalized_similarity == "l2":
        return faiss.METRIC_L2
    raise ValueError(f"Unsupported dense similarity: {similarity!r}")


def _build_flat_faiss_index(dim: int, *, similarity: str) -> faiss.Index:
    metric: int = _resolve_faiss_metric(similarity)
    if metric == faiss.METRIC_L2:
        return faiss.IndexFlatL2(int(dim))
    return faiss.IndexFlatIP(int(dim))


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms: np.ndarray = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


def build_dense_faiss_index(
    shard_infos: Sequence[DenseShardInfo],
    *,
    dim: int,
    similarity: str,
    normalized: bool,
) -> tuple[faiss.Index, list[str], list[str] | None]:
    index: faiss.Index = _build_flat_faiss_index(dim, similarity=similarity)
    doc_ids: list[str] = []
    group_ids: list[str] = []
    has_group_ids: bool = False
    shard_info: DenseShardInfo
    for shard_info in shard_infos:
        shard_vectors: np.ndarray = np.load(shard_info.vectors_path, mmap_mode="r")
        vectors: np.ndarray = np.asarray(shard_vectors, dtype=np.float32)
        if similarity == "cosine" and not normalized and vectors.size > 0:
            vectors = _normalize_rows(vectors)
        with shard_info.doc_ids_path.open("r", encoding="utf-8") as doc_file:
            shard_doc_ids: list[str] = json.load(doc_file)
        if int(vectors.shape[0]) != len(shard_doc_ids):
            raise ValueError(
                "Dense shard doc_ids length does not match vector rows: "
                f"{shard_info.vectors_path}"
            )
        shard_group_ids: list[str] | None = None
        if shard_info.group_ids_path is not None:
            with shard_info.group_ids_path.open("r", encoding="utf-8") as group_file:
                shard_group_ids = json.load(group_file)
            if len(shard_group_ids) != len(shard_doc_ids):
                raise ValueError(
                    "Dense shard group_ids length does not match vector rows: "
                    f"{shard_info.group_ids_path}"
                )
            has_group_ids = True
        if vectors.size > 0:
            index.add(vectors)
        doc_ids.extend(str(doc_id) for doc_id in shard_doc_ids)
        if shard_group_ids is not None:
            group_ids.extend(str(group_id) for group_id in shard_group_ids)
    return index, doc_ids, group_ids if has_group_ids else None
