"""Index loading types for search."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InvertedIndex:
    """Inverted index data loaded from disk."""

    term_ptr: np.ndarray
    post_doc_ids: np.ndarray
    post_weights: np.ndarray
    doc_ids: list[str]
    metadata: dict[str, Any]
    term_max: np.ndarray | None = None
    block_max: np.ndarray | None = None
    block_ptr: np.ndarray | None = None


def load_inverted_index(index_path: Path) -> InvertedIndex:
    """Load an inverted index from disk with memory-mapped arrays."""
    term_ptr_path: Path = index_path / "term_ptr.npy"
    post_doc_ids_path: Path = index_path / "post_doc_ids.npy"
    post_weights_path: Path = index_path / "post_weights.npy"
    term_max_path: Path = index_path / "term_max.npy"
    block_max_path: Path = index_path / "block_max.npy"
    block_ptr_path: Path = index_path / "block_ptr.npy"
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

    has_block_max: bool = bool(metadata.get("has_block_max"))
    term_max: np.ndarray | None = None
    block_max: np.ndarray | None = None
    block_ptr: np.ndarray | None = None
    if has_block_max:
        if not term_max_path.exists():
            raise FileNotFoundError(f"Missing term_max.npy at {term_max_path}")
        if not block_max_path.exists():
            raise FileNotFoundError(f"Missing block_max.npy at {block_max_path}")
        if not block_ptr_path.exists():
            raise FileNotFoundError(f"Missing block_ptr.npy at {block_ptr_path}")
    if term_max_path.exists():
        term_max = np.load(term_max_path, mmap_mode="r")
    if block_max_path.exists():
        block_max = np.load(block_max_path, mmap_mode="r")
    if block_ptr_path.exists():
        block_ptr = np.load(block_ptr_path, mmap_mode="r")

    return InvertedIndex(
        term_ptr=term_ptr,
        post_doc_ids=post_doc_ids,
        post_weights=post_weights,
        doc_ids=doc_ids,
        metadata=metadata,
        term_max=term_max,
        block_max=block_max,
        block_ptr=block_ptr,
    )


__all__ = ["InvertedIndex", "load_inverted_index"]
