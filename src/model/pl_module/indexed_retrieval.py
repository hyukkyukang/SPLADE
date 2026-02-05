import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig

from src.indexing.sparse_index import (
    InvertedIndex,
    load_inverted_index,
    score_query_postings,
    score_query_postings_wand,
    sparsify_batch_gpu_csr,
    sparsify_query_vector,
)
from src.metric.retrieval import resolve_k_list
from src.model.pl_module.utils import prepare_score_buffers, resolve_query_sparsify_config
from src.utils.model_utils import resolve_tagged_output_dir


_PROCESS_INDEX: InvertedIndex | None = None
_PROCESS_BUFFERS: tuple[np.ndarray, np.ndarray] | None = None
_PROCESS_SCORE_METHOD: str = "exact"
_PROCESS_K_MAX: int = 0
_PROCESS_BLOCK_SIZE: int = 0


def _init_process_pool(index_path: str, scoring_method: str, k_max: int) -> None:
    global _PROCESS_INDEX, _PROCESS_BUFFERS, _PROCESS_SCORE_METHOD, _PROCESS_K_MAX
    global _PROCESS_BLOCK_SIZE
    _PROCESS_INDEX = load_inverted_index(Path(index_path))
    _PROCESS_BUFFERS = None
    _PROCESS_SCORE_METHOD = str(scoring_method).lower()
    _PROCESS_K_MAX = int(k_max)
    _PROCESS_BLOCK_SIZE = int(_PROCESS_INDEX.metadata.get("block_size") or 0)
    if _PROCESS_SCORE_METHOD == "wand":
        if (
            _PROCESS_INDEX.term_max is None
            or _PROCESS_INDEX.block_max is None
            or _PROCESS_INDEX.block_ptr is None
        ):
            raise ValueError(
                "WAND scoring requires block max bounds. Rebuild the index."
            )
        if _PROCESS_BLOCK_SIZE <= 0:
            raise ValueError("Index metadata is missing a valid block_size.")


def _get_process_buffers() -> tuple[np.ndarray, np.ndarray]:
    global _PROCESS_BUFFERS
    if _PROCESS_BUFFERS is None:
        if _PROCESS_INDEX is None:
            raise RuntimeError("Process index is not initialized.")
        _PROCESS_BUFFERS = prepare_score_buffers(len(_PROCESS_INDEX.doc_ids))
    return _PROCESS_BUFFERS


def _score_query_process(
    q_indices: np.ndarray, q_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if _PROCESS_INDEX is None:
        raise RuntimeError("Process index is not initialized.")
    scores, seen = _get_process_buffers()
    if _PROCESS_SCORE_METHOD == "wand":
        return score_query_postings_wand(
            _PROCESS_INDEX.term_ptr,
            _PROCESS_INDEX.post_doc_ids,
            _PROCESS_INDEX.post_weights,
            _PROCESS_INDEX.term_max,
            _PROCESS_INDEX.block_max,
            _PROCESS_INDEX.block_ptr,
            q_indices,
            q_values,
            scores=scores,
            seen=seen,
            top_k=_PROCESS_K_MAX,
            block_size=_PROCESS_BLOCK_SIZE,
        )
    return score_query_postings(
        _PROCESS_INDEX.term_ptr,
        _PROCESS_INDEX.post_doc_ids,
        _PROCESS_INDEX.post_weights,
        q_indices,
        q_values,
        scores=scores,
        seen=seen,
        top_k=_PROCESS_K_MAX,
    )


class IndexedRetrievalHelper:
    """Shared helper for index-based retrieval and scoring."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        logger: logging.Logger,
        index_context: str = "retrieval",
    ) -> None:
        self.cfg: DictConfig = cfg
        self._logger: logging.Logger = logger
        self._index_context: str = index_context

        self.k_list: list[int] = resolve_k_list(self.cfg.testing.k_list)
        self.k_max: int = max(self.k_list)

        self._gpu_sparsify: bool = bool(self.cfg.testing.gpu_sparsify)
        scoring_workers_value = self.cfg.testing.scoring_workers
        self._scoring_workers: int = (
            0 if scoring_workers_value is None else int(scoring_workers_value)
        )
        self._use_cpu: bool = bool(self.cfg.testing.use_cpu)
        scoring_method_value = self.cfg.testing.scoring_method
        self._scoring_method: str = str(scoring_method_value).lower()
        scoring_backend_value = self.cfg.testing.scoring_backend
        scoring_backend_normalized = str(scoring_backend_value).lower()
        if scoring_backend_normalized in {"process", "processes"}:
            scoring_backend_normalized = "processes"
        self._scoring_backend: str = scoring_backend_normalized

        self._index: InvertedIndex | None = None
        self._index_path: Path | None = None
        self._doc_ids: list[str] | None = None
        self._doc_count: int = 0
        self._block_size: int = 0
        self._query_exclude_token_ids: list[int] = []
        self._query_exclude_token_ids_tensor: torch.Tensor | None = None
        self._query_min_weight: float = 0.0
        self._query_top_k: int | None = None

        self._thread_local: threading.local = threading.local()
        self._executor: ThreadPoolExecutor | None = None
        self._resolved_workers: int = 1

    @property
    def doc_ids(self) -> list[str]:
        if self._doc_ids is None:
            raise ValueError("Index must be loaded before accessing doc_ids.")
        return self._doc_ids

    def setup(self) -> None:
        """Load index metadata and initialize scoring resources."""
        self._index = self._load_index()
        self._doc_ids = list(self._index.doc_ids)
        self._doc_count = len(self._doc_ids)
        self._block_size = int(self._index.metadata.get("block_size") or 0)
        self._validate_scoring_method()
        (
            self._query_exclude_token_ids,
            self._query_min_weight,
            self._query_top_k,
        ) = resolve_query_sparsify_config(self._index.metadata)
        if self._query_exclude_token_ids:
            self._query_exclude_token_ids_tensor = torch.tensor(
                self._query_exclude_token_ids, dtype=torch.long
            )
        else:
            self._query_exclude_token_ids_tensor = None
        self._start_executor()

    def shutdown(self) -> None:
        """Release any per-test resources such as thread pools."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._thread_local = threading.local()

    def encode_queries(
        self,
        model: Any,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        mark_step: Callable[[], None] | None,
    ) -> torch.Tensor:
        if mark_step is not None:
            mark_step()
        return model.encode_queries(query_input_ids, query_attention_mask)

    def score_queries(
        self, query_reps: torch.Tensor
    ) -> list[tuple[list[str], list[float]]]:
        if self._index is None or self._doc_ids is None:
            raise ValueError("Index must be loaded before scoring queries.")
        q_indices_list, q_values_list = self._sparsify_queries(query_reps)
        scored = self._score_batch(q_indices_list, q_values_list)
        results: list[tuple[list[str], list[float]]] = []
        for top_docs, top_scores in scored:
            selected_doc_ids: list[str] = [
                self._doc_ids[int(doc_idx)] for doc_idx in top_docs.tolist()
            ]
            selected_scores: list[float] = [
                float(score) for score in top_scores.tolist()
            ]
            results.append((selected_doc_ids, selected_scores))
        return results

    def _load_index(self) -> InvertedIndex:
        index_dir_value: str | None = self.cfg.encoding.index_dir
        if not index_dir_value:
            raise ValueError(
                "encoding.index_dir must be set for index-based "
                f"{self._index_context}."
            )
        index_path = resolve_tagged_output_dir(
            index_dir_value,
            model_name=str(self.cfg.model.name),
            tag=self.cfg.tag,
        )
        self._index_path = index_path
        return load_inverted_index(index_path)

    def _validate_scoring_method(self) -> None:
        if self._scoring_method not in {"exact", "wand"}:
            raise ValueError(
                "testing.scoring_method must be 'exact' or 'wand'. "
                f"Got: {self._scoring_method}"
            )
        if self._scoring_method == "wand":
            if self._index is None:
                raise ValueError("Index must be loaded before validating scoring.")
            if (
                self._index.term_max is None
                or self._index.block_max is None
                or self._index.block_ptr is None
            ):
                raise ValueError(
                    "WAND scoring requires block max bounds. "
                    "Rebuild the index with wand_block_size enabled."
                )
            if self._block_size <= 0:
                raise ValueError("Index metadata is missing a valid block_size.")
            cfg_block_size = int(self.cfg.testing.wand_block_size)
            if cfg_block_size != self._block_size:
                self._logger.warning(
                    "WAND block size mismatch (index=%s, config=%s). Using index value.",
                    self._block_size,
                    cfg_block_size,
                )

    def _resolve_scoring_workers(self) -> int:
        if self._scoring_workers > 0:
            workers = self._scoring_workers
        else:
            cpu_count = os.cpu_count() or 1
            workers = min(4, cpu_count)
        return max(1, int(workers))

    def _start_executor(self) -> None:
        self._resolved_workers = self._resolve_scoring_workers()
        if self._resolved_workers > 1:
            if self._scoring_backend == "processes":
                if self._index_path is None:
                    raise ValueError("Index path must be set before scoring.")
                self._executor = ProcessPoolExecutor(
                    max_workers=self._resolved_workers,
                    initializer=_init_process_pool,
                    initargs=(
                        str(self._index_path),
                        self._scoring_method,
                        int(self.k_max),
                    ),
                )
            else:
                if self._scoring_backend != "threads":
                    raise ValueError(
                        "testing.scoring_backend must be 'threads' or 'processes'. "
                        f"Got: {self._scoring_backend}"
                    )
                self._executor = ThreadPoolExecutor(max_workers=self._resolved_workers)
        else:
            self._executor = None

    def _get_thread_buffers(self) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self._thread_local, "buffers"):
            self._thread_local.buffers = prepare_score_buffers(self._doc_count)
        return self._thread_local.buffers

    def _sparsify_queries(
        self, query_reps: torch.Tensor
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self._gpu_sparsify and not self._use_cpu and query_reps.is_cuda:
            indptr, indices, values = sparsify_batch_gpu_csr(
                query_reps,
                exclude_token_ids=self._query_exclude_token_ids_tensor,
                min_weight=self._query_min_weight,
                top_k=self._query_top_k,
                value_dtype=np.float32,
            )
            indptr_np: np.ndarray = indptr.numpy()
            indices_np: np.ndarray = indices.numpy()
            values_np: np.ndarray = values.numpy()
            q_indices_list: list[np.ndarray] = []
            q_values_list: list[np.ndarray] = []
            for start, end in zip(indptr_np[:-1], indptr_np[1:]):
                q_indices_list.append(indices_np[int(start) : int(end)])
                q_values_list.append(values_np[int(start) : int(end)])
            return q_indices_list, q_values_list

        query_reps_cpu: np.ndarray = query_reps.detach().cpu().float().numpy()
        q_indices_list = []
        q_values_list = []
        for query_vector in query_reps_cpu:
            q_indices, q_values = sparsify_query_vector(
                query_vector,
                exclude_token_ids=self._query_exclude_token_ids,
                min_weight=self._query_min_weight,
                top_k=self._query_top_k,
            )
            q_indices_list.append(q_indices)
            q_values_list.append(q_values)
        return q_indices_list, q_values_list

    def _score_batch(
        self,
        q_indices_list: list[np.ndarray],
        q_values_list: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if self._index is None:
            raise ValueError("Index must be loaded before scoring queries.")
        if not q_indices_list:
            return []
        if self._executor is None or len(q_indices_list) <= 1:
            return [
                self._score_single(q_indices, q_values)
                for q_indices, q_values in zip(q_indices_list, q_values_list)
            ]
        if self._scoring_backend == "processes":
            return list(
                self._executor.map(_score_query_process, q_indices_list, q_values_list)
            )
        return list(self._executor.map(self._score_single, q_indices_list, q_values_list))

    def _score_single(
        self, q_indices: np.ndarray, q_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        scores, seen = self._get_thread_buffers()
        if self._index is None:
            raise ValueError("Index must be loaded before scoring queries.")
        if self._scoring_method == "wand":
            return score_query_postings_wand(
                self._index.term_ptr,
                self._index.post_doc_ids,
                self._index.post_weights,
                self._index.term_max,
                self._index.block_max,
                self._index.block_ptr,
                q_indices,
                q_values,
                scores=scores,
                seen=seen,
                top_k=self.k_max,
                block_size=self._block_size,
            )
        return score_query_postings(
            self._index.term_ptr,
            self._index.post_doc_ids,
            self._index.post_weights,
            q_indices,
            q_values,
            scores=scores,
            seen=seen,
            top_k=self.k_max,
        )


__all__ = ["IndexedRetrievalHelper"]
