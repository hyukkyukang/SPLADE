"""Retrieval orchestration for index-based search."""

import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from omegaconf import DictConfig

from src.metric.retrieval import resolve_k_list
from src.search.buffers import prepare_score_buffers, resolve_query_sparsify_config
from src.search.index import InvertedIndex, load_inverted_index
from src.search.scoring import (
    score_query_postings,
    score_query_postings_bmw,
    score_query_postings_wand,
)
from src.search.scoring_gpu import GpuIndex, build_gpu_index, score_batch_gpu
from src.search.sparsify import sparsify_batch_gpu_csr, sparsify_query_vector
from src.utils.model_utils import resolve_tagged_output_dir
from src.utils.output_space import resolve_model_output_exclude_ids
from src.utils.windowed_encoding import encode_and_aggregate_windows


_PROCESS_INDEX: InvertedIndex | None = None
_PROCESS_BUFFERS: tuple[np.ndarray, np.ndarray] | None = None
_PROCESS_SCORE_METHOD: str = "full"
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
        if _PROCESS_INDEX.term_max is None:
            raise ValueError("WAND scoring requires term_max bounds. Rebuild the index.")
    if _PROCESS_SCORE_METHOD == "bmw":
        if (
            _PROCESS_INDEX.term_max is None
            or _PROCESS_INDEX.block_max is None
            or _PROCESS_INDEX.block_ptr is None
        ):
            raise ValueError(
                "BMW scoring requires block max bounds. Rebuild the index."
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
            q_indices,
            q_values,
            scores=scores,
            seen=seen,
            top_k=_PROCESS_K_MAX,
        )
    if _PROCESS_SCORE_METHOD == "bmw":
        return score_query_postings_bmw(
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
        self._exclude_self_match: bool = bool(
            self.cfg.testing.get("exclude_self_match", False)
        )
        self._result_group_key: str = str(
            self.cfg.testing.get("result_group_key", "none")
        ).strip().lower()
        self._group_candidate_pool: int | None = (
            None
            if self.cfg.testing.get("group_candidate_pool") is None
            else max(1, int(self.cfg.testing.group_candidate_pool))
        )
        self._configured_search_top_k: int | None = (
            None
            if self.cfg.testing.get("search_top_k") is None
            else max(1, int(self.cfg.testing.search_top_k))
        )
        self._scoring_top_k: int = self.k_max + (
            1 if self._exclude_self_match else 0
        )
        if self._configured_search_top_k is not None:
            self._scoring_top_k = int(self._configured_search_top_k)

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
        if scoring_backend_normalized not in {"threads", "processes", "gpu"}:
            raise ValueError(
                "testing.scoring_backend must be 'threads', 'processes', or 'gpu'. "
                f"Got: {scoring_backend_value}"
            )
        self._scoring_backend: str = scoring_backend_normalized

        gpu_device_value = self.cfg.testing.get("gpu_scoring_device", "cuda:0")
        self._gpu_scoring_device: str = str(gpu_device_value)
        gpu_values_dtype_value = str(
            self.cfg.testing.get("gpu_scoring_values_dtype", "float16")
        ).lower()
        if gpu_values_dtype_value not in {"float16", "float32"}:
            raise ValueError(
                "testing.gpu_scoring_values_dtype must be 'float16' or 'float32'. "
                f"Got: {gpu_values_dtype_value}"
            )
        self._gpu_scoring_values_dtype: torch.dtype = (
            torch.float16 if gpu_values_dtype_value == "float16" else torch.float32
        )
        gpu_query_chunk_value = self.cfg.testing.get("gpu_query_chunk")
        self._gpu_query_chunk: int | None = (
            None if gpu_query_chunk_value is None else max(1, int(gpu_query_chunk_value))
        )
        self._gpu_index: GpuIndex | None = None

        self._index: InvertedIndex | None = None
        self._index_path: Path | None = None
        self._doc_ids: list[str] | None = None
        self._group_ids: list[str] | None = None
        self._doc_count: int = 0
        self._block_size: int = 0
        self._query_exclude_token_ids: list[int] = []
        self._query_exclude_output_ids: list[int] = []
        self._query_exclude_output_ids_tensor: torch.Tensor | None = None
        self._query_exclude_output_ids_resolved: bool = True
        self._query_min_weight: float = 0.0
        self._query_top_k: int | None = None

        self._thread_local: threading.local = threading.local()
        self._executor: ThreadPoolExecutor | None = None
        self._resolved_workers: int = 1
        max_windows_value: Any | None = self.cfg.testing.get("max_windows_per_forward")
        self._max_windows_per_forward: int | None = (
            None if max_windows_value is None else max(1, int(max_windows_value))
        )
        self._use_fixed_window_chunks: bool = bool(self.cfg.testing.torch_compile) and (
            self._max_windows_per_forward is not None
        )

    @property
    def doc_ids(self) -> list[str]:
        if self._doc_ids is None:
            raise ValueError("Index must be loaded before accessing doc_ids.")
        return self._doc_ids

    def setup(self) -> None:
        """Load index metadata and initialize scoring resources."""
        self._index = self._load_index()
        self._doc_ids = list(self._index.doc_ids)
        self._group_ids = (
            None if self._index.group_ids is None else list(self._index.group_ids)
        )
        self._doc_count = len(self._doc_ids)
        self._block_size = int(self._index.metadata.get("block_size") or 0)
        if self._uses_grouped_results():
            if self._group_ids is None:
                raise ValueError(
                    "testing.result_group_key requested grouped sparse retrieval, "
                    "but the loaded sparse index does not provide group_ids."
                )
            if self._group_candidate_pool is None:
                self._group_candidate_pool = (
                    int(self._configured_search_top_k)
                    if self._configured_search_top_k is not None
                    else max(self.k_max * 8, 2048)
                )
            self._scoring_top_k = max(self._scoring_top_k, int(self._group_candidate_pool))
        self._validate_scoring_method()
        (
            self._query_exclude_token_ids,
            self._query_min_weight,
            self._query_top_k,
        ) = resolve_query_sparsify_config(self.cfg)
        self._query_exclude_output_ids = []
        self._query_exclude_output_ids_tensor = None
        self._query_exclude_output_ids_resolved = not bool(self._query_exclude_token_ids)
        if self._scoring_backend == "gpu":
            if self._use_cpu:
                raise ValueError(
                    "testing.scoring_backend=gpu is incompatible with testing.use_cpu=true."
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "testing.scoring_backend=gpu requested but CUDA is not available."
                )
            if self._scoring_method != "full":
                self._logger.warning(
                    "GPU scoring backend ignores scoring_method=%s and computes exact "
                    "dense matmul scores (equivalent to 'full').",
                    self._scoring_method,
                )
            self._gpu_index = build_gpu_index(
                self._index,
                values_dtype=self._gpu_scoring_values_dtype,
                device=self._gpu_scoring_device,
            )
            self._logger.info(
                "GPU scoring index ready: n_docs=%d vocab=%d dtype=%s device=%s",
                self._gpu_index.n_docs,
                self._gpu_index.vocab_size,
                str(self._gpu_scoring_values_dtype),
                str(self._gpu_index.device),
            )
            self._executor = None
            self._resolved_workers = 1
        else:
            self._start_executor()

    def _resolve_query_exclude_output_ids(self, model: Any) -> None:
        if self._query_exclude_output_ids_resolved:
            return
        raw_exclude_token_ids: list[int] = list(self._query_exclude_token_ids)
        if not raw_exclude_token_ids:
            self._query_exclude_output_ids = []
            self._query_exclude_output_ids_tensor = None
            self._query_exclude_output_ids_resolved = True
            return

        self._query_exclude_output_ids = resolve_model_output_exclude_ids(
            model,
            raw_exclude_token_ids,
        )
        if self._query_exclude_output_ids:
            self._query_exclude_output_ids_tensor = torch.tensor(
                self._query_exclude_output_ids, dtype=torch.long
            )
        else:
            self._query_exclude_output_ids_tensor = None
        self._query_exclude_output_ids_resolved = True

    def shutdown(self) -> None:
        """Release any per-test resources such as thread pools."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._thread_local = threading.local()
        self._group_ids = None
        self._gpu_index = None

    def encode_queries(
        self,
        model: Any,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        mark_step: Callable[[], None] | None,
        query_pooling_mask: torch.Tensor | None = None,
        query_indptr: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._resolve_query_exclude_output_ids(model)
        if query_indptr is None:
            if mark_step is not None:
                mark_step()
            return model.encode_queries(
                query_input_ids,
                query_attention_mask,
                pooling_mask=query_pooling_mask,
            )
        return self._encode_and_aggregate_query_windows(
            model=model,
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            pooling_mask=query_pooling_mask,
            query_indptr=query_indptr,
            mark_step=mark_step,
        )

    def score_queries(
        self,
        query_reps: torch.Tensor,
        *,
        query_ids: Sequence[str] | None = None,
    ) -> list[tuple[list[str], list[float]]]:
        if self._index is None or self._doc_ids is None:
            raise ValueError("Index must be loaded before scoring queries.")
        if self._exclude_self_match and query_ids is None:
            raise ValueError(
                "query_ids must be provided when testing.exclude_self_match=true."
            )
        query_ids_list: list[str] | None = (
            None if query_ids is None else [str(query_id) for query_id in query_ids]
        )
        q_indices_list, q_values_list = self._sparsify_queries(query_reps)
        scored = self._score_batch(q_indices_list, q_values_list)
        results: list[tuple[list[str], list[float]]] = []
        for query_idx, (top_docs, top_scores) in enumerate(scored):
            if self._uses_grouped_results():
                results.append(
                    self._score_grouped_row(
                        top_docs.tolist(),
                        top_scores.tolist(),
                        query_id=(
                            None if query_ids_list is None else query_ids_list[query_idx]
                        ),
                    )
                )
                continue
            raw_doc_ids: list[str] = [
                self._doc_ids[int(doc_idx)] for doc_idx in top_docs.tolist()
            ]
            raw_scores: list[float] = [
                float(score) for score in top_scores.tolist()
            ]
            if self._exclude_self_match and query_ids_list is not None:
                query_id: str = query_ids_list[query_idx]
                selected_doc_ids: list[str] = []
                selected_scores: list[float] = []
                doc_id: str
                score: float
                for doc_id, score in zip(raw_doc_ids, raw_scores):
                    if doc_id == query_id:
                        continue
                    selected_doc_ids.append(doc_id)
                    selected_scores.append(score)
                    if len(selected_doc_ids) >= self.k_max:
                        break
            else:
                selected_doc_ids = raw_doc_ids[: self.k_max]
                selected_scores = raw_scores[: self.k_max]
            results.append((selected_doc_ids, selected_scores))
        return results

    def _uses_grouped_results(self) -> bool:
        return self._result_group_key not in {"", "none", "doc_id", "passage_id"}

    def _score_grouped_row(
        self,
        doc_indexes: Sequence[int],
        scores: Sequence[float],
        *,
        query_id: str | None,
    ) -> tuple[list[str], list[float]]:
        if self._group_ids is None:
            raise ValueError("Grouped sparse retrieval requires loaded group ids.")
        grouped_scores: dict[str, float] = {}
        for doc_idx, score in zip(doc_indexes, scores):
            if int(doc_idx) < 0:
                continue
            group_id: str = self._group_ids[int(doc_idx)]
            if self._exclude_self_match and query_id is not None and group_id == query_id:
                continue
            score_value: float = float(score)
            existing_score: float | None = grouped_scores.get(group_id)
            if existing_score is None or score_value > existing_score:
                grouped_scores[group_id] = score_value

        sorted_groups: list[tuple[str, float]] = sorted(
            grouped_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: self.k_max]
        return (
            [group_id for group_id, _ in sorted_groups],
            [float(score) for _, score in sorted_groups],
        )

    @staticmethod
    def _resolve_pad_token_id(model: Any) -> int:
        mlm: Any | None = getattr(getattr(model, "encoder", None), "mlm", None)
        config: Any | None = None if mlm is None else getattr(mlm, "config", None)
        pad_token_id: Any | None = (
            None if config is None else getattr(config, "pad_token_id", None)
        )
        return 0 if pad_token_id is None else int(pad_token_id)

    def _encode_and_aggregate_query_windows(
        self,
        *,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None,
        query_indptr: Sequence[int] | torch.Tensor,
        mark_step: Callable[[], None] | None,
    ) -> torch.Tensor:
        return encode_and_aggregate_windows(
            input_ids,
            attention_mask,
            pooling_mask,
            indptr=query_indptr,
            encode_fn=lambda chunk_input_ids, chunk_attention_mask, chunk_pooling_mask: (
                model.encode_queries(
                    chunk_input_ids,
                    chunk_attention_mask,
                    pooling_mask=chunk_pooling_mask,
                )
            ),
            pooling_mode=str(model.query_pooling),
            output_dim=int(model.encoder.vocab_size),
            output_dtype=next(model.parameters()).dtype,
            pad_token_id=self._resolve_pad_token_id(model),
            chunk_size=self._max_windows_per_forward,
            use_fixed_size_chunks=self._use_fixed_window_chunks,
            mark_step=mark_step,
            entity_name="query",
        )

    def _load_index(self) -> InvertedIndex:
        index_dir_value: str | None = self.cfg.encoding.index_dir
        if not index_dir_value:
            raise ValueError(
                "encoding.index_dir must be set for index-based "
                f"{self._index_context}."
            )
        index_tag_value: object | None = self.cfg.encoding.index_tag
        index_path = resolve_tagged_output_dir(
            index_dir_value,
            model_name=str(self.cfg.model.name),
            tag=index_tag_value,
        )
        self._index_path = index_path
        return load_inverted_index(index_path)

    def _validate_scoring_method(self) -> None:
        if self._scoring_method not in {"full", "wand", "bmw"}:
            raise ValueError(
                "testing.scoring_method must be 'full', 'wand', or 'bmw'. "
                f"Got: {self._scoring_method}"
            )
        if self._scoring_method == "wand":
            if self._index is None:
                raise ValueError("Index must be loaded before validating scoring.")
            if self._index.term_max is None:
                raise ValueError(
                    "WAND scoring requires term max bounds. Rebuild the index."
                )
        if self._scoring_method == "bmw":
            if self._index is None:
                raise ValueError("Index must be loaded before validating scoring.")
            if (
                self._index.term_max is None
                or self._index.block_max is None
                or self._index.block_ptr is None
            ):
                raise ValueError(
                    "BMW scoring requires block max bounds. "
                    "Rebuild the index with wand_block_size enabled."
                )
            if self._block_size <= 0:
                raise ValueError("Index metadata is missing a valid block_size.")
            cfg_block_size = int(self.cfg.testing.wand_block_size)
            if cfg_block_size != self._block_size:
                self._logger.warning(
                    "BMW block size mismatch (index=%s, config=%s). Using index value.",
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
                self._prewarm_numba()
                self._executor = ProcessPoolExecutor(
                    max_workers=self._resolved_workers,
                    initializer=_init_process_pool,
                    initargs=(
                        str(self._index_path),
                        self._scoring_method,
                        int(self._scoring_top_k),
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

    def _prewarm_numba(self) -> None:
        """Trigger numba JIT in the parent process so workers hit the disk cache.

        Why: numba @njit(cache=True) compiles once per (function, signature) and
        caches to disk. Compiling in the parent first means ProcessPoolExecutor
        workers load from cache instead of each re-JIT-ing independently.
        """
        if self._index is None:
            return
        index: InvertedIndex = self._index
        doc_count: int = len(index.doc_ids)
        scores, seen = prepare_score_buffers(doc_count)
        q_indices: np.ndarray = np.zeros(1, dtype=np.int32)
        q_values: np.ndarray = np.ones(1, dtype=np.float32)
        method: str = self._scoring_method
        top_k: int = max(1, int(self._scoring_top_k))
        if method == "wand" and index.term_max is not None:
            score_query_postings_wand(
                index.term_ptr,
                index.post_doc_ids,
                index.post_weights,
                index.term_max,
                q_indices,
                q_values,
                scores=scores,
                seen=seen,
                top_k=top_k,
            )
        elif (
            method == "bmw"
            and index.term_max is not None
            and index.block_max is not None
            and index.block_ptr is not None
        ):
            block_size: int = int(index.metadata.get("block_size") or 0)
            if block_size > 0:
                score_query_postings_bmw(
                    index.term_ptr,
                    index.post_doc_ids,
                    index.post_weights,
                    index.term_max,
                    index.block_max,
                    index.block_ptr,
                    q_indices,
                    q_values,
                    scores=scores,
                    seen=seen,
                    top_k=top_k,
                    block_size=block_size,
                )
        else:
            score_query_postings(
                index.term_ptr,
                index.post_doc_ids,
                index.post_weights,
                q_indices,
                q_values,
                scores=scores,
                seen=seen,
                top_k=top_k,
            )

    def _get_thread_buffers(self) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self._thread_local, "buffers"):
            self._thread_local.buffers = prepare_score_buffers(self._doc_count)
        return self._thread_local.buffers

    def _sparsify_queries(
        self, query_reps: torch.Tensor
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self._query_exclude_token_ids and not self._query_exclude_output_ids_resolved:
            raise RuntimeError(
                "Query output exclusions are unresolved. Call encode_queries() with "
                "the active model before score_queries()."
            )
        if self._gpu_sparsify and not self._use_cpu and query_reps.is_cuda:
            indptr, indices, values = sparsify_batch_gpu_csr(
                query_reps,
                exclude_output_ids=self._query_exclude_output_ids_tensor,
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
                exclude_output_ids=self._query_exclude_output_ids,
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
        if self._scoring_backend == "gpu":
            if self._gpu_index is None:
                raise RuntimeError(
                    "GPU scoring backend selected but GPU index is not initialized. "
                    "Call setup() before scoring."
                )
            return score_batch_gpu(
                self._gpu_index,
                q_indices_list,
                q_values_list,
                top_k=self._scoring_top_k,
                query_chunk=self._gpu_query_chunk,
            )
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
                q_indices,
                q_values,
                scores=scores,
                seen=seen,
                top_k=self._scoring_top_k,
            )
        if self._scoring_method == "bmw":
            return score_query_postings_bmw(
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
                top_k=self._scoring_top_k,
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
            top_k=self._scoring_top_k,
        )


__all__ = ["IndexedRetrievalHelper"]
