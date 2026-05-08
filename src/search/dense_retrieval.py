"""Dense retrieval orchestration for FAISS-backed search."""

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import faiss
import numpy as np
import torch
from omegaconf import DictConfig

from src.metric.retrieval import resolve_k_list
from src.search.index import DenseFaissIndex, load_dense_faiss_index
from src.utils.model_utils import resolve_tagged_output_dir
from src.utils.windowed_encoding import encode_and_aggregate_windows


def _normalize_query_vectors(vectors: np.ndarray) -> np.ndarray:
    norms: np.ndarray = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


class DenseRetrievalHelper:
    """Shared helper for dense index-based retrieval and scoring."""

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
        self._search_top_k: int = (
            self.k_max + (1 if self._exclude_self_match else 0)
            if self._configured_search_top_k is None
            else int(self._configured_search_top_k)
        )
        self._index: DenseFaissIndex | None = None
        self._search_index: faiss.Index | None = None
        self._doc_ids: list[str] | None = None
        self._group_ids: list[str] | None = None
        self._similarity: str = "dot"
        self._normalized: bool = False
        self._use_gpu: bool = bool(self.cfg.testing.get("faiss_use_gpu", True))
        self._gpu_required: bool = bool(
            self.cfg.testing.get("faiss_gpu_required", False)
        )
        self._gpu_shard: bool = bool(self.cfg.testing.get("faiss_gpu_shard", False))
        self._gpu_use_float16: bool = bool(
            self.cfg.testing.get("faiss_use_float16", True)
        )
        self._max_windows_per_forward: int | None = (
            None
            if self.cfg.testing.get("max_windows_per_forward") is None
            else max(1, int(self.cfg.testing.max_windows_per_forward))
        )
        self._use_fixed_window_chunks: bool = bool(self.cfg.testing.torch_compile) and (
            self._max_windows_per_forward is not None
        )
        self._gpu_resources: Any | None = None

    def setup(self, *, device_index: int | None = None) -> None:
        self._index = self._load_index()
        self._doc_ids = list(self._index.doc_ids)
        self._group_ids = (
            None if self._index.group_ids is None else list(self._index.group_ids)
        )
        self._similarity = str(self._index.metadata.get("similarity", "dot")).lower()
        self._normalized = bool(self._index.metadata.get("normalized", False))
        if self._uses_grouped_results():
            if self._group_ids is None:
                raise ValueError(
                    "testing.result_group_key requested grouped dense retrieval, "
                    "but the loaded FAISS index does not provide group_ids."
                )
            if self._group_candidate_pool is None:
                self._group_candidate_pool = (
                    int(self._configured_search_top_k)
                    if self._configured_search_top_k is not None
                    else max(self.k_max * 8, 2048)
                )
            self._search_top_k = max(self._search_top_k, int(self._group_candidate_pool))
        self._search_index = self._maybe_clone_index_to_gpu(
            self._index.index,
            device_index=device_index,
        )

    def shutdown(self) -> None:
        self._search_index = None
        self._gpu_resources = None
        self._group_ids = None

    def encode_queries(
        self,
        model: Any,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        mark_step: Callable[[], None] | None,
        query_pooling_mask: torch.Tensor | None = None,
        query_indptr: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_indptr is None:
            if mark_step is not None:
                mark_step()
            return model.encode_queries(
                query_input_ids,
                query_attention_mask,
                pooling_mask=query_pooling_mask,
            )
        embeddings: torch.Tensor = encode_and_aggregate_windows(
            query_input_ids,
            query_attention_mask,
            query_pooling_mask,
            indptr=query_indptr,
            encode_fn=lambda chunk_input_ids, chunk_attention_mask, chunk_pooling_mask: (
                model._query_encoder_fn(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                    pooling_mask=chunk_pooling_mask,
                )
            ),
            pooling_mode=str(
                getattr(model, "query_window_pooling", model.query_pooling)
            ),
            output_dim=int(model.embedding_dim),
            output_dtype=next(model.parameters()).dtype,
            pad_token_id=self._resolve_pad_token_id(model),
            chunk_size=self._max_windows_per_forward,
            use_fixed_size_chunks=self._use_fixed_window_chunks,
            mark_step=mark_step,
            entity_name="query",
        )
        return model.postprocess_query_embeddings(embeddings)

    def score_queries(
        self,
        query_reps: torch.Tensor,
        *,
        query_ids: Sequence[str] | None = None,
    ) -> list[tuple[list[str], list[float]]]:
        if self._search_index is None or self._doc_ids is None:
            raise ValueError("Dense index must be loaded before scoring queries.")
        if self._exclude_self_match and query_ids is None:
            raise ValueError(
                "query_ids must be provided when testing.exclude_self_match=true."
            )
        query_vectors: np.ndarray = query_reps.detach().float().cpu().numpy()
        if self._similarity == "cosine":
            query_vectors = _normalize_query_vectors(query_vectors)
        scores: np.ndarray
        doc_indexes: np.ndarray
        scores, doc_indexes = self._search_index.search(query_vectors, self._search_top_k)
        query_ids_list: list[str] | None = (
            None if query_ids is None else [str(query_id) for query_id in query_ids]
        )
        results: list[tuple[list[str], list[float]]] = []
        for row_idx in range(int(doc_indexes.shape[0])):
            if self._uses_grouped_results():
                results.append(
                    self._score_grouped_row(
                        doc_indexes[row_idx].tolist(),
                        scores[row_idx].tolist(),
                        query_id=(
                            None if query_ids_list is None else query_ids_list[row_idx]
                        ),
                    )
                )
                continue
            selected_doc_ids: list[str] = []
            selected_scores: list[float] = []
            query_id: str | None = (
                None if query_ids_list is None else query_ids_list[row_idx]
            )
            for doc_idx, score in zip(doc_indexes[row_idx].tolist(), scores[row_idx].tolist()):
                if int(doc_idx) < 0:
                    continue
                doc_id: str = self._doc_ids[int(doc_idx)]
                if self._exclude_self_match and query_id is not None and doc_id == query_id:
                    continue
                selected_doc_ids.append(doc_id)
                selected_scores.append(float(score))
                if len(selected_doc_ids) >= self.k_max:
                    break
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
        if self._doc_ids is None or self._group_ids is None:
            raise ValueError("Grouped dense retrieval requires loaded group ids.")
        grouped_scores: dict[str, float] = {}
        rank_descending: bool = self._similarity != "l2"
        for doc_idx, score in zip(doc_indexes, scores):
            if int(doc_idx) < 0:
                continue
            group_id: str = self._group_ids[int(doc_idx)]
            if self._exclude_self_match and query_id is not None and group_id == query_id:
                continue
            score_value: float = float(score)
            existing_score: float | None = grouped_scores.get(group_id)
            if existing_score is None:
                grouped_scores[group_id] = score_value
                continue
            if rank_descending and score_value > existing_score:
                grouped_scores[group_id] = score_value
            elif not rank_descending and score_value < existing_score:
                grouped_scores[group_id] = score_value

        sorted_groups: list[tuple[str, float]] = sorted(
            grouped_scores.items(),
            key=lambda item: item[1],
            reverse=rank_descending,
        )[: self.k_max]
        return (
            [group_id for group_id, _ in sorted_groups],
            [float(score) for _, score in sorted_groups],
        )

    def _load_index(self) -> DenseFaissIndex:
        index_dir_value: str | None = self.cfg.encoding.index_dir
        if not index_dir_value:
            raise ValueError(
                "encoding.index_dir must be set for index-based "
                f"{self._index_context}."
            )
        index_path: Path = resolve_tagged_output_dir(
            index_dir_value,
            model_name=str(self.cfg.model.name),
            tag=self.cfg.encoding.index_tag,
        )
        return load_dense_faiss_index(index_path)

    @staticmethod
    def _resolve_pad_token_id(model: Any) -> int:
        backbone: Any | None = getattr(getattr(model, "encoder", None), "backbone", None)
        config: Any | None = None if backbone is None else getattr(backbone, "config", None)
        pad_token_id: Any | None = (
            None if config is None else getattr(config, "pad_token_id", None)
        )
        return 0 if pad_token_id is None else int(pad_token_id)

    def _maybe_clone_index_to_gpu(
        self,
        index: faiss.Index,
        *,
        device_index: int | None,
    ) -> faiss.Index:
        if not self._use_gpu or bool(self.cfg.testing.use_cpu):
            return index
        if not torch.cuda.is_available():
            if self._gpu_required:
                raise RuntimeError("testing.faiss_use_gpu=true but CUDA is unavailable.")
            self._logger.warning("CUDA unavailable; falling back to CPU FAISS search.")
            return index
        if not hasattr(faiss, "StandardGpuResources") or not hasattr(
            faiss, "index_cpu_to_gpu"
        ):
            if self._gpu_required:
                raise RuntimeError(
                    "testing.faiss_use_gpu=true but this FAISS build has no GPU support."
                )
            self._logger.warning(
                "FAISS GPU bindings unavailable; falling back to CPU search."
            )
            return index
        resolved_device_indexes: list[int] = self._resolve_gpu_device_indexes(
            device_index=device_index
        )
        clone_target: str = ",".join(str(device_id) for device_id in resolved_device_indexes)
        try:
            if self._gpu_shard and len(resolved_device_indexes) > 1:
                self._gpu_resources = [
                    faiss.StandardGpuResources() for _ in resolved_device_indexes
                ]
                clone_options = faiss.GpuMultipleClonerOptions()
                clone_options.useFloat16 = self._gpu_use_float16
                clone_options.shard = True
                self._logger.info(
                    "Cloning FAISS index across %s GPUs for %s.",
                    len(resolved_device_indexes),
                    self._index_context,
                )
                return faiss.index_cpu_to_gpu_multiple_py(
                    self._gpu_resources,
                    index,
                    co=clone_options,
                    gpus=resolved_device_indexes,
                )

            resolved_device_index: int = int(resolved_device_indexes[0])
            self._gpu_resources = faiss.StandardGpuResources()
            clone_options = faiss.GpuClonerOptions()
            clone_options.useFloat16 = self._gpu_use_float16
            return faiss.index_cpu_to_gpu(
                self._gpu_resources,
                resolved_device_index,
                index,
                clone_options,
            )
        except RuntimeError as exc:
            self._gpu_resources = None
            if self._gpu_required:
                raise
            self._logger.warning(
                "FAISS GPU index clone failed on device(s) %s; falling back to CPU search. "
                "Error: %s",
                clone_target,
                exc,
            )
            return index

    def _resolve_gpu_device_indexes(self, *, device_index: int | None) -> list[int]:
        if self._gpu_shard:
            visible_device_count: int = int(torch.cuda.device_count())
            if visible_device_count > 1:
                return list(range(visible_device_count))
        resolved_device_index: int = (
            int(device_index)
            if device_index is not None
            else int(torch.cuda.current_device())
        )
        return [resolved_device_index]


__all__ = ["DenseRetrievalHelper"]
