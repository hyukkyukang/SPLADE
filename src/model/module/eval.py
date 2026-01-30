from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.dataclass import Query
from src.indexing.sparse_index import (
    InvertedIndex,
    load_inverted_index,
    score_query_postings,
    sparsify_query_vector,
)
from src.metric.retrieval import RetrievalMetrics, resolve_k_list
from src.model.retriever.sparse.neural.splade import SPLADE, SpladeModel
from src.utils.logging import log_if_rank_zero
from src.utils.model_utils import build_splade_model, load_splade_checkpoint
from src.utils.transformers import build_tokenizer

logger: logging.Logger = logging.getLogger("SPLADEEvaluationModule")


class SPLADEEvaluationModule(L.LightningModule):
    """LightningModule for GenZ-style retrieval evaluation."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.automatic_optimization: bool = False
        self.cfg: DictConfig = cfg
        self.save_hyperparameters(cfg)

        self.model: SpladeModel = self._load_model()
        # Build tokenizer once for retriever setup.
        self._tokenizer: PreTrainedTokenizerBase = build_tokenizer(
            self.cfg.model.huggingface_name
        )
        self._retriever: SPLADE | None = None

        self._k_list: List[int] = resolve_k_list(self.cfg.testing.k_list)
        self._k_max: int = max(self._k_list)
        self.metric_collection: RetrievalMetrics = RetrievalMetrics(
            dataset_name=self.cfg.dataset.name,
            k_list=self._k_list,
            sync_on_compute=False,
        )

        self._doc_ids: List[str] | None = None
        self._index: InvertedIndex | None = None
        self._score_buffer: np.ndarray | None = None
        self._seen_buffer: np.ndarray | None = None
        self._query_exclude_token_ids: list[int] = []
        self._query_min_weight: float = 0.0
        self._query_top_k: int | None = None
        self._local_query_offset: int = 0

    def _load_model(self) -> SpladeModel:
        model: SpladeModel = build_splade_model(
            self.cfg, use_cpu=self.cfg.testing.use_cpu
        )

        checkpoint_path: str | None = getattr(self.cfg.testing, "checkpoint_path", None)
        if checkpoint_path:
            missing: list[str]
            unexpected: list[str]
            missing, unexpected = load_splade_checkpoint(model, checkpoint_path)
            log_if_rank_zero(
                logger,
                f"Loaded checkpoint. Missing: {len(missing)}, unexpected: {len(unexpected)}",
            )

        return model

    def _load_index(self) -> InvertedIndex:
        index_path_value: str | None = getattr(self.cfg.model, "index_path", None)
        if not index_path_value:
            raise ValueError("model.index_path must be set for index-based evaluation.")
        index_path: Path = Path(index_path_value)
        # Load memory-mapped postings for CPU scoring.
        index: InvertedIndex = load_inverted_index(index_path)
        return index

    def _resolve_query_sparsify_config(self, metadata: dict[str, Any]) -> None:
        # Mirror encode-time sparsification settings for queries.
        exclude_ids: list[int] = [
            int(token_id) for token_id in metadata.get("exclude_token_ids") or []
        ]
        min_weight_value: float = float(metadata.get("min_weight") or 0.0)
        top_k_value: int | None = (
            None if metadata.get("top_k") is None else int(metadata["top_k"])
        )
        self._query_exclude_token_ids = exclude_ids
        self._query_min_weight = min_weight_value
        self._query_top_k = top_k_value

    def _prepare_score_buffers(self, doc_count: int) -> None:
        # Allocate reusable buffers to avoid per-query allocations.
        self._score_buffer = np.zeros(int(doc_count), dtype=np.float32)
        self._seen_buffer = np.zeros(int(doc_count), dtype=np.uint8)

    def on_test_start(self) -> None:
        self._local_query_offset = 0
        self.metric_collection.reset()
        self.metric_collection.to(torch.device("cpu"))

        if self._retriever is None:
            self._retriever = SPLADE(
                model=self.model,
                tokenizer=self._tokenizer,
                max_query_length=self.cfg.dataset.max_query_length,
                max_doc_length=self.cfg.dataset.max_doc_length,
                batch_size=self.cfg.testing.batch_size,
                device=self.device,
            )

        index: InvertedIndex = self._load_index()
        self._index = index
        self._doc_ids = list(index.doc_ids)
        self._resolve_query_sparsify_config(index.metadata)
        self._prepare_score_buffers(len(index.doc_ids))

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _ = dataloader_idx
        if (
            self._retriever is None
            or self._index is None
            or self._doc_ids is None
            or self._score_buffer is None
            or self._seen_buffer is None
        ):
            raise ValueError(
                "Retriever and index must be initialized in on_test_start."
            )

        query_texts: List[str] = batch["query_text"]
        qids: List[str] = batch["qid"]
        relevance_judgments_list: List[Dict[str, int]] = batch["relevance_judgments"]

        queries: List[Query] = [
            Query(query_id=str(qid), text=str(text))
            for qid, text in zip(qids, query_texts)
        ]
        query_ids: List[str]
        query_reps: torch.Tensor
        query_ids, query_reps = self._retriever.encode_queries(queries)
        # Score queries on CPU using the inverted index.
        query_reps_cpu: np.ndarray = query_reps.detach().cpu().float().numpy()

        world_size: int = int(self.trainer.world_size)
        global_rank: int = int(self.trainer.global_rank)
        base_offset: int = self._local_query_offset
        # Track per-rank progress to keep unique global indexes across batches.
        self._local_query_offset += len(query_ids)

        for i, relevance_judgments in enumerate(relevance_judgments_list):
            query_vector: np.ndarray = query_reps_cpu[i]
            q_indices: np.ndarray
            q_values: np.ndarray
            q_indices, q_values = sparsify_query_vector(
                query_vector,
                exclude_token_ids=self._query_exclude_token_ids,
                min_weight=self._query_min_weight,
                top_k=self._query_top_k,
            )
            top_docs: np.ndarray
            top_scores: np.ndarray
            top_docs, top_scores = score_query_postings(
                self._index.term_ptr,
                self._index.post_doc_ids,
                self._index.post_weights,
                q_indices,
                q_values,
                scores=self._score_buffer,
                seen=self._seen_buffer,
                top_k=self._k_max,
            )
            selected_doc_ids: List[str] = [
                self._doc_ids[int(doc_idx)] for doc_idx in top_docs.tolist()
            ]
            selected_scores: List[float] = [
                float(score) for score in top_scores.tolist()
            ]

            labels: List[float] = []
            final_scores: List[float] = []
            for doc_id, score in zip(selected_doc_ids, selected_scores):
                relevance: float = float(relevance_judgments.get(str(doc_id), 0))
                labels.append(relevance)
                final_scores.append(float(score))

            min_score: float = min(final_scores) if final_scores else 0.0
            for doc_id, relevance in relevance_judgments.items():
                if relevance > 0 and doc_id not in selected_doc_ids:
                    labels.append(float(relevance))
                    final_scores.append(min_score - 1.0)

            if not final_scores:
                continue

            global_query_idx: int = global_rank + world_size * (base_offset + i)
            score_tensor: torch.Tensor = torch.tensor(
                final_scores, dtype=torch.float32, device=torch.device("cpu")
            )
            label_tensor: torch.Tensor = torch.tensor(
                labels, dtype=torch.float32, device=torch.device("cpu")
            )
            indexes: torch.Tensor = torch.full(
                (len(final_scores),),
                global_query_idx,
                dtype=torch.long,
                device=torch.device("cpu"),
            )
            self.metric_collection.append(score_tensor, label_tensor, indexes)

    def on_test_epoch_end(self) -> None:
        has_data: bool = self.metric_collection.gather(
            world_size=self.trainer.world_size,
            all_gather_fn=self.all_gather if self.trainer.world_size > 1 else None,
        )
        if not has_data:
            log_if_rank_zero(
                logger, "No predictions accumulated during testing.", level="warning"
            )
            return

        if self.trainer.is_global_zero:
            metrics: Dict[str, torch.Tensor] = self.metric_collection.compute()
            self.log_dict(metrics, sync_dist=False, prog_bar=True, rank_zero_only=True)
        self.metric_collection.reset()
