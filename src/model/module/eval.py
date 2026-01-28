from __future__ import annotations

import logging
from typing import Any, Dict, List

import lightning as L
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.dataclass import Query
from src.metric.retrieval import RetrievalMetrics, resolve_k_list
from src.model.retriever.sparse.neural.splade import SPLADE
from src.model.retriever.sparse.neural.splade_model import SpladeModel
from src.utils.logging import log_if_rank_zero
from src.utils.model_utils import build_splade_model, load_splade_checkpoint
from src.utils.transformers import build_tokenizer

logger = logging.getLogger("SPLADEEvaluationModule")


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
        self._doc_reps_t: torch.Tensor | None = None
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

        datamodule: Any = self.trainer.datamodule
        if datamodule is None:
            raise ValueError("Evaluation requires a DataModule with BEIRDataset.")

        corpus_docs: list[Any] = datamodule.test_dataset.corpus
        self._doc_ids: List[str]
        doc_reps: torch.Tensor
        self._doc_ids, doc_reps = self._retriever.encode_corpus(corpus_docs)
        self._doc_reps_t = doc_reps.T

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _ = dataloader_idx
        if self._retriever is None or self._doc_reps_t is None or self._doc_ids is None:
            raise ValueError(
                "Retriever and corpus must be initialized in on_test_start."
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
        scores_batch: torch.Tensor = torch.matmul(query_reps, self._doc_reps_t)

        world_size: int = int(self.trainer.world_size)
        global_rank: int = int(self.trainer.global_rank)
        base_offset: int = self._local_query_offset
        # Track per-rank progress to keep unique global indexes across batches.
        self._local_query_offset += len(query_ids)

        for i, (scores, relevance_judgments) in enumerate(
            zip(scores_batch, relevance_judgments_list)
        ):
            topk_indices: torch.Tensor = torch.topk(
                scores, k=min(self._k_max, scores.size(0))
            ).indices
            selected_doc_ids: List[str] = [
                self._doc_ids[idx] for idx in topk_indices.tolist()
            ]
            selected_scores: List[float] = scores[topk_indices].tolist()

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
                final_scores, dtype=torch.float32, device=scores.device
            )
            label_tensor: torch.Tensor = torch.tensor(
                labels, dtype=torch.float32, device=scores.device
            )
            indexes: torch.Tensor = torch.full(
                (len(final_scores),),
                global_query_idx,
                dtype=torch.long,
                device=scores.device,
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
