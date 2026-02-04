import logging
from typing import Any, Dict, List

import lightning as L
import torch
from omegaconf import DictConfig

from src.metric.retrieval import RetrievalMetrics, resolve_k_list
from src.model.pl_module.utils import (
    build_splade_model_with_checkpoint,
    finalize_retrieval_metrics,
)
from src.model.retriever.sparse.neural.splade import SpladeModel

logger: logging.Logger = logging.getLogger("RerankingLightningModule")


class RerankingLightningModule(L.LightningModule):
    """LightningModule for reranking evaluation."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.automatic_optimization: bool = False
        self.cfg: DictConfig = cfg
        self.save_hyperparameters(cfg)

        self.model: SpladeModel = self._load_model()
        self._k_list: List[int] = resolve_k_list(self.cfg.testing.k_list)
        self.metric_collection: RetrievalMetrics = RetrievalMetrics(
            dataset_name=self.cfg.dataset.name,
            k_list=self._k_list,
            sync_on_compute=False,
        )
        self._local_query_offset: int = 0

    # --- Protected methods ---
    def _load_model(self) -> SpladeModel:
        checkpoint_path: str | None = self.cfg.testing.checkpoint_path
        return build_splade_model_with_checkpoint(
            cfg=self.cfg,
            use_cpu=bool(self.cfg.testing.use_cpu),
            checkpoint_path=checkpoint_path,
            logger=logger,
        )

    def _compute_pairwise_scores(
        self, q_reps: torch.Tensor, doc_reps: torch.Tensor
    ) -> torch.Tensor:
        device_type: str = str(q_reps.device.type)
        q_reps_fp32: torch.Tensor = q_reps.float()
        doc_reps_fp32: torch.Tensor = doc_reps.float()
        # Ensure FP32 matmul to avoid AMP overflow.
        with torch.autocast(device_type=device_type, enabled=False):
            scores_fp32: torch.Tensor = torch.bmm(
                doc_reps_fp32, q_reps_fp32.unsqueeze(2)
            ).squeeze(2)
        return scores_fp32

    # --- Public methods ---
    def on_test_start(self) -> None:
        self._local_query_offset = 0
        self.metric_collection.reset()
        # Keep metrics on CPU to reduce GPU memory pressure.
        self.metric_collection.to(torch.device("cpu"))
        self.model.eval()

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        _ = batch_idx
        query_input_ids: torch.Tensor = batch["query_input_ids"]
        query_attention_mask: torch.Tensor = batch["query_attention_mask"]
        doc_input_ids: torch.Tensor = batch["doc_input_ids"]
        doc_attention_mask: torch.Tensor = batch["doc_attention_mask"]
        doc_mask: torch.Tensor = batch["doc_mask"].to(dtype=torch.bool)
        pos_mask: torch.Tensor = batch["pos_mask"].to(dtype=torch.bool)

        q_reps: torch.Tensor = self.model.encode_queries(
            query_input_ids, query_attention_mask
        )
        bsz: int
        doc_count: int
        seq_len: int
        bsz, doc_count, seq_len = doc_input_ids.shape
        # Flatten docs so the encoder runs on a 2D tensor.
        flat_docs: torch.Tensor = doc_input_ids.view(bsz * doc_count, seq_len)
        flat_masks: torch.Tensor = doc_attention_mask.view(bsz * doc_count, seq_len)
        flat_doc_reps: torch.Tensor = self.model.encode_docs(flat_docs, flat_masks)
        doc_reps: torch.Tensor = flat_doc_reps.view(bsz, doc_count, -1)

        scores: torch.Tensor = self._compute_pairwise_scores(q_reps, doc_reps)
        world_size: int = int(self.trainer.world_size)
        global_rank: int = int(self.trainer.global_rank)
        base_offset: int = self._local_query_offset
        # Track per-rank progress to keep unique global indexes across batches.
        batch_size: int = int(scores.shape[0])
        self._local_query_offset += batch_size

        valid_counts: torch.Tensor = doc_mask.sum(dim=1).to(dtype=torch.long)
        query_indices: torch.Tensor = torch.arange(
            batch_size, device=doc_mask.device, dtype=torch.long
        )
        global_query_indexes: torch.Tensor = global_rank + world_size * (
            base_offset + query_indices
        )
        flat_scores: torch.Tensor = (
            scores.masked_select(doc_mask).detach().float().cpu()
        )
        if int(flat_scores.numel()) == 0:
            return
        flat_labels: torch.Tensor = (
            pos_mask.masked_select(doc_mask).detach().float().cpu()
        )
        flat_indexes: torch.Tensor = torch.repeat_interleave(
            global_query_indexes, valid_counts
        ).to(device=flat_scores.device)
        self.metric_collection.append(flat_scores, flat_labels, flat_indexes)

    def on_test_epoch_end(self) -> None:
        finalize_retrieval_metrics(
            metric_collection=self.metric_collection, module=self, logger=logger
        )
