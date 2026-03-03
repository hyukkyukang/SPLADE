from typing import Any, Callable, Dict, List

import lightning as L
import torch
from omegaconf import DictConfig

from src.metric.retrieval import RetrievalMetrics
from src.search.retrieval import IndexedRetrievalHelper
from src.model.pl_module.utils import (
    build_splade_model_with_checkpoint,
    finalize_retrieval_metrics,
    resolve_cudagraph_mark_step,
    validate_torch_compile_mode,
)
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import get_logger, log_if_rank_zero

logger = get_logger("RetrievalEvalLightningModule")


class RetrievalEvalLightningModule(L.LightningModule):
    """LightningModule for index-based retrieval evaluation."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.automatic_optimization: bool = False
        self.cfg: DictConfig = cfg
        self.save_hyperparameters(cfg)

        self.model: SpladeModel = self._load_model()
        self._torch_compile_mark_step: Callable[[], None] | None = None
        self._setup_torch_compile()

        self._retrieval_helper = IndexedRetrievalHelper(
            cfg=cfg, logger=logger, index_context="evaluation"
        )
        self.metric_collection: RetrievalMetrics = RetrievalMetrics(
            dataset_name=self.cfg.dataset.name,
            k_list=self._retrieval_helper.k_list,
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

    def _setup_torch_compile(self) -> dict[str, Any]:
        compile_enabled: bool = bool(self.cfg.testing.get("torch_compile", False))
        compile_available: bool = hasattr(torch, "compile")
        self._torch_compile_mark_step = None
        if compile_enabled and not compile_available:
            log_if_rank_zero(
                logger,
                "torch.compile is not available in this PyTorch build; continuing "
                "without compilation.",
                level="warning",
            )
            return {}
        if not compile_enabled or not compile_available:
            return {}
        compile_mode_value: Any = self.cfg.testing.get("torch_compile_mode", "default")
        compile_mode, compile_mode_kwargs = validate_torch_compile_mode(
            compile_mode_value
        )
        if compile_mode in {"reduce-overhead", "max-autotune"}:
            self._torch_compile_mark_step = resolve_cudagraph_mark_step()
        query_wrapper: torch.nn.Module = self.model._query_encoder_wrapper
        query_encoder = torch.compile(query_wrapper, **compile_mode_kwargs)
        self.model._query_encoder_fn = query_encoder
        return compile_mode_kwargs

    # --- Public methods ---
    def on_test_start(self) -> None:
        self._local_query_offset = 0
        self.metric_collection.reset()
        self.metric_collection.to(torch.device("cpu"))
        self.model.eval()
        self._retrieval_helper.setup()

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        _ = batch_idx

        relevance_judgments_list: List[Dict[str, int]] = batch["relevance_judgments"]
        query_input_ids: torch.Tensor = batch["query_input_ids"].to(self.device)
        query_attention_mask: torch.Tensor = batch["query_attention_mask"].to(
            self.device
        )
        query_reps: torch.Tensor = self._retrieval_helper.encode_queries(
            self.model,
            query_input_ids,
            query_attention_mask,
            self._torch_compile_mark_step,
        )
        scored_results = self._retrieval_helper.score_queries(query_reps)

        world_size: int = int(self.trainer.world_size)
        global_rank: int = int(self.trainer.global_rank)
        base_offset: int = self._local_query_offset
        # Track per-rank progress to keep unique global indexes across batches.
        batch_size: int = int(query_input_ids.shape[0])
        self._local_query_offset += batch_size

        for i, relevance_judgments in enumerate(relevance_judgments_list):
            selected_doc_ids, selected_scores = scored_results[i]

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
        finalize_retrieval_metrics(
            metric_collection=self.metric_collection, module=self, logger=logger
        )

    def on_test_end(self) -> None:
        self._retrieval_helper.shutdown()
