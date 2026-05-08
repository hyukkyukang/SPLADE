from typing import Any, Callable, Dict, List, Mapping, Sequence

import lightning as L
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from src.metric.retrieval import RetrievalMetrics
from src.model.pl_module.utils import (
    build_retrieval_model_with_checkpoint,
    finalize_retrieval_metrics,
    resolve_cudagraph_mark_step,
    validate_torch_compile_mode,
)
from src.model.retriever.dense.neural.hf_dense import DenseRetrievalModel
from src.search.dense_retrieval import DenseRetrievalHelper
from src.utils.logging import get_logger, log_if_rank_zero

logger = get_logger("DenseRetrievalEvalLightningModule")


def _empty_query_representations(embedding_dim: int) -> torch.Tensor:
    return torch.empty((0, int(embedding_dim)), dtype=torch.float32)


def _normalize_relevance_judgments(
    relevance_judgments: Mapping[str, Any],
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    doc_id: str
    relevance: Any
    for doc_id, relevance in relevance_judgments.items():
        normalized[str(doc_id)] = int(relevance)
    return normalized


def _merge_gathered_dense_query_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    embedding_dim: int,
) -> tuple[torch.Tensor, list[str], list[dict[str, int]]]:
    merged_query_reps: list[torch.Tensor] = []
    merged_query_ids: list[str] = []
    merged_relevance_judgments: list[dict[str, int]] = []

    payload: Mapping[str, Any]
    for payload in payloads:
        query_reps: Any = payload.get("query_reps")
        if not isinstance(query_reps, torch.Tensor):
            raise TypeError("Gathered dense query payload must include a torch.Tensor.")
        if query_reps.ndim != 2:
            raise ValueError("Gathered dense query representations must be rank-2 tensors.")
        if int(query_reps.shape[0]) > 0 and int(query_reps.shape[1]) != int(embedding_dim):
            raise ValueError("Gathered dense query representation dimension mismatch.")

        query_ids: list[str] = [str(query_id) for query_id in payload.get("query_ids", [])]
        raw_relevance_judgments: Sequence[Mapping[str, Any]] = payload.get(
            "relevance_judgments", []
        )
        if int(query_reps.shape[0]) != len(query_ids):
            raise ValueError(
                "Gathered dense query ids length does not match representation rows."
            )
        if len(query_ids) != len(raw_relevance_judgments):
            raise ValueError(
                "Gathered dense relevance judgments length does not match query ids."
            )

        if int(query_reps.shape[0]) > 0:
            merged_query_reps.append(query_reps.detach().float().cpu())
        merged_query_ids.extend(query_ids)
        merged_relevance_judgments.extend(
            _normalize_relevance_judgments(relevance_judgments)
            for relevance_judgments in raw_relevance_judgments
        )

    if not merged_query_reps:
        return (
            _empty_query_representations(embedding_dim),
            merged_query_ids,
            merged_relevance_judgments,
        )
    return (
        torch.cat(merged_query_reps, dim=0),
        merged_query_ids,
        merged_relevance_judgments,
    )


class DenseRetrievalEvalLightningModule(L.LightningModule):
    """LightningModule for FAISS-based dense retrieval evaluation."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.automatic_optimization: bool = False
        self.cfg: DictConfig = cfg
        self.save_hyperparameters(cfg)

        self.model: DenseRetrievalModel = self._load_model()
        self._torch_compile_mark_step: Callable[[], None] | None = None
        self._setup_torch_compile()

        self._retrieval_helper = DenseRetrievalHelper(
            cfg=cfg, logger=logger, index_context="evaluation"
        )
        self.metric_collection: RetrievalMetrics = RetrievalMetrics(
            dataset_name=self.cfg.dataset.name,
            k_list=self._retrieval_helper.k_list,
            metric_families=self.cfg.testing.get("metric_families"),
            sync_on_compute=False,
        )
        self._local_query_offset: int = 0
        self._pending_query_reps: list[torch.Tensor] = []
        self._pending_query_ids: list[str] = []
        self._pending_relevance_judgments: list[dict[str, int]] = []

    def _load_model(self) -> DenseRetrievalModel:
        checkpoint_path: str | None = self.cfg.testing.checkpoint_path
        model = build_retrieval_model_with_checkpoint(
            cfg=self.cfg,
            use_cpu=bool(self.cfg.testing.use_cpu),
            checkpoint_path=checkpoint_path,
            logger=logger,
        )
        if not isinstance(model, DenseRetrievalModel):
            raise TypeError("DenseRetrievalEvalLightningModule requires a dense model.")
        return model

    def _setup_torch_compile(self) -> dict[str, Any]:
        compile_enabled: bool = bool(self.cfg.testing.get("torch_compile", False))
        compile_available: bool = hasattr(torch, "compile")
        self._torch_compile_mark_step = None
        if compile_enabled and not compile_available:
            log_if_rank_zero(
                logger,
                "torch.compile is not available in this PyTorch build; continuing without compilation.",
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
        self.model._query_encoder_fn = torch.compile(
            self.model._query_encoder_wrapper,
            **compile_mode_kwargs,
        )
        return compile_mode_kwargs

    def on_test_start(self) -> None:
        self._local_query_offset = 0
        self._pending_query_reps = []
        self._pending_query_ids = []
        self._pending_relevance_judgments = []
        self.metric_collection.reset()
        self.metric_collection.to(torch.device("cpu"))
        self.model.eval()
        if not self._uses_rank_zero_search() or self.trainer.is_global_zero:
            device_index: int | None = None
            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                device_index = 0 if self.device.index is None else int(self.device.index)
            self._retrieval_helper.setup(device_index=device_index)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        _ = batch_idx
        qids: List[str] = batch["qid"]
        relevance_judgments_list: List[Dict[str, int]] = batch["relevance_judgments"]
        query_input_ids: torch.Tensor = batch["query_input_ids"].to(self.device)
        query_attention_mask: torch.Tensor = batch["query_attention_mask"].to(
            self.device
        )
        query_pooling_mask: torch.Tensor | None = batch.get("query_pooling_mask")
        if query_pooling_mask is not None:
            query_pooling_mask = query_pooling_mask.to(self.device)
        query_indptr: torch.Tensor = batch["query_indptr"]
        query_reps: torch.Tensor = self._retrieval_helper.encode_queries(
            self.model,
            query_input_ids,
            query_attention_mask,
            self._torch_compile_mark_step,
            query_pooling_mask=query_pooling_mask,
            query_indptr=query_indptr,
        )
        if self._uses_rank_zero_search():
            self._accumulate_query_batch(
                query_reps=query_reps,
                qids=qids,
                relevance_judgments_list=relevance_judgments_list,
            )
            return
        scored_results = self._retrieval_helper.score_queries(
            query_reps, query_ids=qids
        )

        world_size: int = int(self.trainer.world_size)
        global_rank: int = int(self.trainer.global_rank)
        base_offset: int = self._local_query_offset
        batch_size: int = len(qids)
        self._local_query_offset += batch_size

        self._append_metric_rows(
            scored_results=scored_results,
            relevance_judgments_list=relevance_judgments_list,
            query_indexes=[
                global_rank + world_size * (base_offset + i)
                for i in range(len(relevance_judgments_list))
            ],
        )

    def on_test_epoch_end(self) -> None:
        if self._uses_rank_zero_search():
            self._score_gathered_queries_on_rank_zero()
        finalize_retrieval_metrics(
            metric_collection=self.metric_collection, module=self, logger=logger
        )

    def on_test_end(self) -> None:
        self._retrieval_helper.shutdown()

    def _uses_rank_zero_search(self) -> bool:
        return int(self.trainer.world_size) > 1

    def _accumulate_query_batch(
        self,
        *,
        query_reps: torch.Tensor,
        qids: Sequence[str],
        relevance_judgments_list: Sequence[Mapping[str, Any]],
    ) -> None:
        query_reps_cpu: torch.Tensor = query_reps.detach().float().cpu()
        if int(query_reps_cpu.shape[0]) != len(qids):
            raise ValueError("Dense query batch ids length does not match embeddings.")
        if len(qids) != len(relevance_judgments_list):
            raise ValueError(
                "Dense relevance judgments length does not match query embeddings."
            )
        self._pending_query_reps.append(query_reps_cpu)
        self._pending_query_ids.extend(str(query_id) for query_id in qids)
        self._pending_relevance_judgments.extend(
            _normalize_relevance_judgments(relevance_judgments)
            for relevance_judgments in relevance_judgments_list
        )

    def _local_query_payload(self) -> dict[str, Any]:
        embedding_dim: int = int(self.model.embedding_dim)
        query_reps: torch.Tensor
        if self._pending_query_reps:
            query_reps = torch.cat(self._pending_query_reps, dim=0)
        else:
            query_reps = _empty_query_representations(embedding_dim)
        return {
            "query_reps": query_reps,
            "query_ids": list(self._pending_query_ids),
            "relevance_judgments": list(self._pending_relevance_judgments),
        }

    def _score_gathered_queries_on_rank_zero(self) -> None:
        payload: dict[str, Any] = self._local_query_payload()
        world_size: int = int(self.trainer.world_size)
        if world_size > 1:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "Distributed dense evaluation requires an initialized process group."
                )
            gathered_payloads: list[dict[str, Any] | None] = [None] * world_size
            dist.all_gather_object(gathered_payloads, payload)
        else:
            gathered_payloads = [payload]

        self._pending_query_reps = []
        self._pending_query_ids = []
        self._pending_relevance_judgments = []

        if not self.trainer.is_global_zero:
            return

        query_reps, query_ids, relevance_judgments_list = (
            _merge_gathered_dense_query_payloads(
                [gathered for gathered in gathered_payloads if gathered is not None],
                embedding_dim=int(self.model.embedding_dim),
            )
        )
        if len(query_ids) == 0:
            return
        scored_results = self._retrieval_helper.score_queries(
            query_reps,
            query_ids=query_ids,
        )
        self._append_metric_rows(
            scored_results=scored_results,
            relevance_judgments_list=relevance_judgments_list,
            query_indexes=list(range(len(query_ids))),
        )

    def _append_metric_rows(
        self,
        *,
        scored_results: Sequence[tuple[list[str], list[float]]],
        relevance_judgments_list: Sequence[Mapping[str, Any]],
        query_indexes: Sequence[int],
    ) -> None:
        if len(scored_results) != len(relevance_judgments_list):
            raise ValueError(
                "Dense scored results length does not match relevance judgments length."
            )
        if len(query_indexes) != len(relevance_judgments_list):
            raise ValueError("Dense query index length does not match result rows.")

        query_index: int
        relevance_judgments: Mapping[str, Any]
        selected_doc_ids: list[str]
        selected_scores: list[float]
        for query_index, relevance_judgments, (
            selected_doc_ids,
            selected_scores,
        ) in zip(query_indexes, relevance_judgments_list, scored_results):
            labels: List[float] = []
            final_scores: List[float] = []
            for doc_id, score in zip(selected_doc_ids, selected_scores):
                relevance: float = float(relevance_judgments.get(str(doc_id), 0))
                labels.append(relevance)
                final_scores.append(float(score))

            min_score: float = min(final_scores) if final_scores else 0.0
            doc_id: str
            relevance: Any
            for doc_id, relevance in relevance_judgments.items():
                if float(relevance) > 0 and doc_id not in selected_doc_ids:
                    labels.append(float(relevance))
                    final_scores.append(min_score - 1.0)

            if not final_scores:
                continue

            score_tensor: torch.Tensor = torch.tensor(
                final_scores, dtype=torch.float32, device=torch.device("cpu")
            )
            label_tensor: torch.Tensor = torch.tensor(
                labels, dtype=torch.float32, device=torch.device("cpu")
            )
            indexes: torch.Tensor = torch.full(
                (len(final_scores),),
                int(query_index),
                dtype=torch.long,
                device=torch.device("cpu"),
            )
            self.metric_collection.append(score_tensor, label_tensor, indexes)
