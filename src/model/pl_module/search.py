import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig

from src.indexing.sparse_index import (
    InvertedIndex,
    load_inverted_index,
    score_query_postings,
    sparsify_query_vector,
)
from src.metric.retrieval import resolve_k_list
from src.model.pl_module.utils import build_splade_model_with_checkpoint
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import log_if_rank_zero
from src.utils.model_utils import resolve_tagged_output_dir

logger: logging.Logger = logging.getLogger("RetrievalSearchLightningModule")


def _resolve_cudagraph_mark_step() -> Callable[[], None] | None:
    if not hasattr(torch, "compiler"):
        return None
    compiler_mod = torch.compiler
    if not hasattr(compiler_mod, "cudagraph_mark_step_begin"):
        return None
    mark_step_fn = compiler_mod.cudagraph_mark_step_begin
    return mark_step_fn if callable(mark_step_fn) else None


def _build_compile_kwargs(mode: str) -> dict[str, Any]:
    return {"mode": mode}


def _append_rank_suffix(path: Path, rank: int) -> Path:
    suffix: str = path.suffix
    stem: str = path.stem
    if suffix:
        return path.with_name(f"{stem}.rank{rank}{suffix}")
    return path.with_name(f"{path.name}.rank{rank}")


class RetrievalSearchLightningModule(L.LightningModule):
    """LightningModule for index-based retrieval search output."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.automatic_optimization: bool = False
        self.cfg: DictConfig = cfg
        self.save_hyperparameters(cfg)

        self.model: SpladeModel = self._load_model()
        self._torch_compile_mark_step: Callable[[], None] | None = None
        self._setup_torch_compile()

        self._k_list: List[int] = resolve_k_list(self.cfg.testing.k_list)
        self._k_max: int = max(self._k_list)

        self._doc_ids: List[str] | None = None
        self._index: InvertedIndex | None = None
        self._score_buffer: np.ndarray | None = None
        self._seen_buffer: np.ndarray | None = None
        self._query_exclude_token_ids: list[int] = []
        self._query_min_weight: float = 0.0
        self._query_top_k: int | None = None

        search_cfg: DictConfig | None = getattr(self.cfg, "search", None)
        self._exclude_positives: bool = bool(
            search_cfg.exclude_positives
            if search_cfg is not None and "exclude_positives" in search_cfg
            else False
        )
        self._include_query_text: bool = bool(
            search_cfg.include_query_text
            if search_cfg is not None and "include_query_text" in search_cfg
            else False
        )
        self._flush_every: int = int(
            search_cfg.flush_every
            if search_cfg is not None and "flush_every" in search_cfg
            else 100
        )

        self._output_handle: Any | None = None
        self._queries_written: int = 0

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
        compile_mode: str = str(compile_mode_value).lower()
        valid_compile_modes: set[str] = {
            "default",
            "reduce-overhead",
            "max-autotune",
        }
        if compile_mode not in valid_compile_modes:
            raise ValueError(
                "Unsupported torch.compile mode: "
                f"{compile_mode_value!r}. Expected one of "
                f"{sorted(valid_compile_modes)}."
            )
        compile_mode_kwargs: dict[str, Any] = _build_compile_kwargs(compile_mode)
        if compile_mode in {"reduce-overhead", "max-autotune"}:
            self._torch_compile_mark_step = _resolve_cudagraph_mark_step()
        query_wrapper: torch.nn.Module = self.model._query_encoder_wrapper
        query_encoder = torch.compile(query_wrapper, **compile_mode_kwargs)
        self.model._query_encoder_fn = query_encoder
        return compile_mode_kwargs

    def _load_index(self) -> InvertedIndex:
        index_dir_value: str | None = self.cfg.encoding.index_dir
        if not index_dir_value:
            raise ValueError("encoding.index_dir must be set for index-based search.")
        index_path: Path = resolve_tagged_output_dir(
            index_dir_value,
            model_name=str(self.cfg.model.name),
            tag=self.cfg.tag,
        )
        index: InvertedIndex = load_inverted_index(index_path)
        return index

    def _resolve_query_sparsify_config(self, metadata: dict[str, Any]) -> None:
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
        self._score_buffer = np.zeros(int(doc_count), dtype=np.float32)
        self._seen_buffer = np.zeros(int(doc_count), dtype=np.uint8)

    def _open_output_handle(self) -> None:
        if not bool(self.cfg.testing.save_run):
            raise ValueError("testing.save_run must be true to save search results.")
        run_path_value: str | None = self.cfg.testing.run_path
        if not run_path_value:
            raise ValueError("testing.run_path must be set to save search results.")
        run_path = Path(str(run_path_value))
        if int(self.trainer.world_size) > 1:
            run_path = _append_rank_suffix(run_path, int(self.trainer.global_rank))
        run_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_handle = run_path.open("w", encoding="utf-8")
        log_if_rank_zero(logger, f"Writing search results to {run_path}")

    def _close_output_handle(self) -> None:
        if self._output_handle is None:
            return
        self._output_handle.close()
        self._output_handle = None

    # --- Public methods ---
    def on_test_start(self) -> None:
        self.model.eval()
        self._queries_written = 0

        index: InvertedIndex = self._load_index()
        self._index = index
        self._doc_ids = list(index.doc_ids)
        self._resolve_query_sparsify_config(index.metadata)
        self._prepare_score_buffers(len(index.doc_ids))
        self._open_output_handle()

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        _ = batch_idx
        if (
            self._index is None
            or self._doc_ids is None
            or self._score_buffer is None
            or self._seen_buffer is None
            or self._output_handle is None
        ):
            raise ValueError(
                "Index, buffers, and output handle must be initialized in on_test_start."
            )

        qids: List[str] = batch["qid"]
        query_texts: List[str] | None = (
            batch.get("query_text") if self._include_query_text else None
        )
        relevance_judgments_list: List[Dict[str, float]] = batch[
            "relevance_judgments"
        ]
        query_input_ids: torch.Tensor = batch["query_input_ids"].to(self.device)
        query_attention_mask: torch.Tensor = batch["query_attention_mask"].to(
            self.device
        )
        if self._torch_compile_mark_step is not None:
            self._torch_compile_mark_step()
        query_reps: torch.Tensor = self.model.encode_queries(
            query_input_ids, query_attention_mask
        )
        query_reps_cpu: np.ndarray = query_reps.detach().cpu().float().numpy()

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

            if self._exclude_positives and relevance_judgments:
                positive_ids: set[str] = {
                    doc_id
                    for doc_id, score in relevance_judgments.items()
                    if float(score) > 0
                }
                if positive_ids:
                    filtered_doc_ids: list[str] = []
                    filtered_scores: list[float] = []
                    for doc_id, score in zip(selected_doc_ids, selected_scores):
                        if doc_id in positive_ids:
                            continue
                        filtered_doc_ids.append(doc_id)
                        filtered_scores.append(score)
                    selected_doc_ids = filtered_doc_ids
                    selected_scores = filtered_scores

            record: dict[str, Any] = {
                "qid": qids[i],
                "doc_ids": selected_doc_ids,
                "scores": selected_scores,
            }
            if query_texts is not None:
                record["query_text"] = query_texts[i]
            self._output_handle.write(json.dumps(record) + "\n")
            self._queries_written += 1
            if self._flush_every > 0 and self._queries_written % self._flush_every == 0:
                self._output_handle.flush()

    def on_test_end(self) -> None:
        self._close_output_handle()
