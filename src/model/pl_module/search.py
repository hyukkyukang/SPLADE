import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

import lightning as L
import torch
from omegaconf import DictConfig

from src.search.retrieval import IndexedRetrievalHelper
from src.model.pl_module.utils import (
    build_splade_model_with_checkpoint,
    resolve_cudagraph_mark_step,
    validate_torch_compile_mode,
)
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import log_if_rank_zero

logger: logging.Logger = logging.getLogger("RetrievalSearchLightningModule")


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

        self._retrieval_helper = IndexedRetrievalHelper(
            cfg=cfg, logger=logger, index_context="search"
        )

        search_cfg: DictConfig = self.cfg.search
        self._flush_every = int(search_cfg.flush_every)

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
        compile_enabled: bool = bool(self.cfg.testing.torch_compile)
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
        compile_mode_value: Any = self.cfg.testing.torch_compile_mode
        compile_mode, compile_mode_kwargs = validate_torch_compile_mode(
            compile_mode_value
        )
        if compile_mode in {"reduce-overhead", "max-autotune"}:
            self._torch_compile_mark_step = resolve_cudagraph_mark_step()
        query_wrapper: torch.nn.Module = self.model._query_encoder_wrapper
        query_encoder = torch.compile(query_wrapper, **compile_mode_kwargs)
        self.model._query_encoder_fn = query_encoder
        return compile_mode_kwargs

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
        self._retrieval_helper.setup()
        self._open_output_handle()

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        _ = batch_idx
        if self._output_handle is None:
            raise ValueError(
                "Output handle must be initialized in on_test_start."
            )

        qids: List[str] = batch["qid"]
        relevance_judgments_list: List[Dict[str, float]] = batch[
            "relevance_judgments"
        ]
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
        for i, relevance_judgments in enumerate(relevance_judgments_list):
            selected_doc_ids, _ = scored_results[i]

            pos_doc_ids: list[str] = [
                doc_id
                for doc_id, score in relevance_judgments.items()
                if float(score) > 0
            ]
            pos_doc_ids = sorted(pos_doc_ids)
            # Keep a safety net for missing positives even if dataset filtering is on.
            if not pos_doc_ids:
                log_if_rank_zero(
                    logger,
                    f"Skipping {qids[i]}: missing positives for hard negatives.",
                    level="warning",
                )
                continue
            pos_id_set: set[str] = set(pos_doc_ids)
            if pos_id_set:
                neg_doc_ids: list[str] = [
                    doc_id for doc_id in selected_doc_ids if doc_id not in pos_id_set
                ]
            else:
                neg_doc_ids = list(selected_doc_ids)

            record: dict[str, Any] = {
                "qid": qids[i],
                "pos_doc_ids": pos_doc_ids,
                "neg_doc_ids": neg_doc_ids,
            }
            self._output_handle.write(json.dumps(record) + "\n")
            self._queries_written += 1
            if self._flush_every > 0 and self._queries_written % self._flush_every == 0:
                self._output_handle.flush()

    def on_test_end(self) -> None:
        self._retrieval_helper.shutdown()
        self._close_output_handle()
