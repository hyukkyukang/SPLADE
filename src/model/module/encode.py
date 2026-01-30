from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import lightning as L
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.indexing.sparse_index import SparseShardWriter
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils import log_if_rank_zero
from src.utils.model_utils import build_splade_model, load_splade_checkpoint
from src.utils.transformers import build_tokenizer

logger: logging.Logger = logging.getLogger("SPLADEEncodeModule")


class SPLADEEncodeModule(L.LightningModule):
    """LightningModule for encoding SPLADE document vectors."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.model: SpladeModel = self._load_model()
        self._tokenizer: PreTrainedTokenizerBase = build_tokenizer(
            self.cfg.model.huggingface_name
        )
        self._writer: SparseShardWriter | None = None

    # --- Protected methods ---
    def _load_model(self) -> SpladeModel:
        model: SpladeModel = build_splade_model(
            self.cfg, use_cpu=bool(self.cfg.encoding.use_cpu)
        )
        checkpoint_path: str | None = getattr(
            self.cfg.encoding, "checkpoint_path", None
        )
        if checkpoint_path:
            missing: list[str]
            unexpected: list[str]
            missing, unexpected = load_splade_checkpoint(model, checkpoint_path)
            log_if_rank_zero(
                logger,
                f"Loaded checkpoint. Missing: {len(missing)}, unexpected: {len(unexpected)}",
            )
        return model

    def _resolve_exclude_token_ids(self) -> list[int]:
        configured_ids: Sequence[int] | None = getattr(
            self.cfg.model, "exclude_token_ids", None
        )
        if configured_ids is not None:
            return [int(token_id) for token_id in configured_ids]
        return [int(token_id) for token_id in self._tokenizer.all_special_ids]

    # --- Public methods ---
    def on_predict_start(self) -> None:
        encode_path_value: str | None = getattr(self.cfg.model, "encode_path", None)
        encode_path: Path = Path(encode_path_value or "encode")
        vocab_size: int = int(self.model.encoder.mlm.config.vocab_size)
        self._writer = SparseShardWriter(
            output_dir=encode_path,
            vocab_size=vocab_size,
            rank=int(self.trainer.global_rank),
            top_k=getattr(self.cfg.model, "sparse_top_k", None),
            min_weight=float(getattr(self.cfg.model, "sparse_min_weight", 0.0)),
            exclude_token_ids=self._resolve_exclude_token_ids(),
            shard_max_docs=int(self.cfg.encoding.shard_max_docs),
            value_dtype=str(self.cfg.encoding.value_dtype),
        )
        self.model.eval()

    def on_predict_end(self) -> None:
        if self._writer is None:
            return
        self._writer.finalize()

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _ = batch_idx
        if self._writer is None:
            raise RuntimeError("Writer is not initialized.")
        doc_texts: list[str] = list(batch["doc_texts"])
        doc_ids: list[str] = list(batch["doc_ids"])
        tokens: dict[str, torch.Tensor] = self._tokenizer(
            doc_texts,
            padding=True,
            truncation=True,
            max_length=int(self.cfg.dataset.max_doc_length),
            return_tensors="pt",
        )
        doc_input_ids: torch.Tensor = tokens["input_ids"].to(self.device)
        doc_attention_mask: torch.Tensor = tokens["attention_mask"].to(self.device)
        doc_reps: torch.Tensor = self.model.encode_docs(
            doc_input_ids, doc_attention_mask
        )
        self._writer.write_batch(doc_ids, doc_reps)
