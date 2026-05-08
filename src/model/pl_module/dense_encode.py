from pathlib import Path
from typing import Any, Callable, Sequence

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.pl_module.common import build_model_tokenizer
from src.index.dense import DenseShardWriter
from src.model.pl_module.utils import (
    build_retrieval_model_with_checkpoint,
    resolve_cudagraph_mark_step,
    validate_torch_compile_mode,
)
from src.model.retriever.dense.neural.hf_dense import DenseRetrievalModel
from src.utils import is_rank_zero, log_if_rank_zero, maybe_barrier
from src.utils.logging import get_logger
from src.utils.model_utils import resolve_tagged_output_dir
from src.utils.windowed_encoding import encode_and_aggregate_windows

logger = get_logger("DenseEncodeModule")


class DenseEncodeModule(L.LightningModule):
    """LightningModule for encoding dense document vectors."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.model: DenseRetrievalModel = self._load_model()
        self._tokenizer = build_model_tokenizer(self.cfg.model)
        pad_token_id: int | None = self._tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define a pad token id for encoding.")
        self._pad_token_id: int = int(pad_token_id)
        self._writer: DenseShardWriter | None = None
        self._max_windows_per_forward: int | None = (
            None
            if self.cfg.encoding.get("max_windows_per_forward") is None
            else max(1, int(self.cfg.encoding.max_windows_per_forward))
        )
        self._use_fixed_window_chunks: bool = False
        self._torch_compile_mark_step: Callable[[], None] | None = None
        self._setup_torch_compile()

    def _load_model(self) -> DenseRetrievalModel:
        checkpoint_path: str | None = self.cfg.encoding.checkpoint_path
        model = build_retrieval_model_with_checkpoint(
            cfg=self.cfg,
            use_cpu=bool(self.cfg.encoding.use_cpu),
            checkpoint_path=checkpoint_path,
            logger=logger,
        )
        if not isinstance(model, DenseRetrievalModel):
            raise TypeError("DenseEncodeModule requires cfg.model.family=dense.")
        return model

    def _setup_torch_compile(self) -> dict[str, Any]:
        compile_enabled: bool = bool(self.cfg.encoding.get("torch_compile", False))
        compile_available: bool = hasattr(torch, "compile")
        self._use_fixed_window_chunks = False
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
        if self._max_windows_per_forward is not None and self._max_windows_per_forward > 0:
            self._use_fixed_window_chunks = True
        compile_mode_value: Any = self.cfg.encoding.get("torch_compile_mode", "default")
        compile_mode, compile_mode_kwargs = validate_torch_compile_mode(
            compile_mode_value
        )
        if compile_mode in {"reduce-overhead", "max-autotune"}:
            self._torch_compile_mark_step = resolve_cudagraph_mark_step()
        self.model._doc_encoder_fn = torch.compile(
            self.model._doc_encoder_wrapper,
            **compile_mode_kwargs,
        )
        return compile_mode_kwargs

    def _encode_and_aggregate_window_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None,
        doc_indptr: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        embeddings: torch.Tensor = encode_and_aggregate_windows(
            input_ids,
            attention_mask,
            pooling_mask,
            indptr=doc_indptr,
            encode_fn=lambda chunk_input_ids, chunk_attention_mask, chunk_pooling_mask: (
                self.model._doc_encoder_fn(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                    pooling_mask=chunk_pooling_mask,
                )
            ),
            pooling_mode=str(self.model.doc_window_pooling),
            output_dim=int(self.model.embedding_dim),
            output_dtype=next(self.model.parameters()).dtype,
            pad_token_id=self._pad_token_id,
            chunk_size=self._max_windows_per_forward,
            use_fixed_size_chunks=self._use_fixed_window_chunks,
            mark_step=self._torch_compile_mark_step,
            entity_name="document",
        )
        return self.model.postprocess_doc_embeddings(embeddings)

    def on_predict_start(self) -> None:
        encode_dir_value: str | None = self.cfg.encoding.encode_dir
        if encode_dir_value is None:
            raise ValueError("encoding.encode_dir must be set for encoding.")
        encode_path: Path = resolve_tagged_output_dir(
            encode_dir_value,
            model_name=str(self.cfg.model.name),
            tag=self.cfg.tag,
        )
        if is_rank_zero():
            encode_path.mkdir(parents=True, exist_ok=True)
            config_path: Path = encode_path / "config.yaml"
            config_path.write_text(OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8")
            log_if_rank_zero(logger, f"Saved encoding config to {config_path}.")
        self._writer = DenseShardWriter(
            output_dir=encode_path,
            dim=int(self.model.embedding_dim),
            rank=int(self.trainer.global_rank),
            model_family=str(self.model.family),
            similarity=str(self.model.similarity),
            normalized=bool(self.model.normalize),
            shard_max_docs=int(self.cfg.encoding.shard_max_docs),
            value_dtype=str(self.cfg.encoding.value_dtype),
        )
        self.model.eval()

    def on_predict_end(self) -> None:
        if self._writer is not None:
            self._writer.finalize()
            self._writer = None
        maybe_barrier()

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _ = batch_idx
        if self._writer is None:
            raise RuntimeError("Dense writer is not initialized.")
        doc_ids: list[str] = list(batch["doc_ids"])
        doc_group_ids: list[str | None] | None = batch.get("doc_group_ids")
        doc_input_ids: torch.Tensor = batch["doc_input_ids"].to(
            self.device, non_blocking=True
        )
        doc_attention_mask: torch.Tensor = batch["doc_attention_mask"].to(
            self.device, non_blocking=True
        )
        doc_pooling_mask: torch.Tensor | None = batch.get("doc_pooling_mask")
        if doc_pooling_mask is not None:
            doc_pooling_mask = doc_pooling_mask.to(self.device, non_blocking=True)
        doc_indptr: torch.Tensor = batch["doc_indptr"]
        doc_reps: torch.Tensor = self._encode_and_aggregate_window_batch(
            doc_input_ids,
            doc_attention_mask,
            doc_pooling_mask,
            doc_indptr,
        )
        self._writer.write_batch(doc_ids, doc_reps, doc_group_ids=doc_group_ids)
