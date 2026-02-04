import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from src.indexing.async_writer import AsyncSparseWriter, SparseWriterConfig
from src.indexing.sparse_index import (
    SparseShardWriter,
    resolve_numpy_dtype,
    sparsify_batch_gpu_csr,
)
from src.model.pl_module.utils import build_splade_model_with_checkpoint
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils import is_rank_zero, log_if_rank_zero
from src.utils.model_utils import resolve_tagged_output_dir
from src.utils.transformers import build_tokenizer

logger: logging.Logger = logging.getLogger("SPLADEEncodeModule")


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
        self._async_writer: AsyncSparseWriter | None = None
        self._exclude_token_ids_tensor: torch.Tensor | None = None
        self._value_dtype = resolve_numpy_dtype(str(self.cfg.encoding.value_dtype))
        self._min_weight: float = float(self.cfg.encoding.sparse_min_weight)
        self._top_k: int | None = self.cfg.encoding.sparse_top_k
        self._async_write_enabled: bool = bool(
            self.cfg.encoding.get("async_write", False)
        )
        self._async_write_queue_size: int = int(
            self.cfg.encoding.get("async_write_queue_size", 8)
        )
        self._torch_compile_mark_step: Callable[[], None] | None = None
        self._setup_torch_compile()

    # --- Protected methods ---
    def _load_model(self) -> SpladeModel:
        checkpoint_path: str | None = self.cfg.encoding.checkpoint_path
        return build_splade_model_with_checkpoint(
            cfg=self.cfg,
            use_cpu=bool(self.cfg.encoding.use_cpu),
            checkpoint_path=checkpoint_path,
            logger=logger,
        )

    def _resolve_exclude_token_ids(self) -> list[int]:
        configured_ids: Sequence[int] | None = self.cfg.model.exclude_token_ids
        if configured_ids is not None:
            return [int(token_id) for token_id in configured_ids]
        return [int(token_id) for token_id in self._tokenizer.all_special_ids]

    def _setup_torch_compile(self) -> dict[str, Any]:
        compile_enabled: bool = bool(self.cfg.encoding.get("torch_compile", False))
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
        compile_mode_value: Any = self.cfg.encoding.get("torch_compile_mode", "default")
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
        doc_wrapper: torch.nn.Module = self.model._doc_encoder_wrapper
        doc_encoder = torch.compile(doc_wrapper, **compile_mode_kwargs)
        self.model._doc_encoder_fn = doc_encoder
        return compile_mode_kwargs

    # --- Public methods ---
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
            config_text: str = OmegaConf.to_yaml(self.cfg, resolve=True)
            config_path.write_text(config_text, encoding="utf-8")
            log_if_rank_zero(logger, f"Saved encoding config to {config_path}.")
        vocab_size: int = int(self.model.encoder.mlm.config.vocab_size)
        exclude_token_ids: list[int] = self._resolve_exclude_token_ids()
        writer_cfg = SparseWriterConfig(
            output_dir=encode_path,
            vocab_size=vocab_size,
            rank=int(self.trainer.global_rank),
            top_k=self.cfg.encoding.sparse_top_k,
            min_weight=float(self.cfg.encoding.sparse_min_weight),
            exclude_token_ids=exclude_token_ids,
            shard_max_docs=int(self.cfg.encoding.shard_max_docs),
            value_dtype=str(self.cfg.encoding.value_dtype),
        )
        if self._async_write_enabled:
            self._async_writer = AsyncSparseWriter(
                writer_cfg,
                queue_size=self._async_write_queue_size,
                log=logger,
            )
            self._async_writer.start()
            self._writer = None
        else:
            self._writer = SparseShardWriter(
                output_dir=writer_cfg.output_dir,
                vocab_size=writer_cfg.vocab_size,
                rank=writer_cfg.rank,
                top_k=writer_cfg.top_k,
                min_weight=writer_cfg.min_weight,
                exclude_token_ids=writer_cfg.exclude_token_ids,
                shard_max_docs=writer_cfg.shard_max_docs,
                value_dtype=writer_cfg.value_dtype,
            )
            self._async_writer = None
        if exclude_token_ids:
            self._exclude_token_ids_tensor = torch.tensor(
                exclude_token_ids, dtype=torch.long, device=self.device
            )
        else:
            self._exclude_token_ids_tensor = None
        self._min_weight = float(self.cfg.encoding.sparse_min_weight)
        self._top_k = self.cfg.encoding.sparse_top_k
        self.model.eval()

    def on_predict_end(self) -> None:
        if self._async_writer is not None:
            self._async_writer.close()
            self._async_writer = None
        if self._writer is not None:
            self._writer.finalize()
            self._writer = None

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _ = batch_idx
        if self._async_write_enabled and self._async_writer is None:
            raise RuntimeError("Async writer is not initialized.")
        if not self._async_write_enabled and self._writer is None:
            raise RuntimeError("Writer is not initialized.")
        doc_ids: list[str] = list(batch["doc_ids"])
        doc_input_ids: torch.Tensor = batch["doc_input_ids"].to(
            self.device, non_blocking=True
        )
        doc_attention_mask: torch.Tensor = batch["doc_attention_mask"].to(
            self.device, non_blocking=True
        )
        if self._torch_compile_mark_step is not None:
            self._torch_compile_mark_step()
        doc_reps: torch.Tensor = self.model.encode_docs(
            doc_input_ids, doc_attention_mask
        )
        indptr, indices, values = sparsify_batch_gpu_csr(
            doc_reps,
            exclude_token_ids=self._exclude_token_ids_tensor,
            min_weight=self._min_weight,
            top_k=self._top_k,
            value_dtype=self._value_dtype,
        )
        if self._async_writer is not None:
            self._async_writer.check_healthy()
            indptr.share_memory_()
            indices.share_memory_()
            values.share_memory_()
            self._async_writer.write(doc_ids, indptr, indices, values)
        elif self._writer is not None:
            self._writer.write_sparse_csr_batch(doc_ids, indptr, indices, values)
