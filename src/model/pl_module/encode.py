from pathlib import Path
from typing import Any, Callable, Sequence

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from src.index.async_writer import AsyncSparseWriter, SparseWriterConfig
from src.index.sparse import (
    SparseShardWriter,
    resolve_numpy_dtype,
    resolve_torch_dtype,
)
from src.search.sparsify import (
    _sparsify_batch_gpu_csr_core_threshold,
    _sparsify_batch_gpu_csr_core_topk,
    sparsify_batch_gpu_csr,
)
from src.model.pl_module.utils import (
    build_splade_model_with_checkpoint,
    resolve_cudagraph_mark_step,
    validate_torch_compile_mode,
)
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils import is_rank_zero, log_if_rank_zero, maybe_barrier
from src.utils.logging import get_logger
from src.utils.model_utils import resolve_tagged_output_dir
from src.utils.output_space import OutputSpaceSpec
from src.utils.transformers import build_tokenizer
from src.utils.windowed_encoding import encode_and_aggregate_windows

logger = get_logger("SPLADEEncodeModule")


class SPLADEEncodeModule(L.LightningModule):
    """LightningModule for encoding SPLADE document vectors."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.model: SpladeModel = self._load_model()
        self._tokenizer: PreTrainedTokenizerBase = build_tokenizer(
            str(self.cfg.model.huggingface_name),
            use_fast_tokenizer=bool(self.cfg.model.use_fast_tokenizer),
            trust_remote_code=bool(self.cfg.model.trust_remote_code),
            require_fast_tokenizer=bool(self.cfg.model.require_fast_tokenizer),
        )
        if bool(self.cfg.model.require_fast_tokenizer) and not bool(
            self._tokenizer.is_fast
        ):
            raise ValueError(
                "Fast tokenizer is required but a slow tokenizer was loaded: "
                f"{self.cfg.model.huggingface_name}"
            )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = (
                self._tokenizer.eos_token or self._tokenizer.cls_token
            )
        pad_token_id: int | None = self._tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define a pad token id for encoding.")
        self._pad_token_id: int = int(pad_token_id)
        self._writer: SparseShardWriter | None = None
        self._async_writer: AsyncSparseWriter | None = None
        self._exclude_output_ids_tensor: torch.Tensor | None = None
        self._value_dtype = resolve_numpy_dtype(str(self.cfg.encoding.value_dtype))
        self._min_weight: float = float(self.cfg.encoding.sparse_min_weight)
        self._top_k: int | None = self.cfg.encoding.sparse_top_k
        self._max_windows_per_forward: int | None = (
            None
            if self.cfg.encoding.get("max_windows_per_forward") is None
            else max(1, int(self.cfg.encoding.max_windows_per_forward))
        )
        self._async_write_enabled: bool = bool(
            self.cfg.encoding.get("async_write", False)
        )
        self._async_write_queue_size: int = int(
            self.cfg.encoding.get("async_write_queue_size", 8)
        )
        self._use_fixed_window_chunks: bool = False
        self._torch_compile_mark_step: Callable[[], None] | None = None
        self._sparsify_core_topk: (
            Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
        ) = None
        self._sparsify_core_threshold: (
            Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
        ) = None
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

    def _resolve_exclude_output_ids(
        self, raw_exclude_token_ids: Sequence[int]
    ) -> list[int]:
        if not raw_exclude_token_ids:
            return []
        output_space: OutputSpaceSpec = self.model.encoder.output_space
        resolved_output_ids: torch.Tensor = output_space.resolve_exclude_token_ids(
            list(raw_exclude_token_ids),
            token_id_to_output_index=self.model.encoder.token_id_to_output_index,
        )
        return [int(output_id) for output_id in resolved_output_ids.tolist()]

    def _setup_torch_compile(self) -> dict[str, Any]:
        compile_enabled: bool = bool(self.cfg.encoding.get("torch_compile", False))
        compile_available: bool = hasattr(torch, "compile")
        self._use_fixed_window_chunks = False
        self._torch_compile_mark_step = None
        self._sparsify_core_topk = None
        self._sparsify_core_threshold = None
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
        if self._max_windows_per_forward is None or self._max_windows_per_forward <= 0:
            log_if_rank_zero(
                logger,
                "torch.compile is enabled without encoding.max_windows_per_forward; "
                "window batch shapes will remain dynamic.",
                level="warning",
            )
        else:
            self._use_fixed_window_chunks = True
        compile_mode_value: Any = self.cfg.encoding.get("torch_compile_mode", "default")
        compile_mode, compile_mode_kwargs = validate_torch_compile_mode(
            compile_mode_value
        )
        if compile_mode in {"reduce-overhead", "max-autotune"}:
            self._torch_compile_mark_step = resolve_cudagraph_mark_step()
        doc_wrapper: torch.nn.Module = self.model._doc_encoder_wrapper
        doc_encoder = torch.compile(doc_wrapper, **compile_mode_kwargs)
        self.model._doc_encoder_fn = doc_encoder
        self._sparsify_core_topk = torch.compile(
            _sparsify_batch_gpu_csr_core_topk, **compile_mode_kwargs
        )
        self._sparsify_core_threshold = torch.compile(
            _sparsify_batch_gpu_csr_core_threshold, **compile_mode_kwargs
        )
        return compile_mode_kwargs

    def _sparsify_batch(
        self, vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._sparsify_core_topk is None and self._sparsify_core_threshold is None:
            return sparsify_batch_gpu_csr(
                vectors,
                exclude_output_ids=self._exclude_output_ids_tensor,
                min_weight=self._min_weight,
                top_k=self._top_k,
                value_dtype=self._value_dtype,
            )
        if vectors.ndim != 2:
            raise ValueError("sparsify_batch_gpu_csr expects a 2D tensor.")
        batch_size: int = int(vectors.shape[0])
        vocab_size: int = int(vectors.shape[1])
        if batch_size == 0:
            indptr = torch.zeros((1,), dtype=torch.int64, device="cpu")
            indices = torch.empty((0,), dtype=torch.int32, device="cpu")
            values = torch.empty(
                (0,), dtype=resolve_torch_dtype(self._value_dtype), device="cpu"
            )
            return indptr, indices, values
        threshold: float = float(self._min_weight) if self._min_weight > 0.0 else 0.0
        exclude_ids: torch.Tensor | None = self._exclude_output_ids_tensor
        if exclude_ids is not None and int(exclude_ids.numel()) > 0:
            exclude_ids = exclude_ids.to(device=vectors.device)
        if self._top_k is not None:
            top_k_int: int = min(int(self._top_k), vocab_size)
            if top_k_int <= 0:
                indptr = torch.zeros((batch_size + 1,), dtype=torch.int64, device="cpu")
                indices = torch.empty((0,), dtype=torch.int32, device="cpu")
                values = torch.empty(
                    (0,), dtype=resolve_torch_dtype(self._value_dtype), device="cpu"
                )
                return indptr, indices, values
            core_fn = self._sparsify_core_topk or _sparsify_batch_gpu_csr_core_topk
            indptr_gpu, flat_indices, flat_values = core_fn(
                vectors,
                exclude_output_ids=exclude_ids,
                threshold=threshold,
                top_k=top_k_int,
            )
        else:
            core_fn = (
                self._sparsify_core_threshold or _sparsify_batch_gpu_csr_core_threshold
            )
            indptr_gpu, flat_indices, flat_values = core_fn(
                vectors,
                exclude_output_ids=exclude_ids,
                threshold=threshold,
            )
        torch_value_dtype: torch.dtype = resolve_torch_dtype(self._value_dtype)
        indptr = indptr_gpu.to(device="cpu")
        indices = flat_indices.to(dtype=torch.int32, device="cpu")
        values = flat_values.to(dtype=torch_value_dtype, device="cpu")
        return indptr, indices, values

    def _encode_and_aggregate_window_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None,
        doc_indptr: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        return encode_and_aggregate_windows(
            input_ids,
            attention_mask,
            pooling_mask,
            indptr=doc_indptr,
            encode_fn=lambda chunk_input_ids, chunk_attention_mask, chunk_pooling_mask: (
                self.model.encode_docs(
                    chunk_input_ids,
                    chunk_attention_mask,
                    pooling_mask=chunk_pooling_mask,
                )
            ),
            pooling_mode=str(self.model.doc_pooling),
            output_dim=int(self.model.encoder.vocab_size),
            output_dtype=next(self.model.parameters()).dtype,
            pad_token_id=self._pad_token_id,
            chunk_size=self._max_windows_per_forward,
            use_fixed_size_chunks=self._use_fixed_window_chunks,
            mark_step=self._torch_compile_mark_step,
            entity_name="document",
        )

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
        vocab_size: int = int(self.model.encoder.vocab_size)
        raw_exclude_token_ids: list[int] = self._resolve_exclude_token_ids()
        exclude_output_ids: list[int] = self._resolve_exclude_output_ids(
            raw_exclude_token_ids
        )
        output_space: OutputSpaceSpec = self.model.encoder.output_space
        writer_cfg = SparseWriterConfig(
            output_dir=encode_path,
            vocab_size=vocab_size,
            rank=int(self.trainer.global_rank),
            top_k=self.cfg.encoding.sparse_top_k,
            min_weight=float(self.cfg.encoding.sparse_min_weight),
            exclude_output_ids=exclude_output_ids,
            source_exclude_token_ids=raw_exclude_token_ids,
            model_family=str(self.model.family),
            output_space=output_space,
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
                exclude_output_ids=writer_cfg.exclude_output_ids,
                source_exclude_token_ids=writer_cfg.source_exclude_token_ids,
                model_family=writer_cfg.model_family,
                output_space=writer_cfg.output_space,
                shard_max_docs=writer_cfg.shard_max_docs,
                value_dtype=writer_cfg.value_dtype,
            )
            self._async_writer = None
        if exclude_output_ids:
            self._exclude_output_ids_tensor = torch.tensor(
                exclude_output_ids, dtype=torch.long, device=self.device
            )
        else:
            self._exclude_output_ids_tensor = None
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
        # Keep all ranks alive until every writer has finalized its last shard.
        maybe_barrier()

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _ = batch_idx
        if self._async_write_enabled and self._async_writer is None:
            raise RuntimeError("Async writer is not initialized.")
        if not self._async_write_enabled and self._writer is None:
            raise RuntimeError("Writer is not initialized.")
        doc_ids: list[str] = list(batch["doc_ids"])
        doc_input_ids: torch.Tensor = batch["doc_input_ids"]
        doc_attention_mask: torch.Tensor = batch["doc_attention_mask"]
        doc_pooling_mask: torch.Tensor | None = batch.get("doc_pooling_mask")
        doc_indptr: torch.Tensor = batch["doc_indptr"]
        if doc_input_ids.ndim != 2 or doc_attention_mask.ndim != 2:
            raise ValueError("Encoding batches must have shape (windows, seq_len).")
        if doc_indptr.ndim != 1:
            raise ValueError("doc_indptr must have shape (batch + 1,).")
        flattened_input_ids: torch.Tensor = doc_input_ids.to(
            self.device, non_blocking=True
        )
        flattened_attention_mask: torch.Tensor = doc_attention_mask.to(
            self.device, non_blocking=True
        )
        flattened_pooling_mask: torch.Tensor | None = None
        if doc_pooling_mask is not None:
            flattened_pooling_mask = doc_pooling_mask.to(
                self.device, non_blocking=True
            )
        doc_reps: torch.Tensor = self._encode_and_aggregate_window_batch(
            flattened_input_ids,
            flattened_attention_mask,
            flattened_pooling_mask,
            doc_indptr,
        )
        indptr, indices, values = self._sparsify_batch(doc_reps)
        if self._async_writer is not None:
            self._async_writer.check_healthy()
            indptr.share_memory_()
            indices.share_memory_()
            values.share_memory_()
            self._async_writer.write(doc_ids, indptr, indices, values)
        elif self._writer is not None:
            self._writer.write_sparse_csr_batch(doc_ids, indptr, indices, values)
