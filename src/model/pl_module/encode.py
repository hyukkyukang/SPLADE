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
from src.utils import is_rank_zero, log_if_rank_zero
from src.utils.logging import get_logger
from src.utils.model_utils import resolve_tagged_output_dir
from src.utils.transformers import build_tokenizer

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

    def _setup_torch_compile(self) -> dict[str, Any]:
        compile_enabled: bool = bool(self.cfg.encoding.get("torch_compile", False))
        compile_available: bool = hasattr(torch, "compile")
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
                exclude_token_ids=self._exclude_token_ids_tensor,
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
        exclude_ids: torch.Tensor | None = self._exclude_token_ids_tensor
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
                exclude_token_ids=exclude_ids,
                threshold=threshold,
                top_k=top_k_int,
            )
        else:
            core_fn = (
                self._sparsify_core_threshold or _sparsify_batch_gpu_csr_core_threshold
            )
            indptr_gpu, flat_indices, flat_values = core_fn(
                vectors,
                exclude_token_ids=exclude_ids,
                threshold=threshold,
            )
        torch_value_dtype: torch.dtype = resolve_torch_dtype(self._value_dtype)
        indptr = indptr_gpu.to(device="cpu")
        indices = flat_indices.to(dtype=torch.int32, device="cpu")
        values = flat_values.to(dtype=torch_value_dtype, device="cpu")
        return indptr, indices, values

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
        indptr, indices, values = self._sparsify_batch(doc_reps)
        if self._async_writer is not None:
            self._async_writer.check_healthy()
            indptr.share_memory_()
            indices.share_memory_()
            values.share_memory_()
            self._async_writer.write(doc_ids, indptr, indices, values)
        elif self._writer is not None:
            self._writer.write_sparse_csr_batch(doc_ids, indptr, indices, values)
