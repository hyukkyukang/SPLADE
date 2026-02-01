from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from omegaconf import DictConfig
from sentence_transformers import SparseEncoder
from sentence_transformers.models import Normalize
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from src.utils.model_utils import resolve_model_dtype
from src.utils.transformers import build_tokenizer
from transformers import PreTrainedTokenizerBase

logger: logging.Logger = logging.getLogger("src.utils.sparse_encoder")


@dataclass
class SparseEncoderCache:
    """Cache of NanoBEIR SparseEncoder components for reuse."""

    mlm_transformer: MLMTransformer
    sparse_encoder: SparseEncoder


class _ModelCardDataStub:
    """Minimal model card interface for evaluation hooks."""

    def set_evaluation_metrics(
        self,
        evaluator: Any,
        metrics: dict[str, Any],
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        _ = evaluator, metrics, epoch, step


class DocOnlySparseEncoderAdapter:
    """Adapter to evaluate SPLADE-doc models with NanoBEIR evaluators."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: torch.device,
        batch_size: int,
        max_query_length: int,
        max_doc_length: int,
    ) -> None:
        self.model: torch.nn.Module = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.device: torch.device = device
        self.batch_size: int = int(batch_size)
        self.max_query_length: int = int(max_query_length)
        self.max_doc_length: int = int(max_doc_length)
        self.similarity_fn_name: str = "dot"
        self.model_card_data: _ModelCardDataStub = _ModelCardDataStub()

    @staticmethod
    def sparsity(embeddings: torch.Tensor) -> dict[str, float]:
        """Proxy to SparseEncoder.sparsity for evaluator stats."""
        return SparseEncoder.sparsity(embeddings)

    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute dot-product scores for dense or sparse tensors."""
        if a.is_sparse:
            a = a.to_dense()
        if b.is_sparse:
            b = b.to_dense()
        return torch.mm(a, b.transpose(0, 1))

    def encode(
        self,
        sentences: str | Sequence[str] | np.ndarray,
        *,
        is_query: bool | None = None,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
        convert_to_sparse_tensor: bool = True,
        save_to_cpu: bool = True,
        max_active_dims: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _ = prompt_name, prompt, kwargs
        if is_query:
            return self.encode_query(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_sparse_tensor=convert_to_sparse_tensor,
                save_to_cpu=save_to_cpu,
                max_active_dims=max_active_dims,
            )
        return self.encode_document(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_sparse_tensor=convert_to_sparse_tensor,
            save_to_cpu=save_to_cpu,
            max_active_dims=max_active_dims,
        )

    def encode_query(
        self,
        sentences: str | Sequence[str] | np.ndarray,
        *,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
        convert_to_sparse_tensor: bool = True,
        save_to_cpu: bool = True,
        max_active_dims: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _ = prompt_name, prompt, kwargs
        return self._encode_texts(
            sentences=sentences,
            max_length=self.max_query_length,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_sparse_tensor=convert_to_sparse_tensor,
            save_to_cpu=save_to_cpu,
            max_active_dims=max_active_dims,
            encode_fn=self.model.encode_queries,
        )

    def encode_document(
        self,
        sentences: str | Sequence[str] | np.ndarray,
        *,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
        convert_to_sparse_tensor: bool = True,
        save_to_cpu: bool = True,
        max_active_dims: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _ = prompt_name, prompt, kwargs
        return self._encode_texts(
            sentences=sentences,
            max_length=self.max_doc_length,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_sparse_tensor=convert_to_sparse_tensor,
            save_to_cpu=save_to_cpu,
            max_active_dims=max_active_dims,
            encode_fn=self.model.encode_docs,
        )

    def _encode_texts(
        self,
        *,
        sentences: str | Sequence[str] | np.ndarray,
        max_length: int,
        batch_size: int | None,
        show_progress_bar: bool,
        convert_to_sparse_tensor: bool,
        save_to_cpu: bool,
        max_active_dims: int | None,
        encode_fn: Any,
    ) -> torch.Tensor:
        text_list: list[str]
        if isinstance(sentences, str):
            text_list = [sentences]
        elif isinstance(sentences, np.ndarray):
            text_list = [str(item) for item in sentences.tolist()]
        else:
            text_list = [str(item) for item in sentences]

        if not text_list:
            vocab_size: int = int(self.model.encoder.mlm.config.vocab_size)
            empty: torch.Tensor = torch.empty(
                (0, vocab_size), dtype=self.model.encoder.mlm.dtype, device=self.device
            )
            return empty

        batch_size_value: int = int(batch_size or self.batch_size)
        batches: Iterable[list[str]] = _batch_texts(
            text_list, batch_size_value, show_progress_bar
        )
        outputs: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for batch in batches:
                tokens: dict[str, torch.Tensor] = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=int(max_length),
                    return_tensors="pt",
                )
                input_ids: torch.Tensor = tokens["input_ids"].to(self.device)
                attention_mask: torch.Tensor = tokens["attention_mask"].to(self.device)
                batch_reps: torch.Tensor = encode_fn(input_ids, attention_mask)
                outputs.append(batch_reps)

        embeddings: torch.Tensor = torch.cat(outputs, dim=0)
        if max_active_dims is not None:
            embeddings = _prune_to_max_active_dims(embeddings, int(max_active_dims))
        if convert_to_sparse_tensor:
            embeddings = embeddings.to_sparse()
        if save_to_cpu:
            embeddings = embeddings.to(device=torch.device("cpu"))
        return embeddings


def _batch_texts(
    texts: list[str], batch_size: int, show_progress_bar: bool
) -> Iterable[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    indices: range = range(0, len(texts), batch_size)
    if show_progress_bar:
        try:
            from tqdm.auto import tqdm

            indices = tqdm(indices, desc="Encoding", leave=False)
        except ImportError:  # pragma: no cover - tqdm is optional
            pass
    for start in indices:
        yield texts[start : start + batch_size]


def _prune_to_max_active_dims(
    embeddings: torch.Tensor, max_active_dims: int
) -> torch.Tensor:
    if max_active_dims <= 0 or embeddings.numel() == 0:
        return embeddings
    top_k: int = min(int(max_active_dims), int(embeddings.shape[1]))
    values: torch.Tensor
    indices: torch.Tensor
    values, indices = torch.topk(embeddings, top_k, dim=1)
    pruned: torch.Tensor = torch.zeros_like(embeddings)
    pruned.scatter_(1, indices, values)
    return pruned


def _strip_prefix(value: str, prefixes: Iterable[str]) -> str | None:
    """Return value with the first matching prefix stripped."""
    prefix: str
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def resolve_nanobeir_compatibility(cfg: DictConfig) -> tuple[bool, str | None]:
    """Check if the config is compatible with NanoBEIR SparseEncoder evaluation."""
    query_pooling: str = str(cfg.model.query_pooling)
    doc_pooling: str = str(cfg.model.doc_pooling)
    if query_pooling != doc_pooling:
        return (
            False,
            f"query_pooling must match doc_pooling (got {query_pooling} vs {doc_pooling}).",
        )
    sparse_activation: str = str(cfg.model.sparse_activation)
    if sparse_activation != "log1p_relu":
        return (
            False,
            f"sparse_activation must be log1p_relu (got {sparse_activation}).",
        )
    return True, None


def build_doc_only_sparse_encoder_adapter(
    cfg: DictConfig,
    model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> DocOnlySparseEncoderAdapter:
    """Build a NanoBEIR adapter for SPLADE-doc query encoding."""
    tokenizer: PreTrainedTokenizerBase = build_tokenizer(
        str(cfg.model.huggingface_name)
    )
    max_length: int = int(cfg.model.max_input_length)
    return DocOnlySparseEncoderAdapter(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(batch_size),
        max_query_length=max_length,
        max_doc_length=max_length,
    )


def _extract_mlm_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extract MLM-only weights from a Lightning checkpoint state dict."""
    prefixes: list[str] = [
        "model.encoder._orig_mod.mlm.",
        "encoder._orig_mod.mlm.",
        "model.encoder.mlm.",
        "encoder.mlm.",
        "model.mlm.",
        "mlm.",
    ]
    mlm_state: dict[str, torch.Tensor] = {}
    key: str
    value: torch.Tensor
    for key, value in state_dict.items():
        stripped_key: str | None = _strip_prefix(key, prefixes)
        if stripped_key is None:
            continue
        mlm_state[stripped_key] = value

    if not mlm_state:
        raise ValueError(
            "No MLM weights found in checkpoint. Expected keys starting with "
            "'model.encoder.mlm.' or 'encoder.mlm.'."
        )
    return mlm_state


def _build_mlm_transformer(cfg: DictConfig) -> MLMTransformer:
    """Build a SentenceTransformers MLMTransformer configured for SPLADE."""
    # Resolve dtype and attention implementation to match the training setup.
    dtype: torch.dtype | None = resolve_model_dtype(
        str(cfg.model.dtype), bool(cfg.testing.use_cpu)
    )
    model_args: dict[str, Any] = {}
    attn_implementation: str | None = cfg.model.attn_implementation
    if attn_implementation:
        model_args["attn_implementation"] = attn_implementation
    if dtype is not None:
        model_args["torch_dtype"] = dtype

    max_input_length: int = int(cfg.model.max_input_length)
    tokenizer_args: dict[str, Any] = {"model_max_length": max_input_length}
    nanobeir_cfg: DictConfig = cfg.nanobeir
    cache_dir: str | None = nanobeir_cfg.cache_dir

    mlm_transformer: MLMTransformer = MLMTransformer(
        model_name_or_path=str(cfg.model.huggingface_name),
        max_seq_length=max_input_length,
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        cache_dir=cache_dir,
    )
    return mlm_transformer


def _load_mlm_transformer_from_state_dict(
    cfg: DictConfig,
    mlm_state_dict: dict[str, torch.Tensor],
) -> MLMTransformer:
    """Load MLMTransformer weights from a provided state dict."""
    mlm_transformer: MLMTransformer = _build_mlm_transformer(cfg)
    _load_mlm_state_dict(mlm_transformer, mlm_state_dict)
    return mlm_transformer


def _load_mlm_state_dict(
    mlm_transformer: MLMTransformer,
    mlm_state_dict: dict[str, torch.Tensor],
) -> None:
    """Load MLM weights into an existing MLMTransformer."""
    incompatible: Any = mlm_transformer.auto_model.load_state_dict(
        mlm_state_dict, strict=False
    )
    missing_keys: list[str] = list(incompatible.missing_keys)
    unexpected_keys: list[str] = list(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        logger.warning(
            "Loaded MLM weights with missing=%d unexpected=%d",
            len(missing_keys),
            len(unexpected_keys),
        )


def _load_mlm_transformer(
    cfg: DictConfig,
    checkpoint_path: str,
) -> MLMTransformer:
    """Load an MLMTransformer and override weights from Lightning checkpoint."""
    mlm_transformer: MLMTransformer = _build_mlm_transformer(cfg)
    # Load the Lightning checkpoint on CPU to avoid device mismatches.
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    raw_state_dict: dict[str, Any] = checkpoint.get("state_dict", checkpoint)
    state_dict: dict[str, torch.Tensor] = {}
    raw_key: str
    raw_value: Any
    for raw_key, raw_value in raw_state_dict.items():
        if isinstance(raw_value, torch.Tensor):
            state_dict[raw_key] = raw_value
    mlm_state_dict: dict[str, torch.Tensor] = _extract_mlm_state_dict(state_dict)
    _load_mlm_state_dict(mlm_transformer, mlm_state_dict)
    return mlm_transformer


def _build_sparse_encoder_from_mlm(
    cfg: DictConfig,
    mlm_transformer: MLMTransformer,
    device: torch.device,
) -> SparseEncoder:
    """Build a SparseEncoder module stack from an MLMTransformer."""
    compatible: bool
    reason: str | None
    compatible, reason = resolve_nanobeir_compatibility(cfg)
    if not compatible:
        raise ValueError(f"NanoBEIR evaluation incompatible: {reason}")

    doc_pooling: str = str(cfg.model.doc_pooling)
    # SentenceTransformers SpladePooling applies log1p after ReLU.
    activation_function: str = "relu"

    splade_pooling: SpladePooling = SpladePooling(
        pooling_strategy=doc_pooling,
        activation_function=activation_function,
        word_embedding_dimension=mlm_transformer.get_sentence_embedding_dimension(),
    )

    modules: list[Any] = [mlm_transformer, splade_pooling]
    if bool(cfg.model.normalize):
        # SentenceTransformers Normalize module mirrors L2 normalization.
        modules.append(Normalize())

    sparse_encoder: SparseEncoder = SparseEncoder(
        modules=modules, similarity_fn_name="dot"
    )
    sparse_encoder.to(device)
    sparse_encoder.eval()
    return sparse_encoder


def build_sparse_encoder_from_checkpoint(
    cfg: DictConfig,
    checkpoint_path: str,
    device: torch.device,
) -> SparseEncoder:
    """Build a SentenceTransformers SparseEncoder from a Lightning checkpoint."""
    # Build the MLM + pooling stack and optional normalization.
    mlm_transformer: MLMTransformer = _load_mlm_transformer(
        cfg=cfg, checkpoint_path=checkpoint_path
    )
    return _build_sparse_encoder_from_mlm(cfg, mlm_transformer, device)


def build_sparse_encoder_from_huggingface(
    cfg: DictConfig,
    device: torch.device,
) -> SparseEncoder:
    """Build a SentenceTransformers SparseEncoder from a Hugging Face model."""
    # Use the Hugging Face weights directly without checkpoint overrides.
    mlm_transformer: MLMTransformer = _build_mlm_transformer(cfg)
    return _build_sparse_encoder_from_mlm(cfg, mlm_transformer, device)


def build_sparse_encoder_from_model(
    cfg: DictConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> SparseEncoder:
    """Build a SentenceTransformers SparseEncoder from an in-memory SPLADE model."""
    raw_state_dict: dict[str, torch.Tensor] = model.state_dict()
    mlm_state_dict: dict[str, torch.Tensor] = _extract_mlm_state_dict(raw_state_dict)
    cpu_state_dict: dict[str, torch.Tensor] = {
        key: value.detach().to("cpu") for key, value in mlm_state_dict.items()
    }
    mlm_transformer: MLMTransformer = _build_mlm_transformer(cfg)
    _load_mlm_state_dict(mlm_transformer, cpu_state_dict)
    return _build_sparse_encoder_from_mlm(cfg, mlm_transformer, device)


def build_sparse_encoder_cache(
    cfg: DictConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> SparseEncoderCache:
    """Build a cached SparseEncoder with weights loaded from the model."""
    raw_state_dict: dict[str, torch.Tensor] = model.state_dict()
    mlm_state_dict: dict[str, torch.Tensor] = _extract_mlm_state_dict(raw_state_dict)
    cpu_state_dict: dict[str, torch.Tensor] = {
        key: value.detach().to("cpu") for key, value in mlm_state_dict.items()
    }
    mlm_transformer: MLMTransformer = _build_mlm_transformer(cfg)
    _load_mlm_state_dict(mlm_transformer, cpu_state_dict)
    sparse_encoder: SparseEncoder = _build_sparse_encoder_from_mlm(
        cfg, mlm_transformer, device
    )
    return SparseEncoderCache(
        mlm_transformer=mlm_transformer, sparse_encoder=sparse_encoder
    )


def update_sparse_encoder_cache(
    cache: SparseEncoderCache,
    model: torch.nn.Module,
    device: torch.device,
) -> SparseEncoder:
    """Update cached SparseEncoder weights and move to device."""
    raw_state_dict: dict[str, torch.Tensor] = model.state_dict()
    mlm_state_dict: dict[str, torch.Tensor] = _extract_mlm_state_dict(raw_state_dict)
    cpu_state_dict: dict[str, torch.Tensor] = {
        key: value.detach().to("cpu") for key, value in mlm_state_dict.items()
    }
    _load_mlm_state_dict(cache.mlm_transformer, cpu_state_dict)
    cache.sparse_encoder.to(device)
    cache.sparse_encoder.eval()
    return cache.sparse_encoder
