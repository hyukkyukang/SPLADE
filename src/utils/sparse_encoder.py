from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SparseEncoder
from sentence_transformers.models import Normalize
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from transformers import PreTrainedTokenizerBase

from src.data.lens_formatting import (
    build_doc_pooling_mask,
    build_query_pooling_mask,
    format_query_text,
    resolve_instruction_text,
    validate_lens_tokenizer,
)
from src.utils.logging import get_logger
from src.utils.lens_instructions import resolve_benchmark_instruction
from src.utils.model_utils import resolve_model_dtype
from src.utils.normalize import normalize_optional_str
from src.utils.peft import is_peft_enabled
from src.utils.transformers import build_tokenizer, resolve_model_name_or_path

logger = get_logger("src.utils.sparse_encoder")
_VALID_BENCHMARK_ADAPTERS: set[str] = {
    "auto",
    "native",
    "sentence_transformers",
}


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


def _clone_model_cfg_with_instruction(
    model_cfg: DictConfig, instruction_text: str
) -> DictConfig:
    copied_cfg: DictConfig = OmegaConf.create(
        OmegaConf.to_container(model_cfg, resolve=False)
    )
    copied_cfg.instruction_text = instruction_text
    return copied_cfg


def _resolve_query_model_cfg(
    model_cfg: DictConfig,
    *,
    prompt_name: str | None,
    prompt: str | None,
) -> DictConfig:
    instruction_text: str = resolve_benchmark_instruction(
        model_cfg,
        prompt_name=prompt_name,
        prompt=prompt,
    )
    if instruction_text == resolve_instruction_text(model_cfg):
        return model_cfg
    return _clone_model_cfg_with_instruction(model_cfg, instruction_text)


class NativeSparseEncoderAdapter:
    """Adapter to evaluate in-memory sparse models without MLMTransformer."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        *,
        model_cfg: DictConfig,
        device: torch.device,
        batch_size: int,
        max_query_length: int,
        max_doc_length: int,
    ) -> None:
        self.model: torch.nn.Module = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model_cfg: DictConfig = model_cfg
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
        _ = kwargs
        if is_query:
            return self.encode_query(
                sentences,
                prompt_name=prompt_name,
                prompt=prompt,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_sparse_tensor=convert_to_sparse_tensor,
                save_to_cpu=save_to_cpu,
                max_active_dims=max_active_dims,
            )
        return self.encode_document(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
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
            is_query=True,
            prompt_name=prompt_name,
            prompt=prompt,
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
            is_query=False,
            prompt_name=prompt_name,
            prompt=prompt,
            encode_fn=self.model.encode_docs,
        )

    def encode_corpus(
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
        return self.encode_document(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_sparse_tensor=convert_to_sparse_tensor,
            save_to_cpu=save_to_cpu,
            max_active_dims=max_active_dims,
            **kwargs,
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
        is_query: bool,
        prompt_name: str | None,
        prompt: str | None,
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
            embeddings = _finalize_encoded_batch(
                torch.empty(
                    (0, int(self.model.encoder.vocab_size)),
                    dtype=self.model.encoder.dtype,
                    device=self.device,
                ),
                max_active_dims=max_active_dims,
                convert_to_sparse_tensor=convert_to_sparse_tensor,
                save_to_cpu=save_to_cpu,
            )
            return embeddings

        batch_size_value: int = int(batch_size or self.batch_size)
        batches: Iterable[list[str]] = _batch_texts(
            text_list, batch_size_value, show_progress_bar
        )
        outputs: list[torch.Tensor] = []
        query_model_cfg: DictConfig | None = None
        if is_query:
            query_model_cfg = _resolve_query_model_cfg(
                self.model_cfg,
                prompt_name=prompt_name,
                prompt=prompt,
            )
        self.model.eval()
        with torch.no_grad():
            for batch in batches:
                batch_texts: list[str] = batch
                if is_query:
                    batch_texts = [
                        format_query_text(text, query_model_cfg) for text in batch
                    ]
                tokens: dict[str, torch.Tensor] = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=int(max_length),
                    return_tensors="pt",
                )
                input_ids: torch.Tensor = tokens["input_ids"].to(self.device)
                attention_mask: torch.Tensor = tokens["attention_mask"].to(self.device)
                pooling_mask: torch.Tensor
                if is_query:
                    pooling_mask = build_query_pooling_mask(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        tokenizer=self.tokenizer,
                        model_cfg=query_model_cfg,
                    ).to(self.device)
                else:
                    pooling_mask = build_doc_pooling_mask(
                        attention_mask=attention_mask,
                        model_cfg=self.model_cfg,
                    ).to(self.device)
                batch_reps: torch.Tensor = encode_fn(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pooling_mask=pooling_mask,
                )
                outputs.append(
                    _finalize_encoded_batch(
                        batch_reps,
                        max_active_dims=max_active_dims,
                        convert_to_sparse_tensor=convert_to_sparse_tensor,
                        save_to_cpu=save_to_cpu,
                    )
                )

        return _concat_encoded_batches(outputs)


DocOnlySparseEncoderAdapter = NativeSparseEncoderAdapter


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


def _finalize_encoded_batch(
    embeddings: torch.Tensor,
    *,
    max_active_dims: int | None,
    convert_to_sparse_tensor: bool,
    save_to_cpu: bool,
) -> torch.Tensor:
    """Apply pruning/storage transforms per batch to bound peak memory."""
    processed: torch.Tensor = embeddings
    if max_active_dims is not None:
        processed = _prune_to_max_active_dims(processed, int(max_active_dims))
    if convert_to_sparse_tensor:
        processed = processed.to_sparse().coalesce()
    if save_to_cpu:
        processed = processed.to(device=torch.device("cpu"))
    return processed


def _concat_encoded_batches(outputs: list[torch.Tensor]) -> torch.Tensor:
    if not outputs:
        raise ValueError("Expected at least one encoded batch to concatenate.")
    if len(outputs) == 1:
        return outputs[0]
    embeddings: torch.Tensor = torch.cat(outputs, dim=0)
    if embeddings.is_sparse:
        return embeddings.coalesce()
    return embeddings


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


def resolve_benchmark_adapter(
    model_cfg: DictConfig | None,
) -> str:
    raw_value: str = normalize_optional_str(
        model_cfg.get("benchmark_adapter") if model_cfg is not None else None
    ) or "auto"
    resolved_value: str = raw_value.lower().replace("-", "_")
    if resolved_value not in _VALID_BENCHMARK_ADAPTERS:
        raise ValueError(
            "model.benchmark_adapter must be one of: auto, native, "
            f"sentence_transformers. Got: {raw_value!r}"
        )
    return resolved_value


def resolve_nanobeir_backend(
    cfg: DictConfig,
    *,
    doc_only_enabled: bool | None = None,
) -> tuple[str, str | None]:
    model_cfg: DictConfig = cfg.model
    benchmark_adapter: str = resolve_benchmark_adapter(model_cfg)
    family: str = str(model_cfg.get("family", "splade")).strip().lower()
    peft_cfg: DictConfig | None = (
        model_cfg.get("peft") if "peft" in model_cfg else None
    )
    if bool(doc_only_enabled) or bool(model_cfg.get("doc_only", False)):
        return "native", "model.doc_only requires native sparse query encoding."
    if family == "lens":
        return "native", "model.family=lens requires the native benchmark adapter."
    if is_peft_enabled(peft_cfg):
        return "native", "PEFT-wrapped models require the native benchmark adapter."

    compatible: bool
    reason: str | None
    compatible, reason = resolve_nanobeir_compatibility(cfg)
    if not compatible:
        return "native", reason

    if benchmark_adapter == "native":
        return "native", "model.benchmark_adapter=native"
    return "sentence_transformers", None


def build_native_sparse_encoder_adapter(
    cfg: DictConfig,
    model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> NativeSparseEncoderAdapter:
    """Build a native sparse adapter for models not served by MLMTransformer."""
    tokenizer: PreTrainedTokenizerBase = build_tokenizer(
        str(cfg.model.huggingface_name),
        use_fast_tokenizer=bool(cfg.model.use_fast_tokenizer),
        trust_remote_code=bool(cfg.model.trust_remote_code),
        require_fast_tokenizer=bool(cfg.model.require_fast_tokenizer),
    )
    validate_lens_tokenizer(tokenizer, cfg.model)
    max_length: int = int(cfg.nanobeir.max_seq_length)
    return NativeSparseEncoderAdapter(
        model=model,
        tokenizer=tokenizer,
        model_cfg=cfg.model,
        device=device,
        batch_size=int(batch_size),
        max_query_length=max_length,
        max_doc_length=max_length,
    )


def build_doc_only_sparse_encoder_adapter(
    cfg: DictConfig,
    model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> DocOnlySparseEncoderAdapter:
    """Backward-compatible alias for the native sparse adapter builder."""
    return build_native_sparse_encoder_adapter(
        cfg=cfg,
        model=model,
        device=device,
        batch_size=batch_size,
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
        normalized_key: str = key.replace("_orig_mod.", "")
        stripped_key: str | None = _strip_prefix(normalized_key, prefixes)
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
    config_args: dict[str, Any] = {}
    attn_implementation: str | None = cfg.model.attn_implementation
    if attn_implementation:
        model_args["attn_implementation"] = attn_implementation
    if dtype is not None:
        model_args["torch_dtype"] = dtype
    config_args["tie_word_embeddings"] = bool(cfg.model.tie_word_embeddings)

    max_seq_length: int = int(cfg.nanobeir.max_seq_length)
    tokenizer_args: dict[str, Any] = {"model_max_length": max_seq_length}
    nanobeir_cfg: DictConfig = cfg.nanobeir
    cache_dir: str | None = nanobeir_cfg.cache_dir
    model_name_or_path: str = resolve_model_name_or_path(str(cfg.model.huggingface_name))

    mlm_transformer: MLMTransformer = MLMTransformer(
        model_name_or_path=model_name_or_path,
        max_seq_length=max_seq_length,
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        config_args=config_args,
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
