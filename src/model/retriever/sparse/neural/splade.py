import inspect
from pathlib import Path
from typing import Any, Callable, Optional, cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel

from src.utils.logging import suppress_output_if_not_rank_zero
from src.utils.transformers import build_masked_lm_model, resolve_model_name_or_path

_COMPACT_HEAD_FILENAME: str = "splade_compact_head.pt"


class _Log1pRelu(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.relu(logits))


class _Log1pSoftplus(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.log1p(F.softplus(logits))


class _Softplus(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softplus(logits)


class _Relu(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.relu(logits)


def _resolve_activation_module(activation: str) -> nn.Module:
    if activation == "log1p_relu":
        return _Log1pRelu()
    if activation == "log1p_softplus":
        return _Log1pSoftplus()
    if activation == "softplus":
        return _Softplus()
    if activation == "relu":
        return _Relu()
    raise ValueError(f"Unsupported sparse activation: {activation}")


def _extract_hidden_module(model: PreTrainedModel) -> nn.Module:
    """Resolve the base transformer module that returns last_hidden_state."""
    if hasattr(model, "model"):
        module: Any = getattr(model, "model")
        if isinstance(module, nn.Module):
            return module
    if hasattr(model, "base_model"):
        module = getattr(model, "base_model")
        if isinstance(module, nn.Module):
            return module
    if hasattr(model, "get_decoder"):
        decoder_fn: Any = getattr(model, "get_decoder")
        if callable(decoder_fn):
            module = decoder_fn()
            if isinstance(module, nn.Module):
                return module
    raise ValueError("Unable to resolve hidden-state backbone module from model.")


def _supports_use_cache_forward(module: nn.Module) -> bool:
    """Return True when module.forward accepts a use_cache keyword."""
    forward_fn: Any = getattr(module, "forward", None)
    if not callable(forward_fn):
        return False
    try:
        return "use_cache" in inspect.signature(forward_fn).parameters
    except (TypeError, ValueError):
        return False


def _resolve_compact_head_path(
    model_name: str,
    model: PreTrainedModel,
) -> Path | None:
    """Return compact-head artifact path when available."""
    resolved_source: str = resolve_model_name_or_path(model_name)
    model_dir: Path = Path(resolved_source).expanduser()
    if not model_dir.is_dir():
        return None
    candidates: list[Path] = []
    config_file: Any = getattr(model.config, "splade_compact_head_file", None)
    if config_file is not None and str(config_file).strip():
        candidates.append(model_dir / str(config_file))
    candidates.append(model_dir / _COMPACT_HEAD_FILENAME)
    candidate: Path
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


class SpladeEncoder(nn.Module):
    # --- Special methods ---
    def __init__(
        self,
        model_name: str,
        sparse_activation: str,
        huggingface_model_class: str = "AutoModelForMaskedLM",
        attn_implementation: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        tie_word_embeddings: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        kwargs: dict[str, Any] = {}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        if dtype is not None:
            kwargs["dtype"] = dtype
        kwargs["tie_word_embeddings"] = tie_word_embeddings
        # Load the configured language-model backbone.
        self.mlm: PreTrainedModel
        # Avoid duplicate load reports on non-zero ranks.
        with suppress_output_if_not_rank_zero():
            self.mlm = build_masked_lm_model(
                model_name,
                model_class_name=huggingface_model_class,
                **kwargs,
            )
        self.sparse_activation: str = sparse_activation
        self.activation: nn.Module = _resolve_activation_module(sparse_activation)
        self._neg_inf: torch.Tensor
        self.register_buffer("_neg_inf", torch.tensor(float("-inf")), persistent=False)
        self._output_vocab_size: int = int(self.mlm.config.vocab_size)
        self._mlm_forward_supports_use_cache: bool = _supports_use_cache_forward(
            self.mlm
        )
        self._hidden_forward_supports_use_cache: bool = False
        if hasattr(self.mlm.config, "use_cache"):
            self.mlm.config.use_cache = False
        self.compact_head: nn.Linear | None = None
        self._hidden_model: nn.Module | None = None
        self.register_buffer(
            "_compact_token_ids",
            torch.empty((0,), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_token_id_to_output_index",
            torch.empty((0,), dtype=torch.long),
            persistent=False,
        )
        self._encode_logits: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        self._setup_compact_head(model_name=model_name, dtype=dtype)
        self._encode_logits = self._resolve_encode_logits()
        self.freeze_backbone: bool = bool(freeze_backbone)
        if self.freeze_backbone:
            self._freeze_backbone_params()

    # --- Protected methods ---
    def _setup_compact_head(
        self, *, model_name: str, dtype: Optional[torch.dtype]
    ) -> None:
        """Load an optional compact output head for faster SPLADE logits."""
        artifact_path: Path | None = _resolve_compact_head_path(model_name, self.mlm)
        if artifact_path is None:
            return

        payload: Any = torch.load(str(artifact_path), map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid compact-head payload at {artifact_path}.")
        raw_weight: Any = payload.get("weight")
        if not isinstance(raw_weight, torch.Tensor) or raw_weight.ndim != 2:
            raise ValueError(
                f"Compact-head weight is missing or invalid at {artifact_path}."
            )

        raw_bias: Any = payload.get("bias")
        has_bias: bool = isinstance(raw_bias, torch.Tensor) and raw_bias.ndim == 1
        out_features: int = int(raw_weight.shape[0])
        in_features: int = int(raw_weight.shape[1])
        compact_head: nn.Linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        )
        compact_head.weight.data.copy_(
            raw_weight.to(
                dtype=compact_head.weight.dtype,
                device=compact_head.weight.device,
            )
        )
        if has_bias and compact_head.bias is not None:
            compact_head.bias.data.copy_(
                raw_bias.to(
                    dtype=compact_head.bias.dtype, device=compact_head.bias.device
                )
            )
        if dtype is not None:
            compact_head = compact_head.to(dtype=dtype)

        token_ids: torch.Tensor = torch.empty((0,), dtype=torch.long)
        raw_token_ids: Any = payload.get("token_ids")
        if isinstance(raw_token_ids, torch.Tensor):
            token_ids = raw_token_ids.to(dtype=torch.long).flatten().cpu()
        elif isinstance(raw_token_ids, list):
            token_ids = torch.tensor(
                [int(token_id) for token_id in raw_token_ids],
                dtype=torch.long,
            )

        self.compact_head = compact_head
        self._hidden_model = _extract_hidden_module(self.mlm)
        self._hidden_forward_supports_use_cache = _supports_use_cache_forward(
            self._hidden_model
        )
        hidden_config: Any = getattr(self._hidden_model, "config", None)
        if hidden_config is not None and hasattr(hidden_config, "use_cache"):
            hidden_config.use_cache = False
        self._output_vocab_size = out_features
        self._drop_unused_mlm_head_for_compact_path()
        self._freeze_unused_mlm_params_for_compact_path()
        if int(token_ids.numel()) == out_features:
            self._compact_token_ids = token_ids
            max_token_id: int = int(token_ids.max().item()) if out_features > 0 else -1
            if max_token_id >= 0:
                token_id_to_output_index: torch.Tensor = torch.full(
                    (max_token_id + 1,),
                    fill_value=-1,
                    dtype=torch.long,
                )
                token_id_to_output_index[token_ids] = torch.arange(
                    out_features, dtype=torch.long
                )
                self._token_id_to_output_index = token_id_to_output_index
            else:
                self._token_id_to_output_index = torch.empty((0,), dtype=torch.long)

    def _drop_unused_mlm_head_for_compact_path(self) -> None:
        """Drop output-head params that compact path never uses."""
        if self.compact_head is None:
            return
        set_output_embeddings_fn: Any = getattr(self.mlm, "set_output_embeddings", None)
        if callable(set_output_embeddings_fn):
            try:
                set_output_embeddings_fn(None)
                return
            except Exception:
                pass

        get_output_embeddings_fn: Any = getattr(self.mlm, "get_output_embeddings", None)
        output_head: Any = None
        if callable(get_output_embeddings_fn):
            try:
                output_head = get_output_embeddings_fn()
            except Exception:
                output_head = None
        if not isinstance(output_head, nn.Module):
            return

        for attr_name in ("lm_head", "score", "classifier"):
            if not hasattr(self.mlm, attr_name):
                continue
            try:
                candidate: Any = getattr(self.mlm, attr_name)
            except Exception:
                continue
            if candidate is output_head:
                setattr(self.mlm, attr_name, nn.Identity())
                return

        child_name: str
        child_module: nn.Module
        for child_name, child_module in self.mlm.named_children():
            if child_module is output_head:
                setattr(self.mlm, child_name, nn.Identity())
                return

    def _freeze_unused_mlm_params_for_compact_path(self) -> None:
        """Disable gradients for MLM-only params that compact path never reads."""
        if self.compact_head is None or self._hidden_model is None:
            return
        used_param_ids: set[int] = {
            id(parameter) for parameter in self.compact_head.parameters()
        }
        used_param_ids.update(
            id(parameter) for parameter in self._hidden_model.parameters()
        )
        parameter: nn.Parameter
        for parameter in self.mlm.parameters():
            if id(parameter) in used_param_ids:
                continue
            parameter.requires_grad_(False)

    def _freeze_backbone_params(self) -> None:
        """Freeze backbone params while keeping the active sparse head trainable."""
        parameter: nn.Parameter
        for parameter in self.mlm.parameters():
            parameter.requires_grad_(False)

        if self.compact_head is not None:
            for parameter in self.compact_head.parameters():
                parameter.requires_grad_(True)
        else:
            output_head: Any = self.mlm.get_output_embeddings()
            if isinstance(output_head, nn.Module):
                for parameter in output_head.parameters():
                    parameter.requires_grad_(True)
        self.mlm.eval()

    def _encode_logits_mlm(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    def _encode_logits_mlm_no_cache(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    def _encode_logits_compact(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden_model: nn.Module = cast(nn.Module, self._hidden_model)
        compact_head: nn.Linear = cast(nn.Linear, self.compact_head)
        hidden_states: torch.Tensor = hidden_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        return compact_head(hidden_states.to(dtype=compact_head.weight.dtype))

    def _encode_logits_compact_no_cache(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden_model: nn.Module = cast(nn.Module, self._hidden_model)
        compact_head: nn.Linear = cast(nn.Linear, self.compact_head)
        hidden_states: torch.Tensor = hidden_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        ).last_hidden_state
        return compact_head(hidden_states.to(dtype=compact_head.weight.dtype))

    def _resolve_encode_logits(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.compact_head is None:
            if self._mlm_forward_supports_use_cache:
                return self._encode_logits_mlm_no_cache
            return self._encode_logits_mlm
        hidden_model: nn.Module | None = self._hidden_model
        if hidden_model is None:
            raise ValueError("Compact head is enabled but hidden model is missing.")
        if self._hidden_forward_supports_use_cache:
            return self._encode_logits_compact_no_cache
        return self._encode_logits_compact

    def build_exclude_mask(self, exclude_ids: torch.Tensor) -> torch.Tensor:
        """Build an output-dimension mask from tokenizer token ids."""
        if int(exclude_ids.numel()) == 0:
            return torch.empty((0,), dtype=torch.bool)
        compact_token_ids: torch.Tensor = self._compact_token_ids
        if int(compact_token_ids.numel()) > 0:
            return torch.isin(
                compact_token_ids,
                exclude_ids.to(device=compact_token_ids.device, dtype=torch.long),
            )
        vocab_size: int = int(self._output_vocab_size)
        mask: torch.Tensor = torch.zeros(vocab_size, dtype=torch.bool)
        valid_ids: torch.Tensor = exclude_ids[
            (exclude_ids >= 0) & (exclude_ids < vocab_size)
        ]
        if int(valid_ids.numel()) > 0:
            mask[valid_ids] = True
        return mask

    @property
    def vocab_size(self) -> int:
        return int(self._output_vocab_size)

    @property
    def token_id_to_output_index(self) -> torch.Tensor:
        return self._token_id_to_output_index

    def train(self, mode: bool = True) -> "SpladeEncoder":
        super().train(mode)
        if self.freeze_backbone:
            # Keep backbone deterministic when it is frozen.
            self.mlm.eval()
        return self

    def _pool_sparse(
        self,
        token_scores: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mode: torch.Tensor,
    ) -> torch.Tensor:
        # Expand mask for token-wise pooling.
        mask: torch.Tensor = attention_mask.unsqueeze(-1).to(token_scores.dtype)
        pooled_sum: torch.Tensor = (token_scores * mask).sum(dim=1)
        neg_inf: torch.Tensor = self._neg_inf.to(
            dtype=token_scores.dtype, device=token_scores.device
        )
        masked: torch.Tensor = token_scores.masked_fill(mask == 0, neg_inf)
        pooled_max: torch.Tensor = torch.clamp(masked.max(dim=1).values, min=0.0)
        pooling_value: torch.Tensor = pooling_mode.to(
            dtype=token_scores.dtype, device=token_scores.device
        )
        pooled: torch.Tensor = pooled_sum + (pooled_max - pooled_sum) * pooling_value
        return pooled

    # --- Public methods ---
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mode: torch.Tensor,
    ) -> torch.Tensor:
        logits: torch.Tensor = self._encode_logits(input_ids, attention_mask)
        token_scores: torch.Tensor = self.activation(logits)
        embeddings: torch.Tensor = self._pool_sparse(
            token_scores, attention_mask, pooling_mode
        )
        return embeddings


class _SpladeEncoderWrapper(nn.Module):
    def __init__(self, encoder: SpladeEncoder, pooling_mode: torch.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("_pooling_mode", pooling_mode, persistent=False)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mode=self._pooling_mode,
        )


class SpladeModel(nn.Module):
    # --- Special methods ---
    def __init__(
        self,
        model_name: str,
        huggingface_model_class: str,
        query_pooling: str,
        doc_pooling: str,
        sparse_activation: str,
        attn_implementation: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        normalize: bool = False,
        doc_only: bool = False,
        tie_word_embeddings: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        # Build encoder shared by query and document pooling.
        self.encoder: SpladeEncoder = SpladeEncoder(
            model_name=model_name,
            sparse_activation=sparse_activation,
            huggingface_model_class=huggingface_model_class,
            attn_implementation=attn_implementation,
            dtype=dtype,
            tie_word_embeddings=tie_word_embeddings,
            freeze_backbone=freeze_backbone,
        )
        self.query_pooling: str = query_pooling
        self.doc_pooling: str = doc_pooling
        self._query_pooling_mode: torch.Tensor
        self.register_buffer(
            "_query_pooling_mode",
            torch.tensor(self._resolve_pooling_mode(query_pooling)),
            persistent=False,
        )
        self._doc_pooling_mode: torch.Tensor
        self.register_buffer(
            "_doc_pooling_mode",
            torch.tensor(self._resolve_pooling_mode(doc_pooling)),
            persistent=False,
        )
        self._query_encoder_wrapper: _SpladeEncoderWrapper = _SpladeEncoderWrapper(
            self.encoder, self._query_pooling_mode
        )
        self._doc_encoder_wrapper: _SpladeEncoderWrapper = _SpladeEncoderWrapper(
            self.encoder, self._doc_pooling_mode
        )
        self._query_encoder_fn: Callable[..., torch.Tensor] = (
            self._query_encoder_wrapper
        )
        self._doc_encoder_fn: Callable[..., torch.Tensor] = self._doc_encoder_wrapper
        self.normalize: bool = normalize
        self.doc_only: bool = bool(doc_only)
        exclude_token_ids: torch.Tensor = self._build_query_exclude_token_ids()
        self.register_buffer(
            "_query_exclude_token_ids", exclude_token_ids, persistent=False
        )
        self._query_encode_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            self._encode_query_terms if self.doc_only else self._encode_query_mlm
        )

    # --- Protected methods ---
    @staticmethod
    def _resolve_pooling_mode(pooling: str) -> float:
        if pooling == "sum":
            return 0.0
        if pooling == "max":
            return 1.0
        raise ValueError(f"Unsupported pooling: {pooling}")

    def _build_query_exclude_token_ids(self) -> torch.Tensor:
        """Collect special token IDs to exclude from query bag-of-words."""
        config: Any = self.encoder.mlm.config
        candidate_ids: list[int] = []
        try:
            value: Any = config.pad_token_id
        except AttributeError:
            value = None
        if value is not None:
            token_id: int = int(value)
            if token_id >= 0:
                candidate_ids.append(token_id)
        try:
            value = config.cls_token_id
        except AttributeError:
            value = None
        if value is not None:
            token_id = int(value)
            if token_id >= 0:
                candidate_ids.append(token_id)
        try:
            value = config.sep_token_id
        except AttributeError:
            value = None
        if value is not None:
            token_id = int(value)
            if token_id >= 0:
                candidate_ids.append(token_id)
        try:
            value = config.bos_token_id
        except AttributeError:
            value = None
        if value is not None:
            token_id = int(value)
            if token_id >= 0:
                candidate_ids.append(token_id)
        try:
            value = config.eos_token_id
        except AttributeError:
            value = None
        if value is not None:
            token_id = int(value)
            if token_id >= 0:
                candidate_ids.append(token_id)
        unique_ids: list[int] = sorted(set(candidate_ids))
        return torch.tensor(unique_ids, dtype=torch.long)

    def _encode_query_terms(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode queries as a bag-of-words over input tokens."""
        batch_size: int = int(input_ids.shape[0])
        vocab_size: int = int(self.encoder.vocab_size)
        device: torch.device = input_ids.device
        dtype: torch.dtype = self.encoder.mlm.dtype

        token_ids: torch.Tensor = input_ids.to(dtype=torch.long)
        # Mask out padding and special tokens before counting terms.
        token_mask: torch.Tensor = attention_mask.to(dtype=torch.bool)
        exclude_ids: torch.Tensor = self._query_exclude_token_ids
        if int(exclude_ids.numel()) > 0:
            token_mask = token_mask & ~torch.isin(token_ids, exclude_ids)
        bow: torch.Tensor = torch.zeros(
            (batch_size, vocab_size), dtype=dtype, device=device
        )
        token_id_to_output_index: torch.Tensor = self.encoder.token_id_to_output_index
        if int(token_id_to_output_index.numel()) > 0:
            max_token_id: int = int(token_id_to_output_index.shape[0]) - 1
            safe_token_ids: torch.Tensor = token_ids.clamp(min=0, max=max_token_id)
            in_range_mask: torch.Tensor = (token_ids >= 0) & (token_ids <= max_token_id)
            mapped_ids: torch.Tensor = token_id_to_output_index[safe_token_ids]
            mapped_ids = mapped_ids.masked_fill(~in_range_mask, -1)
            valid_mask: torch.Tensor = token_mask & (mapped_ids >= 0)
            safe_ids: torch.Tensor = mapped_ids.masked_fill(~valid_mask, 0)
            token_values: torch.Tensor = valid_mask.to(dtype=dtype)
            bow.scatter_add_(1, safe_ids, token_values)
            return bow
        # Default path: output ids are tokenizer ids.
        token_values = token_mask.to(dtype=dtype)
        bow.scatter_add_(1, token_ids, token_values)
        return bow

    def _encode_query_mlm(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode queries using the MLM-based SPLADE encoder."""
        embeddings: torch.Tensor = self._query_encoder_fn(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return embeddings

    # --- Public methods ---
    def encode_queries(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self._query_encode_fn(input_ids, attention_mask)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_docs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self._doc_encoder_fn(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q: torch.Tensor = self.encode_queries(query_input_ids, query_attention_mask)
        d: torch.Tensor = self.encode_docs(doc_input_ids, doc_attention_mask)
        return q, d
