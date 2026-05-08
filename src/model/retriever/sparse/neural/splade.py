import inspect
from pathlib import Path
from typing import Any, Callable, Optional, cast

import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import DictConfig
from transformers import PreTrainedModel

from src.utils.logging import suppress_output_if_not_rank_zero
from src.utils.output_space import (
    OutputSpaceSpec,
    normalize_compact_head_alignment,
)
from src.utils.compact_head import (
    COMPACT_HEAD_FILENAME,
    OFFICIAL_LENS_HEAD_FILENAME,
    load_compact_head_payload,
    torch_load_compact_head_artifact,
)
from src.utils.peft import (
    apply_peft_adapter,
    resolve_peft_settings,
    unwrap_peft_model,
)
from src.utils.transformers import build_masked_lm_model, resolve_model_name_or_path

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
    unwrapped_model: nn.Module = unwrap_peft_model(model)
    if unwrapped_model is not model:
        if isinstance(unwrapped_model, PreTrainedModel):
            return _extract_hidden_module(unwrapped_model)
        if hasattr(unwrapped_model, "model"):
            module: Any = getattr(unwrapped_model, "model")
            if isinstance(module, nn.Module):
                return _extract_hidden_module(cast(PreTrainedModel, module))
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


def _filter_forward_kwargs(
    module: nn.Module,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Drop unsupported kwargs for backbones that expose narrow forward signatures."""
    if not kwargs:
        return {}
    forward_fn: Any = getattr(module, "forward", None)
    if not callable(forward_fn):
        return {}
    try:
        parameters = inspect.signature(forward_fn).parameters.values()
    except (TypeError, ValueError):
        return {}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return dict(kwargs)
    supported_names: set[str] = {parameter.name for parameter in parameters}
    return {name: value for name, value in kwargs.items() if name in supported_names}


def _extract_logits_from_model_output(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    logits: Any = getattr(model_output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if (
        isinstance(model_output, tuple)
        and len(model_output) > 0
        and isinstance(model_output[0], torch.Tensor)
    ):
        return model_output[0]
    raise ValueError("Language-model forward did not return accessible logits.")


def _resolve_module_dtype(
    module: nn.Module,
    *,
    fallback: torch.dtype = torch.float32,
) -> torch.dtype:
    """Resolve module dtype without assuming a `.dtype` attribute exists."""
    parameter: nn.Parameter
    for parameter in module.parameters():
        return parameter.dtype
    buffer: torch.Tensor
    for buffer in module.buffers():
        return buffer.dtype
    dtype_attr: Any = getattr(module, "dtype", None)
    if isinstance(dtype_attr, torch.dtype):
        return dtype_attr
    return fallback


def _maybe_download_hf_artifact(repo_id: str, filename: str) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    try:
        resolved_path: str = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception:
        return None
    candidate: Path = Path(resolved_path).expanduser()
    return candidate if candidate.is_file() else None


def _resolve_compact_head_path(
    model_name: str,
    model: PreTrainedModel,
) -> Path | None:
    """Return compact-head artifact path when available."""
    resolved_source: str = resolve_model_name_or_path(model_name)
    candidate_filenames: list[str] = []
    config_file: Any = getattr(model.config, "splade_compact_head_file", None)
    if config_file is not None and str(config_file).strip():
        candidate_filenames.append(str(config_file).strip())
    for filename in (COMPACT_HEAD_FILENAME, OFFICIAL_LENS_HEAD_FILENAME):
        if filename not in candidate_filenames:
            candidate_filenames.append(filename)

    for source in (resolved_source, model_name):
        model_dir: Path = Path(source).expanduser()
        if not model_dir.is_dir():
            continue
        candidate: Path
        for filename in candidate_filenames:
            candidate = model_dir / filename
            if candidate.is_file():
                return candidate

    seen_repo_ids: set[str] = set()
    repo_id: str
    for repo_id in (str(resolved_source).strip(), str(model_name).strip()):
        if not repo_id or repo_id in seen_repo_ids:
            continue
        seen_repo_ids.add(repo_id)
        if Path(repo_id).expanduser().exists():
            continue
        for filename in candidate_filenames:
            downloaded: Path | None = _maybe_download_hf_artifact(repo_id, filename)
            if downloaded is not None:
                return downloaded
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
        peft_cfg: DictConfig | None = None,
        freeze_backbone: bool = False,
        trust_remote_code: bool = False,
        model_revision: str | None = None,
        local_files_only: bool | None = None,
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
                trust_remote_code=trust_remote_code,
                revision=model_revision,
                local_files_only=local_files_only,
                **kwargs,
            )
        resolved_peft_settings = resolve_peft_settings(
            peft_cfg,
            model_type=getattr(self.mlm.config, "model_type", None),
            huggingface_model_class=huggingface_model_class,
        )
        self._peft_trainable_parameter_names: frozenset[str] = frozenset()
        self._peft_enabled: bool = False
        self._peft_method: str | None = None
        if resolved_peft_settings.enabled:
            self.mlm, self._peft_trainable_parameter_names = apply_peft_adapter(
                self.mlm,
                settings=resolved_peft_settings,
            )
            self._peft_enabled = True
            self._peft_method = resolved_peft_settings.method
        self.sparse_activation: str = sparse_activation
        self.activation: nn.Module = _resolve_activation_module(sparse_activation)
        self._neg_inf: torch.Tensor
        self.register_buffer("_neg_inf", torch.tensor(float("-inf")), persistent=False)
        self._output_vocab_size: int = int(self.mlm.config.vocab_size)
        self._compact_head_alignment: str = "token_ids"
        self._output_token_aligned: bool = True
        self._output_space: OutputSpaceSpec = OutputSpaceSpec.from_alignment(
            vocab_size=self._output_vocab_size,
            compact_head_alignment="token_ids",
        )
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
        self._encode_logits: Callable[..., torch.Tensor]
        self._setup_compact_head(model_name=model_name, dtype=dtype)
        self._encode_logits = self._resolve_encode_logits()
        self.freeze_backbone: bool = bool(freeze_backbone) and not self._peft_enabled
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

        payload: dict[str, Any] = load_compact_head_payload(
            torch_load_compact_head_artifact(artifact_path)
        )
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

        compact_head_alignment: str | None = normalize_compact_head_alignment(
            payload.get("alignment")
        )
        if compact_head_alignment is None:
            compact_head_alignment = (
                "token_ids"
                if int(token_ids.numel()) == out_features
                else "latent_cluster"
            )
        if (
            compact_head_alignment == "token_ids"
            and int(token_ids.numel()) != out_features
        ):
            raise ValueError(
                "Token-aligned compact heads must include one token id per output row."
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
        self._output_space = OutputSpaceSpec.from_alignment(
            vocab_size=out_features,
            compact_head_alignment=compact_head_alignment,
        )
        self._compact_head_alignment = self._output_space.compact_head_alignment
        self._output_token_aligned = self._output_space.output_token_aligned
        self._drop_unused_mlm_head_for_compact_path()
        self._freeze_unused_mlm_params_for_compact_path()
        if self._output_token_aligned:
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
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        forward_kwargs.update(_filter_forward_kwargs(self.mlm, model_kwargs))
        return _extract_logits_from_model_output(
            self.mlm(
                **forward_kwargs,
            )
        )

    def _encode_logits_mlm_no_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        forward_kwargs.update(_filter_forward_kwargs(self.mlm, model_kwargs))
        return _extract_logits_from_model_output(
            self.mlm(
                **forward_kwargs,
            )
        )

    def _encode_logits_compact(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        hidden_model: nn.Module = cast(nn.Module, self._hidden_model)
        compact_head: nn.Linear = cast(nn.Linear, self.compact_head)
        hidden_forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        hidden_forward_kwargs.update(
            _filter_forward_kwargs(hidden_model, model_kwargs)
        )
        hidden_states: torch.Tensor = hidden_model(
            **hidden_forward_kwargs
        ).last_hidden_state
        return compact_head(hidden_states.to(dtype=compact_head.weight.dtype))

    def _encode_logits_compact_no_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        hidden_model: nn.Module = cast(nn.Module, self._hidden_model)
        compact_head: nn.Linear = cast(nn.Linear, self.compact_head)
        hidden_forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
            "use_cache": False,
        }
        hidden_forward_kwargs.update(
            _filter_forward_kwargs(hidden_model, model_kwargs)
        )
        hidden_states: torch.Tensor = hidden_model(
            **hidden_forward_kwargs
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

    def resolve_output_exclude_ids(
        self, exclude_token_ids: torch.Tensor | None
    ) -> torch.Tensor:
        """Map tokenizer token IDs onto output-dimension IDs when supported."""
        return self._output_space.resolve_exclude_token_ids(
            exclude_token_ids,
            token_id_to_output_index=self._token_id_to_output_index,
        )

    def build_exclude_mask(self, exclude_ids: torch.Tensor) -> torch.Tensor:
        """Build an output-dimension mask from tokenizer token ids."""
        resolved_output_ids: torch.Tensor = self.resolve_output_exclude_ids(exclude_ids)
        if int(resolved_output_ids.numel()) == 0:
            return torch.empty((0,), dtype=torch.bool)
        vocab_size: int = int(self._output_vocab_size)
        mask: torch.Tensor = torch.zeros(vocab_size, dtype=torch.bool)
        mask[resolved_output_ids] = True
        return mask

    @property
    def vocab_size(self) -> int:
        return int(self._output_vocab_size)

    @property
    def peft_enabled(self) -> bool:
        return bool(self._peft_enabled)

    @property
    def peft_method(self) -> str | None:
        return self._peft_method

    @property
    def compact_head_alignment(self) -> str:
        return self._compact_head_alignment

    @property
    def output_token_aligned(self) -> bool:
        return bool(self._output_token_aligned)

    @property
    def token_id_to_output_index(self) -> torch.Tensor:
        return self._token_id_to_output_index

    @property
    def output_space(self) -> OutputSpaceSpec:
        return self._output_space

    @property
    def dtype(self) -> torch.dtype:
        return _resolve_module_dtype(self.mlm)

    def train(self, mode: bool = True) -> "SpladeEncoder":
        super().train(mode)
        if self.freeze_backbone:
            # Keep backbone deterministic when it is frozen.
            self.mlm.eval()
        return self

    def _pool_sparse(
        self,
        token_scores: torch.Tensor,
        pooling_mask: torch.Tensor,
        pooling_mode: torch.Tensor,
    ) -> torch.Tensor:
        # Expand mask for token-wise pooling.
        mask: torch.Tensor = pooling_mask.unsqueeze(-1).to(token_scores.dtype)
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
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits: torch.Tensor = self._encode_logits(input_ids, attention_mask)
        token_scores: torch.Tensor = self.activation(logits)
        resolved_pooling_mask: torch.Tensor = (
            attention_mask if pooling_mask is None else pooling_mask
        )
        embeddings: torch.Tensor = self._pool_sparse(
            token_scores, resolved_pooling_mask, pooling_mode
        )
        return embeddings

    def encode_raw_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Return raw LM logits before SPLADE activation/pooling."""
        return self._encode_logits(input_ids, attention_mask, **model_kwargs)


class _SpladeEncoderWrapper(nn.Module):
    def __init__(self, encoder: SpladeEncoder, pooling_mode: torch.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("_pooling_mode", pooling_mode, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mode=self._pooling_mode,
            pooling_mask=pooling_mask,
        )


class SpladeModel(nn.Module):
    # --- Special methods ---
    def __init__(
        self,
        family: str,
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
        peft_cfg: DictConfig | None = None,
        freeze_backbone: bool = False,
        trust_remote_code: bool = False,
        model_revision: str | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        super().__init__()
        self.family: str = str(family).lower()
        # Build encoder shared by query and document pooling.
        self.encoder: SpladeEncoder = SpladeEncoder(
            model_name=model_name,
            sparse_activation=sparse_activation,
            huggingface_model_class=huggingface_model_class,
            attn_implementation=attn_implementation,
            dtype=dtype,
            tie_word_embeddings=tie_word_embeddings,
            peft_cfg=peft_cfg,
            freeze_backbone=freeze_backbone,
            trust_remote_code=trust_remote_code,
            model_revision=model_revision,
            local_files_only=local_files_only,
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
        self.peft_enabled: bool = bool(self.encoder.peft_enabled)
        if self.doc_only and not self.encoder.output_token_aligned:
            raise ValueError(
                "model.doc_only requires token-aligned output dimensions; "
                "clustered compact heads are not supported."
            )
        exclude_token_ids: torch.Tensor = self._build_query_exclude_token_ids()
        self.register_buffer(
            "_query_exclude_token_ids", exclude_token_ids, persistent=False
        )
        self._query_encode_fn: Callable[..., torch.Tensor] = (
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
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode queries as a bag-of-words over input tokens."""
        batch_size: int = int(input_ids.shape[0])
        vocab_size: int = int(self.encoder.vocab_size)
        device: torch.device = input_ids.device
        dtype: torch.dtype = self.encoder.dtype

        token_ids: torch.Tensor = input_ids.to(dtype=torch.long)
        # Mask out padding and special tokens before counting terms.
        resolved_pooling_mask: torch.Tensor = (
            attention_mask if pooling_mask is None else pooling_mask
        )
        token_mask: torch.Tensor = resolved_pooling_mask.to(dtype=torch.bool)
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
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode queries using the MLM-based SPLADE encoder."""
        embeddings: torch.Tensor = self._query_encoder_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mask=pooling_mask,
        )
        return embeddings

    # --- Public methods ---
    def postprocess_query_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def postprocess_doc_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self._query_encode_fn(
            input_ids,
            attention_mask,
            pooling_mask,
        )
        return self.postprocess_query_embeddings(embeddings)

    def encode_docs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self._doc_encoder_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mask=pooling_mask,
        )
        return self.postprocess_doc_embeddings(embeddings)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        query_pooling_mask: torch.Tensor | None = None,
        doc_pooling_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q: torch.Tensor = self.encode_queries(
            query_input_ids,
            query_attention_mask,
            pooling_mask=query_pooling_mask,
        )
        d: torch.Tensor = self.encode_docs(
            doc_input_ids,
            doc_attention_mask,
            pooling_mask=doc_pooling_mask,
        )
        return q, d
