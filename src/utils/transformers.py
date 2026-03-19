from pathlib import Path
from typing import Any, Callable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from src.model.retriever.sparse.neural.bidirectional_mistral import (
    MistralBiForCausalLM,
)
from src.utils.normalize import normalize_optional_str as _normalize_optional_str

_CHECKPOINT_PRIORITY_FILES: tuple[str, ...] = (
    "last.ckpt",
    "mteb-best.ckpt",
    "mteb-best-v1.ckpt",
)
_MODEL_STATE_PREFIXES: tuple[str, ...] = (
    "model._orig_mod.",
    "model.module.",
    "model.",
)
_DEFAULT_HF_MODEL_CLASS_NAME: str = "AutoModelForMaskedLM"
_HF_MODEL_CLASSES: dict[str, Any] = {
    "AutoModelForMaskedLM": AutoModelForMaskedLM,
    "AutoModelForCausalLM": AutoModelForCausalLM,
    "MistralBiForCausalLM": MistralBiForCausalLM,
}


def _normalize_local_path(path_value: str) -> Path | None:
    candidate: Path = Path(path_value).expanduser()
    if not candidate.exists():
        return None
    return candidate


def _extract_tensor_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw_state_dict: Any = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state_dict, dict):
        raise ValueError(
            "Unexpected checkpoint format: expected dict or dict['state_dict']."
        )
    state_dict: dict[str, torch.Tensor] = {}
    raw_key: Any
    raw_value: Any
    for raw_key, raw_value in raw_state_dict.items():
        if isinstance(raw_value, torch.Tensor):
            state_dict[str(raw_key)] = raw_value
    if not state_dict:
        raise ValueError("Checkpoint does not contain tensor parameters.")
    return state_dict


def _strip_model_prefix(key: str) -> str:
    prefix: str
    for prefix in _MODEL_STATE_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def resolve_checkpoint_path(model_name_or_path: str) -> Path | None:
    local_path: Path | None = _normalize_local_path(model_name_or_path)
    if local_path is None:
        return None
    if local_path.is_file() and local_path.suffix == ".ckpt":
        return local_path
    if not local_path.is_dir():
        return None

    prioritized: list[Path] = [local_path / name for name in _CHECKPOINT_PRIORITY_FILES]
    prioritized.extend(
        local_path / "checkpoints" / name for name in _CHECKPOINT_PRIORITY_FILES
    )
    candidate: Path
    for candidate in prioritized:
        if candidate.is_file():
            return candidate

    def _select_latest_checkpoint(checkpoint_files: list[Path]) -> Path | None:
        if not checkpoint_files:
            return None
        return max(
            checkpoint_files,
            key=lambda path: (
                float(path.stat().st_mtime),
                path.name,
            ),
        )

    checkpoint_dir: Path = local_path / "checkpoints"
    if checkpoint_dir.is_dir():
        checkpoint_files: list[Path] = list(checkpoint_dir.glob("*.ckpt"))
        selected_checkpoint: Path | None = _select_latest_checkpoint(checkpoint_files)
        if selected_checkpoint is not None:
            return selected_checkpoint

    root_checkpoint_files: list[Path] = list(local_path.glob("*.ckpt"))
    selected_checkpoint = _select_latest_checkpoint(root_checkpoint_files)
    if selected_checkpoint is not None:
        return selected_checkpoint
    return None


def _resolve_checkpoint_path(model_name_or_path: str) -> Path | None:
    """Backward-compatible alias for older internal callers/tests."""
    return resolve_checkpoint_path(model_name_or_path)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        try:
            return {str(key): item for key, item in value.items()}
        except Exception:
            return {}
    return {}


def _extract_hparams(checkpoint: dict[str, Any]) -> dict[str, Any]:
    hparams: Any = checkpoint.get("hyper_parameters")
    if hparams is None:
        hparams = checkpoint.get("hparams")
    return _as_dict(hparams)


def _extract_model_hparams(checkpoint: dict[str, Any]) -> dict[str, Any]:
    hparams: dict[str, Any] = _extract_hparams(checkpoint)
    return _as_dict(hparams.get("model"))


def _append_dir_candidate(
    candidates: list[Path], seen: set[str], candidate: Path | None
) -> None:
    if candidate is None:
        return
    candidate_path: Path = candidate.expanduser()
    if not candidate_path.is_absolute():
        candidate_path = Path.cwd() / candidate_path
    key: str = str(candidate_path)
    if key in seen:
        return
    seen.add(key)
    if candidate_path.is_dir():
        candidates.append(candidate_path)


def _derive_run_directory(model_name_or_path: str, checkpoint_path: Path) -> Path:
    local_path: Path | None = _normalize_local_path(model_name_or_path)
    if local_path is None:
        return checkpoint_path.parent
    if local_path.is_dir():
        return local_path
    checkpoint_parent: Path = checkpoint_path.parent
    if checkpoint_parent.name == "checkpoints":
        return checkpoint_parent.parent
    return checkpoint_parent


def _collect_local_fallback_dirs(model_name_or_path: str) -> list[Path]:
    local_path: Path | None = _normalize_local_path(model_name_or_path)
    if local_path is None:
        return []

    candidates: list[Path] = []
    seen: set[str] = set()
    if local_path.is_dir():
        _append_dir_candidate(candidates, seen, local_path)
        if local_path.name.startswith("trained_"):
            basename: str = local_path.name[8:]
            _append_dir_candidate(candidates, seen, local_path.parent / basename)
            _append_dir_candidate(
                candidates, seen, local_path.parent / "data" / "model" / basename
            )
            _append_dir_candidate(
                candidates, seen, Path.cwd() / "data" / "model" / basename
            )

    checkpoint_path: Path | None = resolve_checkpoint_path(model_name_or_path)
    if checkpoint_path is None:
        return candidates

    run_dir: Path = _derive_run_directory(model_name_or_path, checkpoint_path)
    _append_dir_candidate(candidates, seen, run_dir)
    _append_dir_candidate(candidates, seen, run_dir / "checkpoints")
    return candidates


def _looks_like_hf_model_dir(path: Path) -> bool:
    config_path: Path = path / "config.json"
    if not config_path.is_file():
        return False
    try:
        import json

        config_data: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(config_data, dict):
        return False
    return _normalize_optional_str(config_data.get("model_type")) is not None


def _looks_like_hf_tokenizer_dir(path: Path) -> bool:
    return bool(
        (path / "tokenizer_config.json").is_file()
        or (path / "tokenizer.json").is_file()
        or (path / "vocab.txt").is_file()
    )


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    """Resolve the best local HF source path for model/tokenizer loading."""
    local_path: Path | None = _normalize_local_path(model_name_or_path)
    if local_path is None:
        return model_name_or_path
    if local_path.is_dir() and _looks_like_hf_model_dir(local_path):
        return str(local_path)
    candidate_dir: Path
    for candidate_dir in _collect_local_fallback_dirs(model_name_or_path):
        if _looks_like_hf_model_dir(candidate_dir):
            return str(candidate_dir)
    return model_name_or_path


def _load_tokenizer_from_source(
    source: str,
    *,
    use_fast_tokenizer: bool,
    trust_remote_code: bool,
    require_fast_tokenizer: bool,
    local_files_only: bool | None,
) -> PreTrainedTokenizerBase:
    tokenizer_kwargs: dict[str, object] = {
        "use_fast": bool(use_fast_tokenizer),
        "trust_remote_code": bool(trust_remote_code),
    }
    if local_files_only is not None:
        tokenizer_kwargs["local_files_only"] = bool(local_files_only)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        source,
        **tokenizer_kwargs,
    )
    if bool(require_fast_tokenizer) and not bool(tokenizer.is_fast):
        raise ValueError(
            "Fast tokenizer is required but the loaded tokenizer is slow: "
            f"{source}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    return tokenizer


def build_tokenizer(
    model_name: str,
    *,
    use_fast_tokenizer: bool = True,
    trust_remote_code: bool = False,
    require_fast_tokenizer: bool = False,
    local_files_only: bool | None = None,
) -> PreTrainedTokenizerBase:
    """Build a tokenizer with local-checkpoint fallback for ANNA-style runs."""
    sources: list[str] = [model_name]
    resolved_model_source: str = resolve_model_name_or_path(model_name)
    if resolved_model_source not in sources:
        sources.append(resolved_model_source)
    candidate_dir: Path
    for candidate_dir in _collect_local_fallback_dirs(model_name):
        if not _looks_like_hf_tokenizer_dir(candidate_dir):
            continue
        candidate_source: str = str(candidate_dir)
        if candidate_source not in sources:
            sources.append(candidate_source)

    first_error: Exception | None = None
    load_attempts: list[tuple[bool, bool]] = [
        (bool(use_fast_tokenizer), bool(require_fast_tokenizer))
    ]
    if bool(use_fast_tokenizer) and not bool(require_fast_tokenizer):
        # Fall back to a slow tokenizer when fast backend construction fails.
        load_attempts.append((False, False))
    source: str
    for source in sources:
        use_fast: bool
        require_fast: bool
        for use_fast, require_fast in load_attempts:
            try:
                return _load_tokenizer_from_source(
                    source,
                    use_fast_tokenizer=use_fast,
                    trust_remote_code=trust_remote_code,
                    require_fast_tokenizer=require_fast,
                    local_files_only=local_files_only,
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                continue
    if first_error is not None:
        raise first_error
    raise RuntimeError(f"Failed to load tokenizer for source: {model_name}")


def _parse_indexed_layer_key(
    key: str, *, prefix: str, group_name: str
) -> tuple[int, str]:
    body: str = key[len(prefix) :]
    parts: list[str] = body.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid {group_name} parameter key: {key}")
    layer_index: int = int(parts[0])
    return layer_index, parts[1]


def _build_bert_config_from_hparams(
    model_hparams: dict[str, Any],
    *,
    tie_word_embeddings: bool,
    attn_implementation: str | None,
) -> BertConfig:
    num_early_layers: int = int(model_hparams["num_early_layers"])
    num_late_layers: int = int(model_hparams["num_late_layers"])
    config: BertConfig = BertConfig(
        vocab_size=int(model_hparams["vocab_size"]),
        hidden_size=int(model_hparams["hidden_size"]),
        num_hidden_layers=num_early_layers + num_late_layers,
        num_attention_heads=int(model_hparams["num_attention_heads"]),
        intermediate_size=int(model_hparams["intermediate_size"]),
        hidden_dropout_prob=float(model_hparams.get("hidden_dropout_prob", 0.1)),
        attention_probs_dropout_prob=float(
            model_hparams.get("attention_probs_dropout_prob", 0.1)
        ),
        max_position_embeddings=int(model_hparams.get("max_position_embeddings", 512)),
        type_vocab_size=int(model_hparams.get("type_vocab_size", 2)),
        initializer_range=float(model_hparams.get("initializer_range", 0.02)),
        layer_norm_eps=float(model_hparams.get("layer_norm_eps", 1e-12)),
    )
    config.tie_word_embeddings = bool(tie_word_embeddings)
    if attn_implementation is not None:
        setattr(config, "_attn_implementation", attn_implementation)
        setattr(config, "attn_implementation", attn_implementation)
    return config


def _map_condenser_state_dict_to_bert(
    *,
    state_dict: dict[str, torch.Tensor],
    num_early_layers: int,
) -> dict[str, torch.Tensor]:
    mapped_state_dict: dict[str, torch.Tensor] = {}
    key: str
    value: torch.Tensor
    ignored_prefixes: tuple[str, ...] = ("head_layers.",)
    for key, value in state_dict.items():
        if key.startswith("embeddings."):
            mapped_state_dict[f"bert.{key}"] = value
            continue
        if key.startswith("early_layers."):
            layer_index, tail = _parse_indexed_layer_key(
                key, prefix="early_layers.", group_name="early_layers"
            )
            mapped_state_dict[f"bert.encoder.layer.{layer_index}.{tail}"] = value
            continue
        if key.startswith("late_layers."):
            layer_index, tail = _parse_indexed_layer_key(
                key, prefix="late_layers.", group_name="late_layers"
            )
            mapped_state_dict[
                f"bert.encoder.layer.{num_early_layers + layer_index}.{tail}"
            ] = value
            continue
        if key.startswith("mlm_head."):
            mapped_state_dict[f"cls.{key[len('mlm_head.'):]}"] = value
            continue
        if key.startswith(ignored_prefixes):
            continue
        raise ValueError(f"Unsupported checkpoint parameter for fallback load: {key}")
    return mapped_state_dict


def _looks_like_condenser_checkpoint(state_dict: dict[str, torch.Tensor]) -> bool:
    has_embeddings: bool = any(key.startswith("embeddings.") for key in state_dict)
    has_early_layers: bool = any(key.startswith("early_layers.") for key in state_dict)
    has_late_layers: bool = any(key.startswith("late_layers.") for key in state_dict)
    has_mlm_head: bool = any(key.startswith("mlm_head.") for key in state_dict)
    return has_embeddings and has_early_layers and has_late_layers and has_mlm_head


def _maybe_load_condenser_checkpoint(
    model_name_or_path: str,
    *,
    attn_implementation: str | None,
    dtype: torch.dtype | None,
    tie_word_embeddings: bool,
) -> PreTrainedModel | None:
    checkpoint_path: Path | None = resolve_checkpoint_path(model_name_or_path)
    if checkpoint_path is None:
        return None

    checkpoint: dict[str, Any] = torch.load(str(checkpoint_path), map_location="cpu")
    raw_state_dict: dict[str, torch.Tensor] = _extract_tensor_state_dict(checkpoint)
    state_dict: dict[str, torch.Tensor] = {
        _strip_model_prefix(key): value for key, value in raw_state_dict.items()
    }
    if not _looks_like_condenser_checkpoint(state_dict):
        return None

    model_hparams: dict[str, Any] = _extract_model_hparams(checkpoint)
    if not model_hparams:
        raise ValueError(
            "Fallback checkpoint loading requires `hyper_parameters.model` in the "
            f"checkpoint: {checkpoint_path}"
        )

    num_early_layers: int = int(model_hparams["num_early_layers"])
    config: BertConfig = _build_bert_config_from_hparams(
        model_hparams,
        tie_word_embeddings=tie_word_embeddings,
        attn_implementation=attn_implementation,
    )
    mapped_state_dict: dict[str, torch.Tensor] = _map_condenser_state_dict_to_bert(
        state_dict=state_dict,
        num_early_layers=num_early_layers,
    )
    model: BertForMaskedLM = BertForMaskedLM(config)
    model.load_state_dict(mapped_state_dict, strict=True)
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    return model


def _resolve_hf_model_loader(
    model_class_name: str | None,
) -> tuple[str, Any]:
    resolved_name: str = _DEFAULT_HF_MODEL_CLASS_NAME
    if model_class_name is not None:
        normalized_name: str = str(model_class_name).strip()
        if normalized_name:
            resolved_name = normalized_name
    model_class: Any | None = _HF_MODEL_CLASSES.get(
        resolved_name
    )
    if model_class is None:
        supported_names: str = ", ".join(sorted(_HF_MODEL_CLASSES))
        raise ValueError(
            "Unsupported huggingface_model_class "
            f"{resolved_name!r}. Supported values: {supported_names}."
        )
    return resolved_name, model_class


def build_masked_lm_model(
    model_name_or_path: str,
    *,
    model_class_name: str | None = None,
    attn_implementation: str | None = None,
    dtype: torch.dtype | None = None,
    tie_word_embeddings: bool = False,
) -> PreTrainedModel:
    """Build an LM model with optional Condenser checkpoint fallback."""
    model_kwargs: dict[str, Any] = {"tie_word_embeddings": tie_word_embeddings}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    resolved_model_class_name: str
    model_class: Any
    resolved_model_class_name, model_class = _resolve_hf_model_loader(model_class_name)

    try:
        return model_class.from_pretrained(model_name_or_path, **model_kwargs)
    except Exception as primary_error:
        if resolved_model_class_name != _DEFAULT_HF_MODEL_CLASS_NAME:
            raise primary_error
        fallback_model: PreTrainedModel | None = _maybe_load_condenser_checkpoint(
            model_name_or_path,
            attn_implementation=attn_implementation,
            dtype=dtype,
            tie_word_embeddings=tie_word_embeddings,
        )
        if fallback_model is not None:
            return fallback_model
        raise primary_error


__all__: list[str] = [
    "build_masked_lm_model",
    "build_tokenizer",
    "resolve_checkpoint_path",
    "resolve_model_name_or_path",
]
