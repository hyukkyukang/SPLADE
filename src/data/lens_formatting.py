from typing import Any

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.utils.normalize import normalize_optional_str

_DEFAULT_INSTRUCTION_TOKEN: str = "<instruct>"
_DEFAULT_QUERY_TOKEN: str = "<query>"
_DEFAULT_RESPONSE_TOKEN: str = "<response>"
_DEFAULT_INSTRUCTION_TEMPLATE: str = (
    "{instruction_token}{instruction}\n{query_token}{query}"
)
_VALID_QUERY_MASK_MODES: set[str] = {"attention", "after_query_token", "auto"}
_VALID_MISSING_QUERY_TOKEN_MODES: set[str] = {"zero", "keep"}


def _cfg_get(model_cfg: DictConfig | None, key: str, default: Any = None) -> Any:
    if model_cfg is None:
        return default
    if hasattr(model_cfg, "get"):
        return model_cfg.get(key, default)
    return getattr(model_cfg, key, default)


def is_lens_model(model_cfg: DictConfig | None) -> bool:
    family: str = str(_cfg_get(model_cfg, "family", "splade")).strip().lower()
    return family == "lens"


def resolve_instruction_token(model_cfg: DictConfig | None) -> str:
    return (
        normalize_optional_str(_cfg_get(model_cfg, "instruction_token"))
        or _DEFAULT_INSTRUCTION_TOKEN
    )


def resolve_query_token(model_cfg: DictConfig | None) -> str:
    return (
        normalize_optional_str(_cfg_get(model_cfg, "query_token"))
        or _DEFAULT_QUERY_TOKEN
    )


def resolve_response_token(model_cfg: DictConfig | None) -> str:
    return (
        normalize_optional_str(_cfg_get(model_cfg, "response_token"))
        or _DEFAULT_RESPONSE_TOKEN
    )


def resolve_instruction_template(model_cfg: DictConfig | None) -> str:
    return (
        normalize_optional_str(_cfg_get(model_cfg, "instruction_template"))
        or _DEFAULT_INSTRUCTION_TEMPLATE
    )


def resolve_instruction_text(model_cfg: DictConfig | None) -> str:
    return normalize_optional_str(_cfg_get(model_cfg, "instruction_text")) or ""


def resolve_query_mask_mode(model_cfg: DictConfig | None) -> str:
    raw_mode: str = str(_cfg_get(model_cfg, "query_mask_mode", "auto")).strip().lower()
    if raw_mode not in _VALID_QUERY_MASK_MODES:
        raise ValueError(
            "model.query_mask_mode must be one of: attention, after_query_token, auto. "
            f"Got: {raw_mode!r}"
        )
    if raw_mode == "auto":
        return "after_query_token" if is_lens_model(model_cfg) else "attention"
    return raw_mode


def resolve_doc_trim_last_tokens(model_cfg: DictConfig | None) -> int:
    raw_value: Any = _cfg_get(model_cfg, "doc_trim_last_tokens")
    if raw_value is None:
        return 2 if is_lens_model(model_cfg) else 0
    if isinstance(raw_value, str):
        normalized: str | None = normalize_optional_str(raw_value)
        if normalized is None:
            return 2 if is_lens_model(model_cfg) else 0
        raw_value = normalized
    trim_count: int = int(raw_value)
    if trim_count < 0:
        raise ValueError("model.doc_trim_last_tokens must be >= 0.")
    return trim_count


def resolve_required_special_tokens(model_cfg: DictConfig | None) -> list[str]:
    if not is_lens_model(model_cfg):
        return []
    tokens: list[str] = [
        resolve_instruction_token(model_cfg),
        resolve_query_token(model_cfg),
        resolve_response_token(model_cfg),
    ]
    unique_tokens: list[str] = []
    seen: set[str] = set()
    token: str
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        unique_tokens.append(token)
    return unique_tokens


def validate_lens_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    model_cfg: DictConfig | None,
) -> None:
    if not is_lens_model(model_cfg):
        return
    vocab: dict[str, int] = tokenizer.get_vocab()
    missing_tokens: list[str] = [
        token
        for token in resolve_required_special_tokens(model_cfg)
        if token not in vocab
    ]
    if missing_tokens:
        joined: str = ", ".join(missing_tokens)
        raise ValueError(
            "LENS tokenization requires a tokenizer with the configured special "
            f"tokens already added. Missing: {joined}. Build or load a preprocessed "
            "HF backbone that includes these tokens before training/evaluation."
        )


def format_query_text(query_text: str, model_cfg: DictConfig | None) -> str:
    if not is_lens_model(model_cfg):
        return query_text
    formatted_text: str = resolve_instruction_template(model_cfg).format(
        instruction=resolve_instruction_text(model_cfg),
        query=query_text,
        instruct_token=resolve_instruction_token(model_cfg),
        instruction_token=resolve_instruction_token(model_cfg),
        query_token=resolve_query_token(model_cfg),
        response_token=resolve_response_token(model_cfg),
    )
    if (
        resolve_query_mask_mode(model_cfg) == "after_query_token"
        and resolve_query_token(model_cfg) not in formatted_text
    ):
        raise ValueError(
            "LENS query formatting must include the configured query token when "
            "model.query_mask_mode=after_query_token."
        )
    return formatted_text


def _resolve_special_token_id(
    tokenizer: PreTrainedTokenizerBase, token: str
) -> int:
    vocab: dict[str, int] = tokenizer.get_vocab()
    token_id: int | None = vocab.get(token)
    if token_id is None:
        raise ValueError(
            f"Tokenizer does not contain required special token: {token!r}"
        )
    return int(token_id)


def build_query_pooling_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    model_cfg: DictConfig | None,
    *,
    missing_query_token_mode: str = "zero",
) -> torch.Tensor:
    mode: str = resolve_query_mask_mode(model_cfg)
    if mode == "attention":
        return attention_mask.clone()
    if missing_query_token_mode not in _VALID_MISSING_QUERY_TOKEN_MODES:
        raise ValueError(
            "missing_query_token_mode must be one of: zero, keep. "
            f"Got: {missing_query_token_mode!r}"
        )

    query_token_id: int = _resolve_special_token_id(
        tokenizer, resolve_query_token(model_cfg)
    )
    original_ndim: int = int(input_ids.ndim)
    if original_ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

    pooling_mask: torch.Tensor = attention_mask.clone()
    special_ids: list[int] = [int(token_id) for token_id in tokenizer.all_special_ids]
    special_ids_tensor: torch.Tensor | None = None
    if special_ids:
        special_ids_tensor = torch.tensor(
            special_ids,
            dtype=torch.long,
            device=input_ids.device,
        )

    row_idx: int
    for row_idx in range(int(input_ids.shape[0])):
        active_mask: torch.Tensor = attention_mask[row_idx].to(dtype=torch.bool)
        query_positions: torch.Tensor = torch.nonzero(
            active_mask & (input_ids[row_idx] == query_token_id),
            as_tuple=False,
        ).flatten()
        if int(query_positions.numel()) > 0:
            query_position: int = int(query_positions[0].item())
            pooling_mask[row_idx, : query_position + 1] = 0
            continue
        if missing_query_token_mode == "keep":
            if special_ids_tensor is not None and int(special_ids_tensor.numel()) > 0:
                special_mask: torch.Tensor = torch.isin(
                    input_ids[row_idx], special_ids_tensor
                )
                pooling_mask[row_idx] = pooling_mask[row_idx] * (
                    ~special_mask
                ).to(dtype=pooling_mask.dtype)
            continue
        pooling_mask[row_idx].zero_()

    if original_ndim == 1:
        return pooling_mask.squeeze(0)
    return pooling_mask


def build_doc_pooling_mask(
    attention_mask: torch.Tensor,
    model_cfg: DictConfig | None,
) -> torch.Tensor:
    trim_count: int = resolve_doc_trim_last_tokens(model_cfg)
    pooling_mask: torch.Tensor = attention_mask.clone()
    if trim_count <= 0:
        return pooling_mask

    original_ndim: int = int(pooling_mask.ndim)
    if original_ndim == 1:
        pooling_mask = pooling_mask.unsqueeze(0)

    row_idx: int
    for row_idx in range(int(pooling_mask.shape[0])):
        active_positions: torch.Tensor = torch.nonzero(
            pooling_mask[row_idx].to(dtype=torch.bool),
            as_tuple=False,
        ).flatten()
        if int(active_positions.numel()) == 0:
            continue
        row_trim_count: int = min(trim_count, int(active_positions.numel()))
        trim_positions: torch.Tensor = active_positions[-row_trim_count:]
        pooling_mask[row_idx, trim_positions] = 0

    if original_ndim == 1:
        return pooling_mask.squeeze(0)
    return pooling_mask
