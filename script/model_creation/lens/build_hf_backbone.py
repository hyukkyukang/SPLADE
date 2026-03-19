import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.prototype.embeddinggemma_lsr.cli import (
    apply_config_overrides,
    parser_default_values,
    resolve_torch_device,
    resolve_torch_dtype,
)

_DEFAULT_INSTRUCTION_TOKEN: str = "<instruct>"
_DEFAULT_QUERY_TOKEN: str = "<query>"
_DEFAULT_RESPONSE_TOKEN: str = "<response>"
_METADATA_FILENAME: str = "lens_backbone_metadata.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Hugging Face CausalLM backbone for LENS by adding the "
            "configured special tokens and saving a self-contained model dir."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional OmegaConf YAML.")
    parser.add_argument("--base-model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/model_creation/lens/hf_backbone",
    )
    parser.add_argument(
        "--instruction-token",
        type=str,
        default=_DEFAULT_INSTRUCTION_TOKEN,
    )
    parser.add_argument(
        "--query-token",
        type=str,
        default=_DEFAULT_QUERY_TOKEN,
    )
    parser.add_argument(
        "--response-token",
        type=str,
        default=_DEFAULT_RESPONSE_TOKEN,
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _default_values() -> dict[str, Any]:
    return parser_default_values(_build_parser())


def _apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    return apply_config_overrides(args, defaults=_default_values())


def _resolve_special_tokens(args: argparse.Namespace) -> list[str]:
    ordered_tokens: list[str] = [
        str(args.instruction_token),
        str(args.query_token),
        str(args.response_token),
    ]
    unique_tokens: list[str] = []
    seen: set[str] = set()
    token: str
    for token in ordered_tokens:
        normalized: str = token.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_tokens.append(normalized)
    if len(unique_tokens) < 3:
        raise ValueError(
            "instruction/query/response tokens must resolve to three distinct, "
            "non-empty strings."
        )
    return unique_tokens


def _set_pad_token_if_missing(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token is not None:
        return
    fallback_token: str | None = tokenizer.eos_token or tokenizer.unk_token
    if fallback_token is None:
        raise ValueError(
            "Tokenizer has no pad_token and no eos_token/unk_token fallback."
        )
    tokenizer.pad_token = fallback_token


def _register_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    *,
    special_tokens: list[str],
) -> tuple[int, list[str]]:
    existing_special_tokens: set[str] = set(tokenizer.all_special_tokens)
    tokens_to_register: list[str] = [
        token for token in special_tokens if token not in existing_special_tokens
    ]
    if not tokens_to_register:
        return 0, []
    added_count: int = int(
        tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens_to_register}
        )
    )
    return added_count, tokens_to_register


def _build_model_kwargs(
    *,
    trust_remote_code: bool,
    local_files_only: bool,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(trust_remote_code),
        "dtype": dtype,
    }
    if bool(local_files_only):
        model_kwargs["local_files_only"] = True
    return model_kwargs


def _build_tokenizer_kwargs(
    *,
    trust_remote_code: bool,
    local_files_only: bool,
) -> dict[str, Any]:
    tokenizer_kwargs: dict[str, Any] = {
        "use_fast": True,
        "trust_remote_code": bool(trust_remote_code),
    }
    if bool(local_files_only):
        tokenizer_kwargs["local_files_only"] = True
    return tokenizer_kwargs


def _save_metadata(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)


def main() -> None:
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args()
    args = _apply_config_overrides(args)

    output_dir: Path = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype: torch.dtype = resolve_torch_dtype(str(args.dtype))
    device: torch.device = resolve_torch_device(str(args.device))
    special_tokens: list[str] = _resolve_special_tokens(args)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        str(args.base_model),
        **_build_tokenizer_kwargs(
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
        ),
    )
    tokenizer_length_before: int = len(tokenizer)
    added_count: int
    registered_tokens: list[str]
    added_count, registered_tokens = _register_special_tokens(
        tokenizer,
        special_tokens=special_tokens,
    )
    _set_pad_token_if_missing(tokenizer)

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        **_build_model_kwargs(
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
            dtype=dtype,
        ),
    )
    model.to(device)

    tokenizer_length_after: int = len(tokenizer)
    if tokenizer_length_after != tokenizer_length_before:
        model.resize_token_embeddings(tokenizer_length_after)

    special_token_ids: dict[str, int] = {
        token: int(tokenizer.convert_tokens_to_ids(token)) for token in special_tokens
    }
    setattr(model.config, "lens_base_model_name_or_path", str(args.base_model))
    setattr(model.config, "lens_instruction_token", str(args.instruction_token))
    setattr(model.config, "lens_query_token", str(args.query_token))
    setattr(model.config, "lens_response_token", str(args.response_token))
    setattr(model.config, "lens_special_tokens", special_tokens)
    setattr(model.config, "lens_special_token_ids", special_token_ids)
    setattr(model.config, "pad_token_id", tokenizer.pad_token_id)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata: dict[str, Any] = {
        "base_model": str(args.base_model),
        "output_dir": str(output_dir),
        "dtype": str(args.dtype),
        "device": str(device),
        "tokenizer_length_before": tokenizer_length_before,
        "tokenizer_length_after": tokenizer_length_after,
        "added_token_count": added_count,
        "registered_special_tokens": registered_tokens,
        "special_tokens": special_tokens,
        "special_token_ids": special_token_ids,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": (
            int(tokenizer.pad_token_id)
            if tokenizer.pad_token_id is not None
            else None
        ),
    }
    _save_metadata(output_dir / _METADATA_FILENAME, metadata)

    print(f"Saved LENS backbone to {output_dir}")
    print(
        "Special tokens: "
        + ", ".join(
            f"{token}={token_id}" for token, token_id in special_token_ids.items()
        )
    )


if __name__ == "__main__":
    main()
