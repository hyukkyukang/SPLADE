import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.prototype.embeddinggemma_lsr.artifacts import (
    DF_MAP_FILENAME,
    load_vocab_artifacts,
    write_json,
    write_text_lines,
)
from src.prototype.embeddinggemma_lsr.cli import (
    apply_config_overrides,
    parser_default_values,
    resolve_torch_device,
    resolve_torch_dtype,
)
from src.prototype.embeddinggemma_lsr.model import resolve_boundary_token_ids
from src.utils.compact_head import (
    COMPACT_HEAD_FILENAME,
    build_token_aligned_compact_head_payload,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Hugging Face CausalLM backbone for SPLADE from EmbeddingGemma "
            "with semantic LM-head initialization over target terms."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional OmegaConf YAML.")
    parser.add_argument("--base-model", type=str, default="google/embeddinggemma-300m")
    parser.add_argument("--vocab-artifact-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--fragment-threshold", type=int, default=4)
    parser.add_argument("--term-batch-size", type=int, default=256)
    parser.add_argument("--max-target-terms", type=int, default=None)

    parser.add_argument(
        "--add-all-target-terms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, add every target term to tokenizer as needed so each term maps "
            "to one token id. If false, only add terms with subword length >= fragment-threshold."
        ),
    )
    parser.add_argument(
        "--allow-unresolved-terms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, skip target terms that still do not map to one token id.",
    )
    parser.add_argument(
        "--zero-non-target-lm-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, zero LM-head rows that are not target-term token ids.",
    )
    parser.add_argument(
        "--non-target-bias",
        type=float,
        default=0.0,
        help="Bias value to apply on non-target rows when output head has bias.",
    )
    parser.add_argument(
        "--untie-word-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, detach output head from input embeddings before LM-head init.",
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


def _validate_required_args(args: argparse.Namespace) -> None:
    required_keys: tuple[str, ...] = ("vocab_artifact_dir", "output_dir")
    key: str
    for key in required_keys:
        value: Any | None = getattr(args, key, None)
        if value is None or not str(value).strip():
            raise ValueError(
                f"Missing required argument `{key}`. "
                "Provide it directly or via --config."
            )


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    return resolve_torch_dtype(dtype_name)


def _resolve_device(device_value: str) -> torch.device:
    return resolve_torch_device(device_value)


def _extract_hidden_module(model: PreTrainedModel) -> nn.Module:
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
    raise ValueError(
        "Unable to resolve a hidden-state backbone module from the CausalLM model."
    )


def _ensure_untied_output_embeddings(model: PreTrainedModel) -> bool:
    output_head: Any = model.get_output_embeddings()
    input_embeddings: Any = model.get_input_embeddings()
    if (
        output_head is None
        or input_embeddings is None
        or not hasattr(output_head, "weight")
        or not hasattr(input_embeddings, "weight")
    ):
        return False

    output_weight: torch.Tensor = output_head.weight
    input_weight: torch.Tensor = input_embeddings.weight
    if output_weight.data_ptr() != input_weight.data_ptr():
        return False

    output_head.weight = nn.Parameter(output_weight.detach().clone())
    if hasattr(model, "config"):
        setattr(model.config, "tie_word_embeddings", False)
    return True


def _resolve_terms_to_add(
    *,
    tokenizer: PreTrainedTokenizerBase,
    target_vocab: list[str],
    fragment_threshold: int,
    add_all_target_terms: bool,
) -> list[str]:
    if add_all_target_terms:
        return list(target_vocab)

    terms_to_add: list[str] = []
    term: str
    for term in target_vocab:
        token_ids: list[int] = list(tokenizer(term, add_special_tokens=False)["input_ids"])
        if len(token_ids) >= int(fragment_threshold):
            terms_to_add.append(term)
    return terms_to_add


def _build_term_to_token_id(
    *,
    tokenizer: PreTrainedTokenizerBase,
    target_vocab: list[str],
) -> tuple[dict[str, int], list[str]]:
    term_to_token_id: dict[str, int] = {}
    unresolved: list[str] = []
    term: str
    for term in target_vocab:
        token_ids: list[int] = list(tokenizer(term, add_special_tokens=False)["input_ids"])
        if len(token_ids) == 1:
            term_to_token_id[term] = int(token_ids[0])
        else:
            unresolved.append(term)
    return term_to_token_id, unresolved


def _compute_semantic_vectors(
    *,
    hidden_module: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    terms: list[str],
    device: torch.device,
    boundary_token_ids: list[int],
    term_batch_size: int,
) -> dict[str, torch.Tensor]:
    if not terms:
        return {}

    vectors: dict[str, torch.Tensor] = {}
    boundary_tensor: torch.Tensor | None = None
    if boundary_token_ids:
        boundary_tensor = torch.tensor(
            boundary_token_ids,
            dtype=torch.long,
            device=device,
        )

    hidden_module.eval()
    with torch.no_grad():
        start: int
        for start in range(0, len(terms), int(term_batch_size)):
            chunk: list[str] = terms[start : start + int(term_batch_size)]
            tokens: dict[str, torch.Tensor] = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            )
            input_ids: torch.Tensor = tokens["input_ids"].to(device)
            attention_mask: torch.Tensor = tokens["attention_mask"].to(device)

            outputs: Any = hidden_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            if not hasattr(outputs, "last_hidden_state"):
                raise ValueError("Backbone module does not expose last_hidden_state.")
            hidden: torch.Tensor = outputs.last_hidden_state

            valid_mask: torch.Tensor = attention_mask.to(dtype=torch.bool)
            if boundary_tensor is not None and int(boundary_tensor.numel()) > 0:
                boundary_mask: torch.Tensor = torch.isin(input_ids, boundary_tensor)
                valid_mask = valid_mask & ~boundary_mask
            fallback_rows: torch.Tensor = valid_mask.sum(dim=1) == 0
            if bool(fallback_rows.any()):
                valid_mask = valid_mask.clone()
                valid_mask[fallback_rows] = attention_mask[fallback_rows].to(dtype=torch.bool)

            valid_weight: torch.Tensor = valid_mask.unsqueeze(-1).to(hidden.dtype)
            summed: torch.Tensor = (hidden * valid_weight).sum(dim=1)
            counts: torch.Tensor = valid_weight.sum(dim=1).clamp(min=1.0)
            mean_vectors: torch.Tensor = summed / counts
            normalized: torch.Tensor = torch.nn.functional.normalize(
                mean_vectors,
                p=2,
                dim=1,
            )

            idx: int
            term: str
            for idx, term in enumerate(chunk):
                vectors[term] = normalized[idx].detach().cpu()
    return vectors


def _initialize_lm_head(
    *,
    model: PreTrainedModel,
    term_to_token_id: dict[str, int],
    semantic_vectors: dict[str, torch.Tensor],
    df_map: dict[str, int],
    alpha: float,
    zero_non_target_lm_head: bool,
    non_target_bias: float,
) -> dict[str, Any]:
    output_head: Any = model.get_output_embeddings()
    if output_head is None or not hasattr(output_head, "weight"):
        raise ValueError("Model output head has no weight parameter.")

    weight: torch.Tensor = output_head.weight.data
    vocab_size: int = int(weight.shape[0])

    target_mask: torch.Tensor = torch.zeros(vocab_size, dtype=torch.bool, device=weight.device)
    target_token_ids: list[int] = []
    token_id: int
    for token_id in term_to_token_id.values():
        if 0 <= int(token_id) < vocab_size:
            target_mask[int(token_id)] = True
            target_token_ids.append(int(token_id))

    if bool(zero_non_target_lm_head):
        weight.zero_()

    bias_tensor: torch.Tensor | None = None
    if hasattr(output_head, "bias") and output_head.bias is not None:
        bias_tensor = output_head.bias.data
        if bool(zero_non_target_lm_head):
            bias_tensor.fill_(float(non_target_bias))

    final_logits_bias: torch.Tensor | None = None
    if hasattr(model, "final_logits_bias"):
        candidate: Any = getattr(model, "final_logits_bias")
        if isinstance(candidate, torch.Tensor):
            final_logits_bias = candidate.data
            if bool(zero_non_target_lm_head):
                final_logits_bias.fill_(float(non_target_bias))

    term: str
    for term, token_id in term_to_token_id.items():
        vector: torch.Tensor | None = semantic_vectors.get(term)
        if vector is None:
            continue
        weight[int(token_id)] = vector.to(device=weight.device, dtype=weight.dtype)
        bias_value: float = -float(alpha) * math.log(float(df_map.get(term, 0)) + 1.0)
        if bias_tensor is not None:
            bias_tensor[int(token_id)] = bias_value
        if final_logits_bias is not None:
            if final_logits_bias.ndim == 2 and int(final_logits_bias.shape[0]) == 1:
                final_logits_bias[0, int(token_id)] = bias_value
            elif final_logits_bias.ndim == 1:
                final_logits_bias[int(token_id)] = bias_value

    target_rows: torch.Tensor = weight[target_mask]
    norms: torch.Tensor = torch.linalg.vector_norm(target_rows.float(), ord=2, dim=1)
    return {
        "vocab_size": vocab_size,
        "target_token_count": int(target_mask.sum().item()),
        "target_weight_norm_min": float(norms.min().item()) if int(norms.numel()) > 0 else 0.0,
        "target_weight_norm_max": float(norms.max().item()) if int(norms.numel()) > 0 else 0.0,
        "target_weight_norm_mean": float(norms.mean().item()) if int(norms.numel()) > 0 else 0.0,
        "output_has_bias": bool(bias_tensor is not None),
        "has_final_logits_bias": bool(final_logits_bias is not None),
    }


def _build_compact_head_payload(
    *,
    terms: list[str],
    semantic_vectors: dict[str, torch.Tensor],
    term_to_token_id: dict[str, int],
    df_map: dict[str, int],
    alpha: float,
) -> dict[str, Any]:
    weight_rows: list[torch.Tensor] = []
    token_ids: list[int] = []
    bias_values: list[float] = []
    term_to_index: dict[str, int] = {}
    idx: int = 0
    term: str
    for term in terms:
        vector: torch.Tensor | None = semantic_vectors.get(term)
        token_id: int | None = term_to_token_id.get(term)
        if vector is None or token_id is None:
            continue
        weight_rows.append(vector.float().cpu())
        token_ids.append(int(token_id))
        bias_values.append(-float(alpha) * math.log(float(df_map.get(term, 0)) + 1.0))
        term_to_index[term] = idx
        idx += 1
    if not weight_rows:
        raise ValueError("No resolved terms were available to build compact head.")
    weight: torch.Tensor = torch.stack(weight_rows, dim=0).contiguous()
    bias: torch.Tensor = torch.tensor(bias_values, dtype=torch.float32)
    return build_token_aligned_compact_head_payload(
        weight=weight,
        bias=bias,
        token_ids=token_ids,
        extra_metadata={
            "terms": [
                term
                for term, _ in sorted(
                    term_to_index.items(),
                    key=lambda item: item[1],
                )
            ],
            "term_to_index": term_to_index,
            "term_to_token_id": {
                term: int(term_to_token_id[term]) for term in term_to_index
            },
        },
    )


def main() -> None:
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args()
    args = _apply_config_overrides(args)
    _validate_required_args(args)

    vocab_artifact_dir: Path = Path(args.vocab_artifact_dir)
    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_vocab, df_map = _load_vocab_artifacts(vocab_artifact_dir)
    if args.max_target_terms is not None:
        max_terms: int = max(int(args.max_target_terms), 1)
        target_vocab = target_vocab[:max_terms]
        df_map = {term: int(df_map.get(term, 0)) for term in target_vocab}

    dtype: torch.dtype = _resolve_dtype(args.dtype)
    device: torch.device = _resolve_device(args.device)

    tokenizer_kwargs: dict[str, Any] = {
        "use_fast": True,
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if bool(args.local_files_only):
        tokenizer_kwargs["local_files_only"] = True
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.base_model,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(args.trust_remote_code),
        "dtype": dtype,
        "tie_word_embeddings": False if bool(args.untie_word_embeddings) else True,
    }
    if bool(args.local_files_only):
        model_kwargs["local_files_only"] = True
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )

    terms_to_add: list[str] = _resolve_terms_to_add(
        tokenizer=tokenizer,
        target_vocab=target_vocab,
        fragment_threshold=int(args.fragment_threshold),
        add_all_target_terms=bool(args.add_all_target_terms),
    )
    added_count: int = int(tokenizer.add_tokens(terms_to_add))
    if added_count > 0:
        model.resize_token_embeddings(len(tokenizer))

    untied: bool = False
    if bool(args.untie_word_embeddings):
        untied = _ensure_untied_output_embeddings(model)

    term_to_token_id, unresolved = _build_term_to_token_id(
        tokenizer=tokenizer,
        target_vocab=target_vocab,
    )
    if unresolved and not bool(args.allow_unresolved_terms):
        sample_unresolved: str = ", ".join(unresolved[:10])
        raise RuntimeError(
            "Some target terms do not map to one token id after tokenizer update. "
            f"unresolved_count={len(unresolved)} sample=[{sample_unresolved}]"
        )
    resolved_terms: list[str] = [term for term in target_vocab if term in term_to_token_id]

    model.to(device)
    hidden_module: nn.Module = _extract_hidden_module(model)
    boundary_token_ids: list[int] = resolve_boundary_token_ids(tokenizer)

    semantic_vectors: dict[str, torch.Tensor] = _compute_semantic_vectors(
        hidden_module=hidden_module,
        tokenizer=tokenizer,
        terms=resolved_terms,
        device=device,
        boundary_token_ids=boundary_token_ids,
        term_batch_size=int(args.term_batch_size),
    )

    lm_stats: dict[str, Any] = _initialize_lm_head(
        model=model,
        term_to_token_id=term_to_token_id,
        semantic_vectors=semantic_vectors,
        df_map=df_map,
        alpha=float(args.alpha),
        zero_non_target_lm_head=bool(args.zero_non_target_lm_head),
        non_target_bias=float(args.non_target_bias),
    )
    compact_head_payload: dict[str, Any] = _build_compact_head_payload(
        terms=resolved_terms,
        semantic_vectors=semantic_vectors,
        term_to_token_id=term_to_token_id,
        df_map=df_map,
        alpha=float(args.alpha),
    )
    compact_head_path: Path = output_dir / COMPACT_HEAD_FILENAME
    torch.save(compact_head_payload, str(compact_head_path))
    setattr(model.config, "splade_compact_head_file", COMPACT_HEAD_FILENAME)
    setattr(model.config, "splade_compact_head_alignment", "token_ids")
    setattr(
        model.config,
        "splade_compact_vocab_size",
        int(compact_head_payload["weight"].shape[0]),
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    init_summary: dict[str, Any] = {
        "base_model": str(args.base_model),
        "output_dir": str(output_dir),
        "target_vocab_size": int(len(target_vocab)),
        "resolved_target_terms": int(len(resolved_terms)),
        "unresolved_target_terms": int(len(unresolved)),
        "fragment_threshold": int(args.fragment_threshold),
        "add_all_target_terms": bool(args.add_all_target_terms),
        "terms_requested_for_add": int(len(terms_to_add)),
        "added_token_count": int(added_count),
        "untied_word_embeddings": bool(untied),
        "alpha": float(args.alpha),
        "dtype": str(dtype),
        "device": str(device),
        "zero_non_target_lm_head": bool(args.zero_non_target_lm_head),
        "non_target_bias": float(args.non_target_bias),
        "compact_head_file": COMPACT_HEAD_FILENAME,
        "compact_vocab_size": int(compact_head_payload["weight"].shape[0]),
        **lm_stats,
    }
    write_text_lines(output_dir / "target_vocab.txt", target_vocab)
    write_json(output_dir / DF_MAP_FILENAME, df_map, sort_keys=True)
    write_json(output_dir / "term_to_token_id.json", term_to_token_id, sort_keys=True)
    write_json(output_dir / "unresolved_terms.json", unresolved)
    write_json(output_dir / "init_summary.json", init_summary, sort_keys=True)

    print(f"Saved HF backbone to {output_dir}")
    print(
        json.dumps(
            {
                "event": "hf_backbone_built",
                "resolved_terms": len(resolved_terms),
                "unresolved_terms": len(unresolved),
                "added_token_count": added_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
