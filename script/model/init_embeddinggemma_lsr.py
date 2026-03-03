import argparse
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.prototype.embeddinggemma_lsr.model import (
    EmbeddingGemmaLSRModel,
    apply_projection_initialization,
    build_semantic_projection_initialization,
    discover_fragmented_terms,
    resolve_boundary_token_ids,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize EmbeddingGemma-LSR projection head using semantic term vectors."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional OmegaConf YAML.")
    parser.add_argument("--base-model", type=str, default="google/embeddinggemma-300m")
    parser.add_argument("--vocab-artifact-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--fragment-threshold", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _default_values() -> dict[str, Any]:
    parser: argparse.ArgumentParser = _build_parser()
    defaults: dict[str, Any] = {}
    action: argparse.Action
    for action in parser._actions:
        if action.dest in {None, "help"}:
            continue
        defaults[str(action.dest)] = action.default
    return defaults


def _apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    cfg = OmegaConf.load(args.config)
    payload: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    defaults: dict[str, Any] = _default_values()
    for key, value in payload.items():
        if not hasattr(args, key):
            continue
        if key in defaults and getattr(args, key) == defaults[key]:
            setattr(args, key, value)
    return args


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    key: str = str(dtype_name).lower()
    if key == "float16":
        return torch.float16
    if key == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _resolve_device(device_value: str) -> torch.device:
    text: str = str(device_value).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _load_vocab_artifacts(vocab_artifact_dir: Path) -> tuple[list[str], dict[str, int]]:
    vocab_path: Path = vocab_artifact_dir / "v_target.txt"
    df_map_path: Path = vocab_artifact_dir / "df_map.json"

    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing file: {vocab_path}")
    if not df_map_path.is_file():
        raise FileNotFoundError(f"Missing file: {df_map_path}")

    target_vocab: list[str] = [
        line.strip()
        for line in vocab_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    df_map_raw: dict[str, Any] = json.loads(df_map_path.read_text(encoding="utf-8"))
    df_map: dict[str, int] = {str(k): int(v) for k, v in df_map_raw.items()}
    return target_vocab, df_map


def _validate_initialization(
    *,
    weights: torch.Tensor,
    biases: torch.Tensor,
) -> dict[str, Any]:
    finite_weights: bool = bool(torch.isfinite(weights).all().item())
    finite_biases: bool = bool(torch.isfinite(biases).all().item())
    if not finite_weights:
        raise RuntimeError("Semantic initialization produced non-finite projection weights.")
    if not finite_biases:
        raise RuntimeError("Semantic initialization produced non-finite projection biases.")

    norms: torch.Tensor = torch.linalg.vector_norm(weights.float(), ord=2, dim=1)
    return {
        "weight_norm_min": float(norms.min().item()),
        "weight_norm_max": float(norms.max().item()),
        "weight_norm_mean": float(norms.mean().item()),
        "bias_min": float(biases.min().item()),
        "bias_max": float(biases.max().item()),
        "bias_mean": float(biases.mean().item()),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args = _apply_config_overrides(args)

    vocab_artifact_dir: Path = Path(args.vocab_artifact_dir)
    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_vocab, df_map = _load_vocab_artifacts(vocab_artifact_dir)
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

    fragmented_terms, tokenization_report = discover_fragmented_terms(
        tokenizer,
        target_vocab,
        threshold=int(args.fragment_threshold),
    )
    added_count: int = 0
    if fragmented_terms:
        added_count = int(tokenizer.add_tokens(fragmented_terms))

    boundary_token_ids: list[int] = resolve_boundary_token_ids(tokenizer)
    model: EmbeddingGemmaLSRModel = EmbeddingGemmaLSRModel.from_backbone_name(
        backbone_name_or_path=args.base_model,
        target_vocab=target_vocab,
        boundary_token_ids=boundary_token_ids,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    model.to(device)

    if added_count > 0:
        model.backbone.resize_token_embeddings(len(tokenizer))

    weights, biases, metadata = build_semantic_projection_initialization(
        model=model,
        tokenizer=tokenizer,
        target_vocab=target_vocab,
        df_map=df_map,
        alpha=float(args.alpha),
        device=device,
    )
    apply_projection_initialization(model, weights=weights, biases=biases)

    added_vocab: dict[str, int] = tokenizer.get_added_vocab()
    added_terms_set: set[str] = set(added_vocab.keys())
    term_entry: dict[str, Any]
    for term_entry in metadata:
        term: str = str(term_entry["term"])
        term_entry["used_added_token"] = bool(term in added_terms_set)

    init_stats: dict[str, Any] = _validate_initialization(weights=weights, biases=biases)
    init_summary: dict[str, Any] = {
        "base_model": args.base_model,
        "target_vocab_size": len(target_vocab),
        "fragment_threshold": int(args.fragment_threshold),
        "fragmented_term_count": len(fragmented_terms),
        "added_token_count": added_count,
        "alpha": float(args.alpha),
        "dtype": str(dtype),
        "device": str(device),
        **init_stats,
    }

    model.save_pretrained(
        output_dir,
        tokenizer=tokenizer,
        extra_metadata=init_summary,
    )

    (output_dir / "target_vocab.txt").write_text(
        "\n".join(target_vocab) + "\n",
        encoding="utf-8",
    )
    (output_dir / "df_map.json").write_text(
        json.dumps(df_map, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "tokenization_report.json").write_text(
        json.dumps(tokenization_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "init_metadata.json").write_text(
        json.dumps(
            {
                "summary": init_summary,
                "terms": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved initialized EmbeddingGemma-LSR artifacts to {output_dir}")
    print(f"Added tokens for fragmented terms: {added_count}")


if __name__ == "__main__":
    main()
