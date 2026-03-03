import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

BYTE_FALLBACK_PATTERN: re.Pattern[str] = re.compile(r"^<0x[0-9A-F]{2}>$")
DEFAULT_KEEP_REGEX: str = r"^[\x00-\x7F▁]+$"
TEXT_SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<eos>", "<bos>", "<unk>", "<mask>")
IMAGE_SPECIAL_TOKENS: tuple[str, ...] = (
    "<start_of_image>",
    "<end_of_image>",
    "<image_soft_token>",
)
WEIGHT_FILE_SUFFIXES: tuple[str, ...] = (".safetensors", ".bin")
KEEP_REPORT_FILENAME: str = "kept_token_ids.tsv"
DROP_REPORT_FILENAME: str = "dropped_token_ids.tsv"
SUMMARY_FILENAME: str = "prune_summary.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prune google/embeddinggemma-300m vocabulary for English-centric text "
            "usage and rewrite CausalLM embedding/output rows to the reduced vocab."
        )
    )
    parser.add_argument(
        "--input-model",
        type=str,
        default="google/embeddinggemma-300m",
        help="Input HF model name or local model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/model/trained_embeddinggemma_300m_pruned",
        help="Output directory for the pruned model/tokenizer artifacts.",
    )
    parser.add_argument(
        "--keep-regex",
        type=str,
        action="append",
        default=[],
        help=(
            "Regex for token keep rules. If omitted, defaults to ASCII + ▁ "
            f"({DEFAULT_KEEP_REGEX}). Can be repeated."
        ),
    )
    parser.add_argument(
        "--keep-token",
        type=str,
        action="append",
        default=[],
        help="Token string to force-keep. Can be repeated.",
    )
    parser.add_argument(
        "--drop-token",
        type=str,
        action="append",
        default=[],
        help="Token string to force-drop. Can be repeated.",
    )
    parser.add_argument(
        "--drop-special-token",
        type=str,
        action="append",
        default=[],
        help="Special token string to drop. Can be repeated.",
    )
    parser.add_argument(
        "--keep-image-special-tokens",
        action="store_true",
        help=(
            "Keep image-related special tokens (<start_of_image>, <end_of_image>, "
            "<image_soft_token>). By default they are dropped."
        ),
    )
    parser.add_argument(
        "--no-byte-fallback-keep",
        action="store_true",
        help=(
            "Do not force-keep byte fallback tokens (<0x00>..<0xFF>). "
            "By default these are retained."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print prune statistics without writing model artifacts.",
    )
    parser.add_argument(
        "--report-top-k",
        type=int,
        default=50,
        help="How many sample dropped tokens to print.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing json file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_bpe_vocab_tokens(tokenizer_json: dict[str, Any]) -> list[str]:
    model_section: dict[str, Any] = tokenizer_json.get("model", {})
    if model_section.get("type") != "BPE":
        raise ValueError("This script supports tokenizer.json model.type=BPE only.")

    raw_vocab: Any = model_section.get("vocab")
    if not isinstance(raw_vocab, dict):
        raise ValueError("tokenizer.json BPE model is missing a valid `vocab` mapping.")

    pairs: list[tuple[int, str]] = []
    token: Any
    raw_id: Any
    max_id: int = -1
    for token, raw_id in raw_vocab.items():
        if not isinstance(token, str):
            raise ValueError("Encountered non-string token in tokenizer.json vocab.")
        token_id: int = int(raw_id)
        if token_id < 0:
            raise ValueError("Encountered negative token id in tokenizer.json vocab.")
        pairs.append((token_id, token))
        if token_id > max_id:
            max_id = token_id

    expected_size: int = len(raw_vocab)
    if max_id + 1 != expected_size:
        raise ValueError(
            "tokenizer.json vocab ids are not contiguous [0..N-1]; cannot derive "
            "ordered vocab list safely."
        )

    ordered_tokens: list[str | None] = [None] * expected_size
    token_id: int
    for token_id, token in pairs:
        if ordered_tokens[token_id] is not None:
            raise ValueError("Encountered duplicate token id in tokenizer.json vocab.")
        ordered_tokens[token_id] = token
    if any(token is None for token in ordered_tokens):
        raise ValueError("tokenizer.json vocab has missing token ids.")
    return [token for token in ordered_tokens if token is not None]


def _collect_special_tokens(
    tokenizer: Any, tokenizer_config: dict[str, Any]
) -> set[str]:
    special_tokens: set[str] = set()

    for token in tokenizer.all_special_tokens:
        if isinstance(token, str) and token:
            special_tokens.add(token)

    key: str
    value: Any
    for key, value in tokenizer_config.items():
        if not key.endswith("_token"):
            continue
        if isinstance(value, str) and value:
            special_tokens.add(value)

    model_specials: Any = tokenizer_config.get("model_specific_special_tokens")
    if isinstance(model_specials, dict):
        for value in model_specials.values():
            if isinstance(value, str) and value:
                special_tokens.add(value)
    return special_tokens


def _build_keep_patterns(raw_patterns: list[str]) -> list[re.Pattern[str]]:
    patterns: list[str] = list(raw_patterns) if raw_patterns else [DEFAULT_KEEP_REGEX]
    compiled: list[re.Pattern[str]] = []
    pattern: str
    for pattern in patterns:
        compiled.append(re.compile(pattern))
    return compiled


def _compute_keep_drop_ids(
    vocab_tokens: list[str],
    *,
    keep_patterns: list[re.Pattern[str]],
    keep_special_tokens: set[str],
    force_keep_tokens: set[str],
    force_drop_tokens: set[str],
    keep_byte_fallback_tokens: bool,
) -> tuple[list[int], list[int]]:
    token_to_id: dict[str, int] = {token: idx for idx, token in enumerate(vocab_tokens)}

    required_keep_ids: set[int] = set()
    token: str
    for token in TEXT_SPECIAL_TOKENS:
        if token in token_to_id:
            required_keep_ids.add(token_to_id[token])

    special_keep_ids: set[int] = {
        token_to_id[token] for token in keep_special_tokens if token in token_to_id
    }
    force_keep_ids: set[int] = {
        token_to_id[token] for token in force_keep_tokens if token in token_to_id
    }
    force_drop_ids: set[int] = {
        token_to_id[token] for token in force_drop_tokens if token in token_to_id
    }

    keep_ids: list[int] = []
    drop_ids: list[int] = []
    idx: int
    for idx, token in enumerate(vocab_tokens):
        if idx in required_keep_ids:
            keep_ids.append(idx)
            continue
        if idx in force_drop_ids:
            drop_ids.append(idx)
            continue
        if idx in special_keep_ids or idx in force_keep_ids:
            keep_ids.append(idx)
            continue
        if keep_byte_fallback_tokens and BYTE_FALLBACK_PATTERN.match(token):
            keep_ids.append(idx)
            continue
        if any(pattern.search(token) for pattern in keep_patterns):
            keep_ids.append(idx)
            continue
        drop_ids.append(idx)

    return keep_ids, drop_ids


def _normalize_merge_item(item: Any) -> tuple[str, str, str]:
    if isinstance(item, list) and len(item) == 2:
        left: Any = item[0]
        right: Any = item[1]
        if isinstance(left, str) and isinstance(right, str):
            return left, right, "list"
    if isinstance(item, str):
        # BPE string merge line format: "<left> <right>"
        parts: list[str] = item.split(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1], "str"
    raise ValueError(f"Unsupported BPE merge item format: {item!r}")


def _build_pruned_tokenizer_json(
    tokenizer_json: dict[str, Any],
    *,
    keep_ids: list[int],
    vocab_tokens: list[str],
) -> tuple[dict[str, Any], int]:
    token_to_new_id: dict[str, int] = {
        vocab_tokens[old_id]: new_id for new_id, old_id in enumerate(keep_ids)
    }
    keep_token_set: set[str] = set(token_to_new_id)

    pruned: dict[str, Any] = json.loads(json.dumps(tokenizer_json))
    model_section: dict[str, Any] = pruned.get("model", {})
    if model_section.get("type") != "BPE":
        raise ValueError(
            "This script currently supports tokenizer.json model.type=BPE only."
        )
    model_section["vocab"] = token_to_new_id

    raw_merges: Any = model_section.get("merges", [])
    if not isinstance(raw_merges, list):
        raise ValueError("tokenizer.json model.merges must be a list.")

    new_merges: list[Any] = []
    dropped_merges: int = 0
    merge_item: Any
    for merge_item in raw_merges:
        left, right, fmt = _normalize_merge_item(merge_item)
        merged_token: str = left + right
        if (
            left not in keep_token_set
            or right not in keep_token_set
            or merged_token not in keep_token_set
        ):
            dropped_merges += 1
            continue
        if fmt == "list":
            new_merges.append([left, right])
        else:
            new_merges.append(f"{left} {right}")
    model_section["merges"] = new_merges

    new_added_tokens: list[dict[str, Any]] = []
    added_item: Any
    for added_item in pruned.get("added_tokens", []):
        if not isinstance(added_item, dict):
            continue
        token: Any = added_item.get("content")
        if not isinstance(token, str):
            continue
        new_id: int | None = token_to_new_id.get(token)
        if new_id is None:
            continue
        copied: dict[str, Any] = dict(added_item)
        copied["id"] = int(new_id)
        new_added_tokens.append(copied)
    pruned["added_tokens"] = new_added_tokens

    padding: dict[str, Any] | None = pruned.get("padding")
    if isinstance(padding, dict):
        pad_token: Any | None = padding.get("pad_token")
        if isinstance(pad_token, str) and pad_token in token_to_new_id:
            padding["pad_id"] = int(token_to_new_id[pad_token])

    truncation: dict[str, Any] | None = pruned.get("truncation")
    if isinstance(truncation, dict):
        stride_value: Any | None = truncation.get("stride")
        if isinstance(stride_value, bool):
            truncation["stride"] = int(stride_value)

    return pruned, dropped_merges


def _rewrite_tokenizer_config(
    *,
    tokenizer_config_path: Path,
    kept_tokens: set[str],
) -> None:
    if not tokenizer_config_path.is_file():
        return
    payload: dict[str, Any] = _load_json(tokenizer_config_path)
    changed: bool = False

    key: str
    value: Any
    for key, value in list(payload.items()):
        if not key.endswith("_token"):
            continue
        if isinstance(value, str) and value not in kept_tokens:
            payload[key] = None
            changed = True

    additional_specials: Any = payload.get("additional_special_tokens")
    if isinstance(additional_specials, list):
        filtered: list[Any] = [
            token
            for token in additional_specials
            if not isinstance(token, str) or token in kept_tokens
        ]
        if filtered != additional_specials:
            payload["additional_special_tokens"] = filtered
            changed = True

    model_specials: Any = payload.get("model_specific_special_tokens")
    if isinstance(model_specials, dict):
        filtered_model_specials: dict[str, Any] = {}
        for key, value in model_specials.items():
            if isinstance(value, str) and value not in kept_tokens:
                changed = True
                continue
            filtered_model_specials[str(key)] = value
        if filtered_model_specials != model_specials:
            payload["model_specific_special_tokens"] = filtered_model_specials
            changed = True

    if changed:
        _write_json(tokenizer_config_path, payload)


def _rewrite_special_tokens_map(
    *,
    special_tokens_map_path: Path,
    kept_tokens: set[str],
) -> None:
    if not special_tokens_map_path.is_file():
        return
    payload: dict[str, Any] = _load_json(special_tokens_map_path)
    changed: bool = False
    key: str
    value: Any
    for key, value in list(payload.items()):
        if isinstance(value, str):
            if value not in kept_tokens:
                payload.pop(key, None)
                changed = True
            continue
        if isinstance(value, dict):
            content: Any = value.get("content")
            if isinstance(content, str) and content not in kept_tokens:
                payload.pop(key, None)
                changed = True
    if changed:
        _write_json(special_tokens_map_path, payload)


def _prune_model_rows(
    *,
    input_model: str,
    output_dir: Path,
    keep_ids: list[int],
    trust_remote_code: bool,
) -> None:
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        input_model,
        trust_remote_code=trust_remote_code,
    )
    keep_index: torch.Tensor = torch.tensor(keep_ids, dtype=torch.long)

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise ValueError("Model does not expose input embeddings.")
    old_input_weight: torch.Tensor = input_embeddings.weight.detach().clone().cpu()

    output_embeddings = model.get_output_embeddings()
    old_output_weight: torch.Tensor | None = None
    old_output_bias: torch.Tensor | None = None
    if output_embeddings is not None:
        old_output_weight = output_embeddings.weight.detach().clone().cpu()
        bias_attr: Any | None = getattr(output_embeddings, "bias", None)
        if isinstance(bias_attr, torch.Tensor):
            old_output_bias = bias_attr.detach().clone().cpu()

    new_vocab_size: int = len(keep_ids)
    model.resize_token_embeddings(new_vocab_size)

    new_input_embeddings = model.get_input_embeddings()
    if new_input_embeddings is None:
        raise ValueError("Resized model does not expose input embeddings.")
    new_input_weight: torch.Tensor = new_input_embeddings.weight
    new_input_weight.data.copy_(
        old_input_weight.index_select(0, keep_index).to(
            device=new_input_weight.device,
            dtype=new_input_weight.dtype,
        )
    )

    new_output_embeddings = model.get_output_embeddings()
    if (
        new_output_embeddings is not None
        and old_output_weight is not None
        and int(new_output_embeddings.weight.shape[0]) == new_vocab_size
    ):
        new_output_embeddings.weight.data.copy_(
            old_output_weight.index_select(0, keep_index).to(
                device=new_output_embeddings.weight.device,
                dtype=new_output_embeddings.weight.dtype,
            )
        )
        new_bias: Any | None = getattr(new_output_embeddings, "bias", None)
        if isinstance(new_bias, torch.Tensor) and old_output_bias is not None:
            if int(old_output_bias.shape[0]) >= int(keep_index.max().item() + 1):
                new_bias.data.copy_(
                    old_output_bias.index_select(0, keep_index).to(
                        device=new_bias.device,
                        dtype=new_bias.dtype,
                    )
                )

    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.config.vocab_size = new_vocab_size
    model.save_pretrained(str(output_dir))


def _copy_non_weight_files(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.is_dir():
        return
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix in WEIGHT_FILE_SUFFIXES:
            continue
        if path.name in {"tokenizer.json", "tokenizer_config.json"}:
            continue
        destination: Path = output_dir / path.name
        if not destination.exists():
            shutil.copy2(path, destination)


def _save_reports(
    *,
    output_dir: Path,
    vocab_tokens: list[str],
    keep_ids: list[int],
    drop_ids: list[int],
    keep_patterns: list[re.Pattern[str]],
    dropped_special_tokens: list[str],
    dropped_merges: int,
) -> None:
    kept_path: Path = output_dir / KEEP_REPORT_FILENAME
    dropped_path: Path = output_dir / DROP_REPORT_FILENAME

    with kept_path.open("w", encoding="utf-8") as handle:
        handle.write("old_id\tnew_id\ttoken\n")
        for new_id, old_id in enumerate(keep_ids):
            token: str = vocab_tokens[old_id]
            handle.write(f"{old_id}\t{new_id}\t{token}\n")

    with dropped_path.open("w", encoding="utf-8") as handle:
        handle.write("old_id\ttoken\n")
        for old_id in drop_ids:
            token = vocab_tokens[old_id]
            handle.write(f"{old_id}\t{token}\n")

    summary: dict[str, Any] = {
        "old_vocab_size": len(vocab_tokens),
        "new_vocab_size": len(keep_ids),
        "dropped_vocab_size": len(drop_ids),
        "kept_ratio": len(keep_ids) / max(len(vocab_tokens), 1),
        "dropped_ratio": len(drop_ids) / max(len(vocab_tokens), 1),
        "keep_patterns": [pattern.pattern for pattern in keep_patterns],
        "dropped_special_tokens": dropped_special_tokens,
        "dropped_merges": int(dropped_merges),
        "keep_report": KEEP_REPORT_FILENAME,
        "drop_report": DROP_REPORT_FILENAME,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _validate_pruned_artifacts(
    *,
    output_dir: Path,
    expected_vocab_size: int,
    trust_remote_code: bool,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        str(output_dir),
        use_fast=True,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    if len(tokenizer) != expected_vocab_size:
        raise RuntimeError(
            "Tokenizer vocab size mismatch after pruning: "
            f"expected={expected_vocab_size}, actual={len(tokenizer)}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(output_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    encoded: dict[str, torch.Tensor] = tokenizer(
        "vocab prune validation text",
        return_tensors="pt",
    )
    max_input_id: int = int(encoded["input_ids"].max().item())
    if max_input_id >= expected_vocab_size:
        raise RuntimeError(
            "Tokenizer produced token id outside model vocab range after pruning: "
            f"max_input_id={max_input_id}, vocab_size={expected_vocab_size}"
        )
    with torch.no_grad():
        outputs = model(**encoded)
    logits: torch.Tensor = outputs.logits
    if int(logits.shape[-1]) != expected_vocab_size:
        raise RuntimeError(
            "Model logits vocab size mismatch after pruning: "
            f"expected={expected_vocab_size}, actual={int(logits.shape[-1])}"
        )


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    tmp_dir: Path = output_dir.with_name(f"{output_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def _finalize_output_dir(tmp_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    tmp_dir.replace(output_dir)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_model: str = str(args.input_model)
    output_dir: Path = Path(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        input_model,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if not bool(tokenizer.is_fast):
        raise ValueError(
            "A fast tokenizer is required for pruning. "
            f"Loaded tokenizer is slow: {tokenizer.__class__.__name__}"
        )

    with tempfile.TemporaryDirectory(prefix="embeddinggemma_tokenizer_") as tmp:
        source_tokenizer_dir: Path = Path(tmp)
        tokenizer.save_pretrained(str(source_tokenizer_dir))
        tokenizer_json: dict[str, Any] = _load_json(source_tokenizer_dir / "tokenizer.json")
        tokenizer_config: dict[str, Any] = _load_json(
            source_tokenizer_dir / "tokenizer_config.json"
        )

        vocab_tokens: list[str] = _read_bpe_vocab_tokens(tokenizer_json)
        special_tokens: set[str] = _collect_special_tokens(tokenizer, tokenizer_config)
        dropped_special_tokens: set[str] = set(args.drop_special_token)
        if not bool(args.keep_image_special_tokens):
            dropped_special_tokens.update(IMAGE_SPECIAL_TOKENS)
        keep_special_tokens: set[str] = {
            token for token in special_tokens if token not in dropped_special_tokens
        }

        keep_patterns: list[re.Pattern[str]] = _build_keep_patterns(
            list(args.keep_regex)
        )
        force_keep_tokens: set[str] = set(args.keep_token)
        force_drop_tokens: set[str] = set(args.drop_token)
        # Explicit special-token drops must override regex-based keep rules.
        force_drop_tokens.update(dropped_special_tokens)
        force_drop_tokens.difference_update(force_keep_tokens)
        keep_ids, drop_ids = _compute_keep_drop_ids(
            vocab_tokens,
            keep_patterns=keep_patterns,
            keep_special_tokens=keep_special_tokens,
            force_keep_tokens=force_keep_tokens,
            force_drop_tokens=force_drop_tokens,
            keep_byte_fallback_tokens=not bool(args.no_byte_fallback_keep),
        )

        old_vocab_size: int = len(vocab_tokens)
        new_vocab_size: int = len(keep_ids)
        dropped_vocab_size: int = len(drop_ids)
        kept_ratio: float = new_vocab_size / max(old_vocab_size, 1)
        dropped_ratio: float = dropped_vocab_size / max(old_vocab_size, 1)

        print(f"input_model={input_model}")
        print(f"output_dir={output_dir}")
        print(f"old_vocab_size={old_vocab_size}")
        print(f"new_vocab_size={new_vocab_size}")
        print(f"dropped_vocab_size={dropped_vocab_size}")
        print(f"kept_ratio={kept_ratio:.4f}")
        print(f"dropped_ratio={dropped_ratio:.4f}")
        print(f"keep_patterns={[pattern.pattern for pattern in keep_patterns]}")
        print(f"special_token_count={len(special_tokens)}")
        print(f"kept_special_token_count={len(keep_special_tokens)}")
        if dropped_special_tokens:
            print(f"dropped_special_tokens={sorted(dropped_special_tokens)}")
        if force_keep_tokens:
            print(f"forced_keep_tokens={sorted(force_keep_tokens)}")
        if force_drop_tokens:
            print(f"forced_drop_tokens={sorted(force_drop_tokens)}")

        report_top_k: int = max(int(args.report_top_k), 0)
        if report_top_k > 0:
            sample_tokens: list[str] = [
                vocab_tokens[idx] for idx in drop_ids[:report_top_k]
            ]
            print(f"sample_dropped_tokens({len(sample_tokens)}): {sample_tokens}")

        if new_vocab_size <= 0:
            raise ValueError("Pruning removed all tokens; aborting.")
        if args.dry_run:
            return

        tmp_dir: Path = _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
        try:
            _prune_model_rows(
                input_model=input_model,
                output_dir=tmp_dir,
                keep_ids=keep_ids,
                trust_remote_code=bool(args.trust_remote_code),
            )
            tokenizer.save_pretrained(str(tmp_dir))
            _copy_non_weight_files(source_tokenizer_dir, tmp_dir)

            pruned_tokenizer_json, dropped_merges = _build_pruned_tokenizer_json(
                tokenizer_json,
                keep_ids=keep_ids,
                vocab_tokens=vocab_tokens,
            )
            (tmp_dir / "tokenizer.json").write_text(
                json.dumps(pruned_tokenizer_json, ensure_ascii=False),
                encoding="utf-8",
            )

            kept_tokens: set[str] = {
                vocab_tokens[old_id] for old_id in keep_ids
            }
            _rewrite_tokenizer_config(
                tokenizer_config_path=tmp_dir / "tokenizer_config.json",
                kept_tokens=kept_tokens,
            )
            _rewrite_special_tokens_map(
                special_tokens_map_path=tmp_dir / "special_tokens_map.json",
                kept_tokens=kept_tokens,
            )

            _save_reports(
                output_dir=tmp_dir,
                vocab_tokens=vocab_tokens,
                keep_ids=keep_ids,
                drop_ids=drop_ids,
                keep_patterns=keep_patterns,
                dropped_special_tokens=sorted(dropped_special_tokens),
                dropped_merges=dropped_merges,
            )
            _validate_pruned_artifacts(
                output_dir=tmp_dir,
                expected_vocab_size=new_vocab_size,
                trust_remote_code=bool(args.trust_remote_code),
            )
            _finalize_output_dir(tmp_dir, output_dir)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    print("done")


if __name__ == "__main__":
    main()
