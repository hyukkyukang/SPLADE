import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

HANGUL_PATTERN: re.Pattern[str] = re.compile(
    r"[\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uAC00-\uD7AF]"
)
CJK_PATTERN: re.Pattern[str] = re.compile(
    r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\u31F0-\u31FF"
    r"\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uAC00-\uD7AF]"
)
WEIGHT_FILE_SUFFIXES: tuple[str, ...] = (".safetensors", ".bin")
KEEP_REPORT_FILENAME: str = "kept_token_ids.tsv"
DROP_REPORT_FILENAME: str = "dropped_token_ids.tsv"
SUMMARY_FILENAME: str = "prune_summary.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prune a WordPiece tokenizer/model vocabulary (e.g., ANNA) by removing "
            "tokens that match Unicode/script regex patterns, then rewrite MLM "
            "embedding/head rows to the reduced vocab."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/model/anna_base_hf",
        help="Input HF model directory containing config/model/tokenizer artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output HF model directory for the pruned artifacts. Required unless "
            "--dry-run is set."
        ),
    )
    parser.add_argument(
        "--drop-hangul",
        action="store_true",
        help="Drop tokens containing Hangul script characters.",
    )
    parser.add_argument(
        "--drop-cjk",
        action="store_true",
        help="Drop tokens containing CJK (Han/Hangul/Kana) characters.",
    )
    parser.add_argument(
        "--drop-regex",
        type=str,
        action="append",
        default=[],
        help=(
            "Additional regex for token dropping. Can be passed multiple times. "
            "Regex is applied to token strings."
        ),
    )
    parser.add_argument(
        "--keep-token",
        type=str,
        action="append",
        default=[],
        help=(
            "Token string to force-keep even if it matches a drop pattern. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only analyze and print prune stats without writing output or loading "
            "model weights."
        ),
    )
    parser.add_argument(
        "--report-top-k",
        type=int,
        default=50,
        help="How many sample dropped tokens to print in dry-run/report logs.",
    )
    return parser


def _read_vocab(
    vocab_path: Path,
    *,
    tokenizer_json: dict[str, Any] | None = None,
) -> list[str]:
    if vocab_path.is_file():
        return vocab_path.read_text(encoding="utf-8").splitlines()
    if tokenizer_json is None:
        raise FileNotFoundError(f"Missing vocab file: {vocab_path}")

    model_section: dict[str, Any] = tokenizer_json.get("model", {})
    if model_section.get("type") != "WordPiece":
        raise FileNotFoundError(
            f"Missing vocab file: {vocab_path}. Fallback only supports "
            "tokenizer.json model.type=WordPiece."
        )

    raw_vocab: Any = model_section.get("vocab")
    if not isinstance(raw_vocab, dict):
        raise ValueError(
            "tokenizer.json WordPiece model is missing a valid `vocab` mapping."
        )

    pairs: list[tuple[int, str]] = []
    max_id: int = -1
    token: Any
    raw_id: Any
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing json file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_special_tokens(
    input_dir: Path,
    tokenizer_json: dict[str, Any],
) -> set[str]:
    special_tokens: set[str] = set()

    model_section: dict[str, Any] = tokenizer_json.get("model", {})
    unk_token: Any | None = model_section.get("unk_token")
    if isinstance(unk_token, str) and unk_token:
        special_tokens.add(unk_token)

    for item in tokenizer_json.get("added_tokens", []):
        if not isinstance(item, dict):
            continue
        if bool(item.get("special")) and isinstance(item.get("content"), str):
            special_tokens.add(item["content"])

    special_tokens_map_path: Path = input_dir / "special_tokens_map.json"
    if special_tokens_map_path.is_file():
        special_tokens_map: dict[str, Any] = _load_json(special_tokens_map_path)
        for value in special_tokens_map.values():
            if isinstance(value, str):
                special_tokens.add(value)
            elif isinstance(value, dict):
                content: Any | None = value.get("content")
                if isinstance(content, str):
                    special_tokens.add(content)
    return special_tokens


def _build_drop_patterns(
    *,
    drop_hangul: bool,
    drop_cjk: bool,
    drop_regexes: list[str],
) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    if drop_hangul:
        patterns.append(HANGUL_PATTERN)
    if drop_cjk:
        patterns.append(CJK_PATTERN)
    for expr in drop_regexes:
        patterns.append(re.compile(expr))
    if not patterns:
        raise ValueError(
            "No drop pattern selected. Use --drop-hangul, --drop-cjk, "
            "and/or --drop-regex."
        )
    return patterns


def _compute_keep_drop_ids(
    vocab_tokens: list[str],
    *,
    patterns: list[re.Pattern[str]],
    special_tokens: set[str],
    forced_keep_tokens: set[str],
) -> tuple[list[int], list[int]]:
    token_to_id: dict[str, int] = {token: idx for idx, token in enumerate(vocab_tokens)}
    special_ids: set[int] = {
        token_to_id[token] for token in special_tokens if token in token_to_id
    }
    forced_keep_ids: set[int] = {
        token_to_id[token] for token in forced_keep_tokens if token in token_to_id
    }

    drop_ids: list[int] = []
    for idx, token in enumerate(vocab_tokens):
        if idx in special_ids or idx in forced_keep_ids:
            continue
        if any(pattern.search(token) for pattern in patterns):
            drop_ids.append(idx)

    drop_set: set[int] = set(drop_ids)
    keep_ids: list[int] = [idx for idx in range(len(vocab_tokens)) if idx not in drop_set]
    return keep_ids, drop_ids


def _build_pruned_tokenizer_json(
    tokenizer_json: dict[str, Any],
    *,
    keep_ids: list[int],
    vocab_tokens: list[str],
) -> dict[str, Any]:
    token_to_new_id: dict[str, int] = {
        vocab_tokens[old_id]: new_id for new_id, old_id in enumerate(keep_ids)
    }

    pruned: dict[str, Any] = json.loads(json.dumps(tokenizer_json))
    model_section: dict[str, Any] = pruned.get("model", {})
    if model_section.get("type") != "WordPiece":
        raise ValueError(
            "This script currently supports tokenizer.json model.type=WordPiece."
        )
    model_section["vocab"] = token_to_new_id

    new_added_tokens: list[dict[str, Any]] = []
    for item in pruned.get("added_tokens", []):
        if not isinstance(item, dict):
            continue
        token: Any | None = item.get("content")
        if not isinstance(token, str):
            continue
        if token not in token_to_new_id:
            if bool(item.get("special")):
                raise ValueError(
                    f"Special token was dropped by prune rules: {token!r}"
                )
            continue
        copied: dict[str, Any] = dict(item)
        copied["id"] = int(token_to_new_id[token])
        new_added_tokens.append(copied)
    pruned["added_tokens"] = new_added_tokens

    post_processor: dict[str, Any] = pruned.get("post_processor", {})
    if post_processor.get("type") == "BertProcessing":
        for key in ("sep", "cls"):
            pair: Any | None = post_processor.get(key)
            if isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str):
                token: str = pair[0]
                if token not in token_to_new_id:
                    raise ValueError(
                        f"Post-processor token missing after pruning: {token!r}"
                    )
                post_processor[key] = [token, int(token_to_new_id[token])]

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
    return pruned


def _copy_non_weight_files(input_dir: Path, output_dir: Path) -> None:
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix in WEIGHT_FILE_SUFFIXES:
            continue
        if path.name in {"config.json", "tokenizer.json", "vocab.txt"}:
            continue
        destination: Path = output_dir / path.name
        if not destination.exists():
            shutil.copy2(path, destination)


def _prune_model_rows(
    *,
    input_dir: Path,
    output_dir: Path,
    keep_ids: list[int],
    trust_remote_code: bool,
) -> None:
    model = AutoModelForMaskedLM.from_pretrained(
        str(input_dir),
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


def _save_reports(
    *,
    output_dir: Path,
    vocab_tokens: list[str],
    keep_ids: list[int],
    drop_ids: list[int],
    patterns: list[re.Pattern[str]],
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
        "patterns": [pattern.pattern for pattern in patterns],
        "keep_report": KEEP_REPORT_FILENAME,
        "drop_report": DROP_REPORT_FILENAME,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_pruned_tokenizer_files(
    *,
    output_dir: Path,
    vocab_tokens: list[str],
    keep_ids: list[int],
    pruned_tokenizer_json: dict[str, Any],
) -> None:
    vocab_path: Path = output_dir / "vocab.txt"
    with vocab_path.open("w", encoding="utf-8") as handle:
        for old_id in keep_ids:
            handle.write(vocab_tokens[old_id])
            handle.write("\n")
    (output_dir / "tokenizer.json").write_text(
        json.dumps(pruned_tokenizer_json, ensure_ascii=True),
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

    model = AutoModelForMaskedLM.from_pretrained(
        str(output_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    encoded: dict[str, torch.Tensor] = tokenizer(
        "vocab prune validation text",
        return_tensors="pt",
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

    input_dir: Path = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir: Path | None = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    if not args.dry_run and output_dir is None:
        raise ValueError("--output-dir is required unless --dry-run is set.")

    tokenizer_json: dict[str, Any] = _load_json(input_dir / "tokenizer.json")
    vocab_tokens: list[str] = _read_vocab(
        input_dir / "vocab.txt",
        tokenizer_json=tokenizer_json,
    )
    special_tokens: set[str] = _collect_special_tokens(input_dir, tokenizer_json)
    patterns: list[re.Pattern[str]] = _build_drop_patterns(
        drop_hangul=bool(args.drop_hangul),
        drop_cjk=bool(args.drop_cjk),
        drop_regexes=list(args.drop_regex),
    )
    forced_keep_tokens: set[str] = set(args.keep_token)
    keep_ids, drop_ids = _compute_keep_drop_ids(
        vocab_tokens,
        patterns=patterns,
        special_tokens=special_tokens,
        forced_keep_tokens=forced_keep_tokens,
    )

    old_vocab_size: int = len(vocab_tokens)
    new_vocab_size: int = len(keep_ids)
    dropped_vocab_size: int = len(drop_ids)
    kept_ratio: float = new_vocab_size / max(old_vocab_size, 1)
    dropped_ratio: float = dropped_vocab_size / max(old_vocab_size, 1)
    print(f"input_dir={input_dir}")
    if output_dir is not None:
        print(f"output_dir={output_dir}")
    print(f"old_vocab_size={old_vocab_size}")
    print(f"new_vocab_size={new_vocab_size}")
    print(f"dropped_vocab_size={dropped_vocab_size}")
    print(f"kept_ratio={kept_ratio:.4f}")
    print(f"dropped_ratio={dropped_ratio:.4f}")
    print(f"drop_patterns={[p.pattern for p in patterns]}")
    print(f"special_token_count={len(special_tokens)}")
    if forced_keep_tokens:
        print(f"forced_keep_tokens={sorted(forced_keep_tokens)}")

    report_top_k: int = max(int(args.report_top_k), 0)
    if report_top_k > 0:
        sample_tokens: list[str] = [vocab_tokens[idx] for idx in drop_ids[:report_top_k]]
        print(f"sample_dropped_tokens({len(sample_tokens)}): {sample_tokens}")

    if new_vocab_size <= 0:
        raise ValueError("Pruning removed all tokens; aborting.")
    if args.dry_run:
        return

    assert output_dir is not None
    tmp_dir: Path = _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    _prune_model_rows(
        input_dir=input_dir,
        output_dir=tmp_dir,
        keep_ids=keep_ids,
        trust_remote_code=bool(args.trust_remote_code),
    )
    _copy_non_weight_files(input_dir, tmp_dir)

    pruned_tokenizer_json: dict[str, Any] = _build_pruned_tokenizer_json(
        tokenizer_json,
        keep_ids=keep_ids,
        vocab_tokens=vocab_tokens,
    )
    _write_pruned_tokenizer_files(
        output_dir=tmp_dir,
        vocab_tokens=vocab_tokens,
        keep_ids=keep_ids,
        pruned_tokenizer_json=pruned_tokenizer_json,
    )
    _save_reports(
        output_dir=tmp_dir,
        vocab_tokens=vocab_tokens,
        keep_ids=keep_ids,
        drop_ids=drop_ids,
        patterns=patterns,
    )
    _validate_pruned_artifacts(
        output_dir=tmp_dir,
        expected_vocab_size=new_vocab_size,
        trust_remote_code=bool(args.trust_remote_code),
    )
    _finalize_output_dir(tmp_dir, output_dir)
    print("done")


if __name__ == "__main__":
    main()
