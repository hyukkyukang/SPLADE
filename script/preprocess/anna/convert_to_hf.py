import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from script.preprocess.anna.anna_tokenizer import AnnaTokenizer
from script.preprocess.anna.conversion_utils import load_anna_masked_lm_model
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.script_setup import configure_script_environment

logger: logging.Logger = get_logger("script.preprocess.anna.convert_to_hf", __file__)

configure_script_environment(
    load_env=False,
    set_tokenizers_parallelism=True,
    set_matmul_precision=False,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ANNA checkpoint directory into a standard Hugging Face "
            "MaskedLM model directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/model/anna",
        help="Source ANNA directory containing config/vocab/checkpoint files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Destination directory for Hugging Face save_pretrained artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Save model as safetensors (model.safetensors) instead of .bin.",
    )
    parser.add_argument(
        "--untie-word-embeddings",
        action="store_true",
        help="Export with tie_word_embeddings=False (default is tied).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-export validation load checks.",
    )
    parser.add_argument(
        "--skip-tokenizer-equivalence",
        action="store_true",
        help="Skip tokenizer equivalence check (source ANNA vs exported HF).",
    )
    return parser


def _prepare_tmp_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Pass --overwrite to replace it."
        )
    tmp_dir: Path = output_dir.with_name(f"{output_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def _finalize_output_dir(tmp_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    tmp_dir.replace(output_dir)


def _repo_anna_tokenizer_path() -> Path:
    repo_root: Path = Path(__file__).resolve().parent.parent.parent.parent
    path: Path = repo_root / "script" / "preprocess" / "anna" / "anna_tokenizer.py"
    if not path.is_file():
        raise FileNotFoundError(f"ANNA tokenizer module not found: {path}")
    return path


def _export_hf_tokenizer_artifacts(
    input_dir: Path,
    output_dir: Path,
) -> None:
    vocab_path: Path = input_dir / "vocab.txt"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    tokenizer = AnnaTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_vocabulary(str(output_dir))
    tokenizer_config: dict[str, object] = {
        "auto_map": {
            "AutoTokenizer": [
                "anna_tokenizer.AnnaTokenizer",
                "anna_tokenizer.AnnaTokenizerFast",
            ]
        },
        "tokenizer_class": "AnnaTokenizer",
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "model_max_length": 512,
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2), encoding="utf-8"
    )
    special_tokens_map: dict[str, str] = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
    }
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2), encoding="utf-8"
    )
    shutil.copy2(_repo_anna_tokenizer_path(), output_dir / "anna_tokenizer.py")


def _validate_exported_hf_dir(output_dir: Path) -> None:
    _ = AutoConfig.from_pretrained(str(output_dir), local_files_only=True)
    slow_tokenizer_cls = get_class_from_dynamic_module(
        "anna_tokenizer.AnnaTokenizer",
        str(output_dir),
        local_files_only=True,
    )
    tokenizer_slow = slow_tokenizer_cls.from_pretrained(
        str(output_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer_fast = AutoTokenizer.from_pretrained(
        str(output_dir),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=True,
    )
    if not bool(tokenizer_fast.is_fast):
        raise RuntimeError(
            "use_fast=True did not resolve to a fast ANNA tokenizer backend."
        )
    model = AutoModelForMaskedLM.from_pretrained(
        str(output_dir),
        local_files_only=True,
    )
    encoded = tokenizer_fast(
        ["anna conversion validation"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16,
    )
    input_keys: set[str] = {"input_ids", "attention_mask", "token_type_ids"}
    model_inputs = {key: value for key, value in encoded.items() if key in input_keys}
    with torch.no_grad():
        outputs = model(**model_inputs)
    logits = outputs.logits
    if logits.ndim != 3:
        raise RuntimeError(
            "Validation failed: expected MLM logits with shape "
            f"[batch, seq, vocab], got ndim={logits.ndim}."
        )
    encoded_slow = tokenizer_slow(
        ["anna conversion validation"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16,
    )
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if key in encoded and key in encoded_slow and not torch.equal(
            encoded[key], encoded_slow[key]
        ):
            raise RuntimeError(
                "Fast and slow tokenizer encodings diverge during conversion "
                f"validation for key: {key}"
            )


_EDGE_CASE_TEXTS: tuple[str, ...] = (
    "",
    " ",
    "anna conversion validation",
    "Hello, world!",
    "lower UPPER 123",
    "punctuation: period. comma, semi; colon:",
    "CJK 中文 日本語",
    "accents café naïve",
)


def _validate_tokenizer_equivalence(
    input_dir: Path,
    output_dir: Path,
    *,
    skip_if_no_tf: bool = True,
) -> None:
    from script.preprocess.anna.conversion_utils import load_anna_tokenizer

    try:
        source_tokenizer = load_anna_tokenizer(str(input_dir), do_lower_case=True)
    except ModuleNotFoundError as e:
        if skip_if_no_tf and (e.name == "tensorflow" or "tensorflow" in str(e).lower()):
            log_if_rank_zero(
                logger,
                "Skipping tokenizer equivalence (TensorFlow required for source ANNA).",
            )
            return
        raise
    slow_tokenizer_cls = get_class_from_dynamic_module(
        "anna_tokenizer.AnnaTokenizer",
        str(output_dir),
        local_files_only=True,
    )
    hf_tokenizer_slow = slow_tokenizer_cls.from_pretrained(
        str(output_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    hf_tokenizer_fast = AutoTokenizer.from_pretrained(
        str(output_dir),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=True,
    )
    if not bool(hf_tokenizer_fast.is_fast):
        raise RuntimeError(
            "use_fast=True did not resolve to a fast ANNA tokenizer backend."
        )
    for text in _EDGE_CASE_TEXTS:
        source_tokens: list[str] = source_tokenizer.tokenize(text)
        slow_tokens: list[str] = hf_tokenizer_slow.tokenize(text)
        fast_tokens: list[str] = hf_tokenizer_fast.tokenize(text)
        if source_tokens != slow_tokens:
            raise RuntimeError(
                f"Tokenizer mismatch for text {text!r}: "
                f"source={source_tokens!r} hf_slow={slow_tokens!r}"
            )
        if source_tokens != fast_tokens:
            raise RuntimeError(
                f"Tokenizer mismatch for text {text!r}: "
                f"source={source_tokens!r} hf_fast={fast_tokens!r}"
            )
        source_ids: list[int] = source_tokenizer.convert_tokens_to_ids(source_tokens)
        slow_ids: list[int] = hf_tokenizer_slow.convert_tokens_to_ids(slow_tokens)
        fast_ids: list[int] = hf_tokenizer_fast.convert_tokens_to_ids(fast_tokens)
        if source_ids != slow_ids:
            raise RuntimeError(
                f"Tokenizer ID mismatch for text {text!r}: "
                f"source_ids={source_ids!r} hf_slow_ids={slow_ids!r}"
            )
        if source_ids != fast_ids:
            raise RuntimeError(
                f"Tokenizer ID mismatch for text {text!r}: "
                f"source_ids={source_ids!r} hf_fast_ids={fast_ids!r}"
            )
    log_if_rank_zero(logger, "Tokenizer equivalence check passed.")


def _save_anna_as_hf(
    *,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
    safe_serialization: bool,
    tie_word_embeddings: bool,
    skip_validation: bool,
    skip_tokenizer_equivalence: bool,
) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    log_if_rank_zero(
        logger,
        f"Loading ANNA model from {input_dir} "
        f"(tie_word_embeddings={tie_word_embeddings}).",
    )
    model = load_anna_masked_lm_model(
        str(input_dir),
        map_location="cpu",
        tie_word_embeddings=tie_word_embeddings,
    )

    tmp_output_dir: Path = _prepare_tmp_output_dir(output_dir, overwrite=overwrite)
    model.save_pretrained(
        str(tmp_output_dir),
        safe_serialization=bool(safe_serialization),
    )
    _export_hf_tokenizer_artifacts(input_dir, tmp_output_dir)

    if not skip_tokenizer_equivalence:
        log_if_rank_zero(logger, "Running tokenizer equivalence check...")
        _validate_tokenizer_equivalence(input_dir, tmp_output_dir)

    if not skip_validation:
        log_if_rank_zero(logger, "Validating exported Hugging Face artifacts...")
        _validate_exported_hf_dir(tmp_output_dir)

    _finalize_output_dir(tmp_output_dir, output_dir, overwrite=overwrite)

    artifact_names: list[str] = sorted(
        path.name for path in output_dir.iterdir() if path.is_file()
    )
    log_if_rank_zero(logger, f"Conversion complete. Output: {output_dir}")
    log_if_rank_zero(logger, f"Exported files: {artifact_names}")


def main() -> None:
    parser: argparse.ArgumentParser = _build_parser()
    args = parser.parse_args()
    input_dir: Path = Path(args.input_dir)
    output_dir: Path = Path(args.output_dir)
    tie_word_embeddings: bool = not bool(args.untie_word_embeddings)
    _save_anna_as_hf(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=bool(args.overwrite),
        safe_serialization=bool(args.safe_serialization),
        tie_word_embeddings=tie_word_embeddings,
        skip_validation=bool(args.skip_validation),
        skip_tokenizer_equivalence=bool(args.skip_tokenizer_equivalence),
    )


if __name__ == "__main__":
    main()
