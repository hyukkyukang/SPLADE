import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertTokenizerFast, PreTrainedTokenizerBase

from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.script_setup import configure_script_environment, normalize_optional_str

logger: logging.Logger = get_logger(
    "script.experiment.analyze_msmarco_query_token_lengths", __file__
)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=False,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


@dataclass(frozen=True)
class TokenLengthStats:
    item_count: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize MSMARCO queries/documents with a BERT tokenizer and report "
            "average/min/max token counts."
        )
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=("query", "document", "both"),
        default="query",
        help="Which MSMARCO text set to analyze.",
    )
    parser.add_argument(
        "--hf-name",
        type=str,
        default="sentence-transformers/msmarco",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--query-subset",
        type=str,
        default="queries",
        help="Subset name for query texts.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name.",
    )
    parser.add_argument(
        "--query-text-column",
        type=str,
        default="query",
        help="Column containing query text.",
    )
    parser.add_argument(
        "--document-subset",
        type=str,
        default="corpus",
        help="Subset name for document texts.",
    )
    parser.add_argument(
        "--document-text-column",
        type=str,
        default="passage",
        help="Column containing document text.",
    )
    parser.add_argument(
        "--bert-model-name",
        type=str,
        default="bert-base-uncased",
        help="BERT tokenizer name or path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size used for tokenizer calls.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional cap on processed queries (for quick checks).",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap on processed documents (for quick checks).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--use-slow-tokenizer",
        action="store_true",
        help="Use BertTokenizer instead of BertTokenizerFast.",
    )
    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="Do not include special tokens ([CLS], [SEP]) in token counts.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path for writing stats JSON.",
    )
    return parser


def _build_bert_tokenizer(
    model_name: str, *, use_slow_tokenizer: bool
) -> PreTrainedTokenizerBase:
    if use_slow_tokenizer:
        return BertTokenizer.from_pretrained(model_name)
    return BertTokenizerFast.from_pretrained(model_name)


def _load_text_dataset(
    *,
    hf_name: str,
    hf_subset: str | None,
    split: str,
    cache_dir: str | None,
) -> Dataset:
    return load_dataset(
        hf_name,
        name=hf_subset,
        split=split,
        cache_dir=cache_dir,
    )


def _resolve_text_column(
    dataset: Dataset,
    configured_text_column: str,
    *,
    fallback_columns: Sequence[str],
    field_label: str,
) -> str:
    if configured_text_column in dataset.column_names:
        return configured_text_column
    for fallback_column in fallback_columns:
        if fallback_column in dataset.column_names:
            log_if_rank_zero(
                logger,
                (
                    f"{field_label} text column '{configured_text_column}' not found. "
                    f"Using fallback column '{fallback_column}'."
                ),
                level="warning",
            )
            return fallback_column
    available_columns: str = ", ".join(dataset.column_names)
    raise ValueError(
        f"Could not find {field_label} text column. "
        f"Available columns: {available_columns}"
    )


def _compute_token_length_stats(
    dataset: Dataset,
    *,
    text_column: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    add_special_tokens: bool,
    max_items: int | None,
    progress_desc: str,
    item_label: str,
) -> TokenLengthStats:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    total_items: int = int(len(dataset))
    target_items: int = total_items
    if max_items is not None:
        if max_items <= 0:
            raise ValueError(f"max {item_label} must be positive when provided.")
        target_items = min(total_items, max_items)
    if target_items == 0:
        raise ValueError(f"No {item_label} available for token length analysis.")

    processed_items: int = 0
    token_total: int = 0
    min_tokens: int | None = None
    max_tokens: int = 0

    for start_idx in tqdm(
        range(0, target_items, batch_size),
        desc=progress_desc,
        unit="batch",
    ):
        end_idx: int = min(start_idx + batch_size, target_items)
        batch_rows: dict[str, list[Any]] = dataset[start_idx:end_idx]
        texts: list[str] = [
            "" if value is None else str(value) for value in batch_rows[text_column]
        ]
        tokenized = tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids_batch: list[list[int]] = tokenized["input_ids"]
        for token_ids in input_ids_batch:
            token_count: int = int(len(token_ids))
            token_total += token_count
            processed_items += 1
            if min_tokens is None or token_count < min_tokens:
                min_tokens = token_count
            if token_count > max_tokens:
                max_tokens = token_count

    if min_tokens is None:
        raise ValueError(
            f"Failed to compute token statistics; no {item_label} processed."
        )

    avg_tokens: float = float(token_total) / float(processed_items)
    return TokenLengthStats(
        item_count=processed_items,
        avg_tokens=avg_tokens,
        min_tokens=int(min_tokens),
        max_tokens=int(max_tokens),
    )


def _write_json_output(path: str, payload: dict[str, Any]) -> None:
    output_path: Path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    hf_name: str = str(args.hf_name)
    split: str = str(args.split)
    cache_dir: str | None = normalize_optional_str(args.cache_dir)
    target: str = str(args.target)
    add_special_tokens: bool = not bool(args.no_special_tokens)
    tokenizer: PreTrainedTokenizerBase = _build_bert_tokenizer(
        str(args.bert_model_name),
        use_slow_tokenizer=bool(args.use_slow_tokenizer),
    )
    payload: dict[str, Any] = {
        "hf_name": hf_name,
        "split": split,
        "tokenizer": str(tokenizer.name_or_path),
        "add_special_tokens": add_special_tokens,
        "target": target,
    }
    batch_size: int = int(args.batch_size)

    if target in {"query", "both"}:
        query_subset: str | None = normalize_optional_str(args.query_subset)
        query_dataset: Dataset = _load_text_dataset(
            hf_name=hf_name,
            hf_subset=query_subset,
            split=split,
            cache_dir=cache_dir,
        )
        query_text_column: str = _resolve_text_column(
            query_dataset,
            str(args.query_text_column),
            fallback_columns=("query", "text"),
            field_label="query",
        )
        query_stats: TokenLengthStats = _compute_token_length_stats(
            query_dataset,
            text_column=query_text_column,
            tokenizer=tokenizer,
            batch_size=batch_size,
            add_special_tokens=add_special_tokens,
            max_items=None if args.max_queries is None else int(args.max_queries),
            progress_desc="Tokenizing MSMARCO queries",
            item_label="queries",
        )
        payload["query_stats"] = {
            "hf_subset": query_subset,
            "text_column": query_text_column,
            "query_count": query_stats.item_count,
            "avg_tokens": query_stats.avg_tokens,
            "min_tokens": query_stats.min_tokens,
            "max_tokens": query_stats.max_tokens,
        }

    if target in {"document", "both"}:
        document_subset: str | None = normalize_optional_str(args.document_subset)
        document_dataset: Dataset = _load_text_dataset(
            hf_name=hf_name,
            hf_subset=document_subset,
            split=split,
            cache_dir=cache_dir,
        )
        document_text_column: str = _resolve_text_column(
            document_dataset,
            str(args.document_text_column),
            fallback_columns=("passage", "text"),
            field_label="document",
        )
        document_stats: TokenLengthStats = _compute_token_length_stats(
            document_dataset,
            text_column=document_text_column,
            tokenizer=tokenizer,
            batch_size=batch_size,
            add_special_tokens=add_special_tokens,
            max_items=None if args.max_documents is None else int(args.max_documents),
            progress_desc="Tokenizing MSMARCO documents",
            item_label="documents",
        )
        payload["document_stats"] = {
            "hf_subset": document_subset,
            "text_column": document_text_column,
            "document_count": document_stats.item_count,
            "avg_tokens": document_stats.avg_tokens,
            "min_tokens": document_stats.min_tokens,
            "max_tokens": document_stats.max_tokens,
        }

    log_if_rank_zero(logger, f"Token stats: {json.dumps(payload, ensure_ascii=False)}")

    output_json_path: str | None = normalize_optional_str(args.output_json)
    if output_json_path is not None:
        _write_json_output(output_json_path, payload)
        log_if_rank_zero(logger, f"Wrote stats JSON to: {output_json_path}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
