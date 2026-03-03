#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class RunningTokenStats:
    count: int = 0
    total_tokens: int = 0
    min_tokens: int | None = None
    max_tokens: int | None = None
    min_query_id: str | None = None
    max_query_id: str | None = None

    def update(self, token_count: int, query_id: str | None) -> None:
        self.count += 1
        self.total_tokens += int(token_count)
        if self.min_tokens is None or token_count < self.min_tokens:
            self.min_tokens = int(token_count)
            self.min_query_id = query_id
        if self.max_tokens is None or token_count > self.max_tokens:
            self.max_tokens = int(token_count)
            self.max_query_id = query_id

    @property
    def average_tokens(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self.total_tokens) / float(self.count)


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    text: str = value.strip()
    return text if text else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize the MSMARCO query set and report average/min/max token lengths."
        )
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
        help="Subset name that contains query text.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column containing query text.",
    )
    parser.add_argument(
        "--query-id-column",
        type=str,
        default="_id",
        help="Optional query id column for min/max reporting.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer name/path (BERT tokenizer by default).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for tokenizer calls.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional max token length (enables truncation when set).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for dataset loading.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit (debugging/smoke test).",
    )
    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="Exclude tokenizer special tokens from token counts.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    args: argparse.Namespace = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError("--max-length must be positive when provided.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    return args


def _load_query_dataset(
    *,
    hf_name: str,
    query_subset: str | None,
    split: str,
    cache_dir: str | None,
    streaming: bool,
) -> Dataset | IterableDataset:
    dataset_kwargs: dict[str, Any] = {
        "split": split,
        "cache_dir": cache_dir,
        "streaming": streaming,
    }
    if query_subset is None:
        return load_dataset(hf_name, **dataset_kwargs)
    return load_dataset(hf_name, name=query_subset, **dataset_kwargs)


def _token_lengths_for_batch(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    add_special_tokens: bool,
    max_length: int | None,
) -> list[int]:
    tokenizer_kwargs: dict[str, Any] = {
        "padding": False,
        "add_special_tokens": add_special_tokens,
    }
    if max_length is None:
        tokenizer_kwargs["truncation"] = False
    else:
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = int(max_length)

    encoded: dict[str, Any] = tokenizer(texts, **tokenizer_kwargs)
    input_ids: list[list[int]] = [list(ids) for ids in encoded["input_ids"]]
    return [len(ids) for ids in input_ids]


def _write_json_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = _parse_args()

    query_subset: str | None = _normalize_optional_str(args.query_subset)
    cache_dir: str | None = _normalize_optional_str(args.cache_dir)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    dataset: Dataset | IterableDataset = _load_query_dataset(
        hf_name=str(args.hf_name),
        query_subset=query_subset,
        split=str(args.split),
        cache_dir=cache_dir,
        streaming=bool(args.streaming),
    )

    total_rows: int | None = None
    if isinstance(dataset, Dataset):
        total_rows = int(len(dataset))
        if args.limit is not None:
            total_rows = min(total_rows, int(args.limit))
    elif args.limit is not None:
        total_rows = int(args.limit)

    stats = RunningTokenStats()
    batch_texts: list[str] = []
    batch_query_ids: list[str | None] = []
    processed_rows: int = 0
    add_special_tokens: bool = not bool(args.no_special_tokens)
    progress = tqdm(total=total_rows, desc="Tokenizing queries", unit="query")

    for row in dataset:
        if args.limit is not None and processed_rows >= int(args.limit):
            break
        if args.text_column not in row:
            raise KeyError(
                f"Missing text column '{args.text_column}' in dataset row keys: "
                f"{list(row.keys())}"
            )
        text_value: Any | None = row[args.text_column]
        text: str = "" if text_value is None else str(text_value)
        query_id_value: Any | None = row.get(args.query_id_column)
        query_id: str | None = None if query_id_value is None else str(query_id_value)
        batch_texts.append(text)
        batch_query_ids.append(query_id)
        processed_rows += 1

        if len(batch_texts) >= int(args.batch_size):
            lengths = _token_lengths_for_batch(
                tokenizer,
                batch_texts,
                add_special_tokens=add_special_tokens,
                max_length=args.max_length,
            )
            for length, row_query_id in zip(lengths, batch_query_ids):
                stats.update(length, row_query_id)
            progress.update(len(batch_texts))
            batch_texts = []
            batch_query_ids = []

    if batch_texts:
        lengths = _token_lengths_for_batch(
            tokenizer,
            batch_texts,
            add_special_tokens=add_special_tokens,
            max_length=args.max_length,
        )
        for length, row_query_id in zip(lengths, batch_query_ids):
            stats.update(length, row_query_id)
        progress.update(len(batch_texts))

    progress.close()

    if stats.count == 0:
        raise ValueError("No queries were processed. Check dataset/split settings.")

    print("MSMARCO query token statistics")
    print(f"Dataset: {args.hf_name} / subset={query_subset} / split={args.split}")
    print(f"Tokenizer: {args.tokenizer_name}")
    print(f"Add special tokens: {add_special_tokens}")
    if args.max_length is None:
        print("Truncation: disabled")
    else:
        print(f"Truncation: enabled (max_length={int(args.max_length)})")
    print(f"Processed queries: {stats.count}")
    print(f"Average tokens: {stats.average_tokens:.6f}")
    print(f"Minimum tokens: {stats.min_tokens} (query_id={stats.min_query_id})")
    print(f"Maximum tokens: {stats.max_tokens} (query_id={stats.max_query_id})")

    output_path_value: str | None = _normalize_optional_str(args.output_path)
    if output_path_value is not None:
        output_path: Path = Path(output_path_value)
        payload: dict[str, Any] = {
            "hf_name": str(args.hf_name),
            "query_subset": query_subset,
            "split": str(args.split),
            "text_column": str(args.text_column),
            "query_id_column": str(args.query_id_column),
            "tokenizer_name": str(args.tokenizer_name),
            "add_special_tokens": add_special_tokens,
            "max_length": None if args.max_length is None else int(args.max_length),
            "processed_queries": int(stats.count),
            "average_tokens": float(stats.average_tokens),
            "min_tokens": int(stats.min_tokens or 0),
            "min_query_id": stats.min_query_id,
            "max_tokens": int(stats.max_tokens or 0),
            "max_query_id": stats.max_query_id,
        }
        _write_json_output(output_path, payload)
        print(f"Saved stats JSON: {output_path}")


if __name__ == "__main__":
    main()
