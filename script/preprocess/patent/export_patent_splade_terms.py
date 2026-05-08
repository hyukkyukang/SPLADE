from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocess.patent_splade_terms import (
    DEFAULT_DOCUMENT_ENCODING_MODE,
    DEFAULT_TERM_OUTPUT_MODE,
    VALID_DOCUMENT_ENCODING_MODES,
    VALID_TERM_OUTPUT_MODES,
    export_patent_splade_terms,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export SPLADE sparse term-weight dictionaries for the patent ids "
            "referenced by one or more testset JSON files."
        )
    )
    parser.add_argument(
        "--dataset-json",
        nargs="+",
        required=True,
        help="Input JSON files with question_id/label_id rows.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output JSON path for the final per-document SPLADE payload.",
    )
    parser.add_argument(
        "--model-name",
        default="naver/splade-v3",
        help="Hugging Face model name or local path for the SPLADE model.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional Lightning checkpoint to load on top of the base model.",
    )
    parser.add_argument(
        "--patent-source",
        default="huggingface",
        choices=["huggingface", "parquet", "hf", "opensearch"],
        help="Where to read patent title/abstract/claims from.",
    )
    parser.add_argument(
        "--corpus-path",
        default="/home/user/SPLADE/.cache/hf/patent-us-corpus",
        help=(
            "Local patent corpus directory, manifest JSON, single parquet file, "
            "or parquet glob. Used when patent-source=huggingface/parquet/hf."
        ),
    )
    parser.add_argument(
        "--opensearch-url",
        default="http://10.4.43.27:9200",
        help="Base URL for the patent OpenSearch cluster.",
    )
    parser.add_argument(
        "--opensearch-index",
        default="patent_search",
        help="OpenSearch index or alias that stores the patent corpus.",
    )
    parser.add_argument(
        "--opensearch-batch-size",
        type=int,
        default=512,
        help="Document batch size for OpenSearch _mget requests.",
    )
    parser.add_argument(
        "--document-batch-size",
        type=int,
        default=256,
        help="Number of patents per DataLoader batch before window flattening.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=32,
        help="Window batch size for SPLADE document encoding.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for patent window preparation.",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=2,
        help="Per-worker DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenizer input length per SPLADE window.",
    )
    parser.add_argument(
        "--claim-overlap-tokens",
        type=int,
        default=0,
        help="Overlap tokens between adjacent claim windows.",
    )
    parser.add_argument(
        "--document-encoding-mode",
        default=DEFAULT_DOCUMENT_ENCODING_MODE,
        choices=list(VALID_DOCUMENT_ENCODING_MODES),
        help=(
            "How to format and truncate patent documents before SPLADE encoding. "
            "'split_fields_windowed' keeps the current abstract/claims windowing; "
            "'combined_fields_truncate_head' builds one combined Title/Abstract/Claims "
            "sequence and keeps only its leading max-length tokens."
        ),
    )
    parser.add_argument(
        "--term-output-mode",
        default=DEFAULT_TERM_OUTPUT_MODE,
        choices=list(VALID_TERM_OUTPUT_MODES),
        help=(
            "How to serialize extracted SPLADE terms per document. "
            "'flat_terms' writes {term: weight}; "
            "'source_token_terms' writes {source_token_key: {term: weight}} where "
            "source_token_key records the winning max-pooled window/token position."
        ),
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Drop terms with weights <= this threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional maximum number of terms to keep per document.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index for multi-process patent splitting.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of contiguous patent-id shards.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="json",
        help="Output format. Use jsonl for per-GPU shard files.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Timeout for each OpenSearch request.",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU inference even when CUDA is available.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the Hugging Face model.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_patent_splade_terms(
        dataset_paths=[Path(path) for path in args.dataset_json],
        output_path=args.output_path,
        model_name=str(args.model_name),
        checkpoint_path=None
        if args.checkpoint_path is None
        else str(args.checkpoint_path),
        patent_source=str(args.patent_source),
        corpus_path=str(args.corpus_path),
        opensearch_url=str(args.opensearch_url),
        opensearch_index=str(args.opensearch_index),
        opensearch_batch_size=int(args.opensearch_batch_size),
        document_batch_size=int(args.document_batch_size),
        encode_batch_size=int(args.encode_batch_size),
        dataloader_num_workers=int(args.dataloader_num_workers),
        dataloader_prefetch_factor=(
            None
            if int(args.dataloader_num_workers) <= 0
            else int(args.dataloader_prefetch_factor)
        ),
        max_length=int(args.max_length),
        claim_overlap_tokens=int(args.claim_overlap_tokens),
        document_encoding_mode=str(args.document_encoding_mode),
        term_output_mode=str(args.term_output_mode),
        min_weight=float(args.min_weight),
        top_k=None if args.top_k is None else int(args.top_k),
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
        output_format=str(args.output_format),
        use_cpu=bool(args.use_cpu),
        trust_remote_code=bool(args.trust_remote_code),
        timeout_seconds=int(args.timeout_seconds),
        show_progress=not bool(args.no_progress),
    )


if __name__ == "__main__":
    main()
