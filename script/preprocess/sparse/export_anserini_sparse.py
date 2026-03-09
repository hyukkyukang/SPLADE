#!/usr/bin/env python3
"""Export existing sparse encode shards to Anserini sparse-vector formats.

This reuses already-materialized sparse shard outputs and can apply a smaller
doc-side top-k at export time, so the Lucene index can be built without running
model encoding again.

Outputs:
- documents: one JSONL file per sparse shard in Anserini JsonVectorCollection format
- queries: a TSV with repeated tokens, compatible with Anserini impact search
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from transformers import AutoTokenizer
import yaml

from src.index.sparse import ShardInfo, load_shard_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encode-dir",
        type=Path,
        required=True,
        help="Existing sparse encode output directory containing shards/",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer path or HF model name used to map term ids to tokens.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write Anserini export files into.",
    )
    parser.add_argument(
        "--export-docs",
        action="store_true",
        help="Export document JSONL files for Anserini JsonVectorCollection.",
    )
    parser.add_argument(
        "--export-queries",
        action="store_true",
        help="Export query TSV using doc vectors for the requested query ids.",
    )
    parser.add_argument(
        "--query-ids-path",
        type=Path,
        default=None,
        help="Optional path containing query ids. Supports .parquet/.json/.jsonl/.txt.",
    )
    parser.add_argument(
        "--query-id-column",
        type=str,
        default="query_id",
        help="Column name when --query-ids-path points to a parquet/jsonl file.",
    )
    parser.add_argument(
        "--doc-top-k",
        type=int,
        default=128,
        help="Keep at most this many terms per document at export time.",
    )
    parser.add_argument(
        "--query-top-k",
        type=int,
        default=32,
        help="Keep at most this many terms per query when exporting query TSV.",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=None,
        help="Optional override min-weight before quantization. Defaults to source manifest min_weight.",
    )
    parser.add_argument(
        "--quantization-factor",
        type=float,
        default=100.0,
        help="Scale factor before integer rounding, matching SPLADE's Anserini export convention.",
    )
    parser.add_argument(
        "--fallback-token-id",
        type=int,
        default=998,
        help="Token id inserted when pruning+quantization produces an empty vector.",
    )
    parser.add_argument(
        "--allow-query-doc-mismatch",
        action="store_true",
        help="Bypass the safety check that query/doc encoders must match to reuse doc vectors as queries.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Optional debug cap on number of shards to export.",
    )
    parser.add_argument(
        "--start-shard-index",
        type=int,
        default=0,
        help="Start from this global shard index in the sorted shard manifest.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional debug cap on number of source docs to process.",
    )
    parser.add_argument(
        "--progress-every-docs",
        type=int,
        default=100000,
        help="Emit progress every N processed source documents. Set <=0 to disable.",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not write metadata.json. Useful when multiple worker exports share one output dir.",
    )
    args = parser.parse_args()
    if not args.export_docs and not args.export_queries:
        parser.error("At least one of --export-docs or --export-queries must be set.")
    if args.export_queries and args.query_ids_path is None:
        parser.error("--query-ids-path is required when --export-queries is set.")
    if args.doc_top_k is not None and args.doc_top_k <= 0:
        parser.error("--doc-top-k must be positive.")
    if args.query_top_k is not None and args.query_top_k <= 0:
        parser.error("--query-top-k must be positive.")
    return args


def _load_token_id_to_token(tokenizer_name_or_path: str) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    vocab = tokenizer.get_vocab()
    if not vocab:
        raise ValueError(f"Tokenizer has empty vocab: {tokenizer_name_or_path}")
    max_id = max(int(token_id) for token_id in vocab.values())
    id_to_token = [""] * (max_id + 1)
    for token, token_id in vocab.items():
        id_to_token[int(token_id)] = str(token)
    if any(token == "" for token in id_to_token):
        raise ValueError(
            "Tokenizer vocab ids are not dense; unsupported for direct id lookup."
        )
    return id_to_token


def _load_id_set(path: Path, column: str) -> set[str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=[column])
        return {str(value) for value in table.column(0).to_pylist()}
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return {str(value) for value in payload}
        raise ValueError(f"Expected list JSON in {path}")
    if suffix == ".jsonl":
        values: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if column not in record:
                    raise ValueError(f"Missing {column} in {path}")
                values.add(str(record[column]))
        return values
    values = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                values.add(value)
    return values


def _validate_query_export_compatibility(
    encode_dir: Path, *, allow_mismatch: bool
) -> None:
    config_path = encode_dir / "config.yaml"
    if not config_path.exists():
        if allow_mismatch:
            return
        raise FileNotFoundError(
            f"Missing {config_path}; cannot verify query/doc compatibility."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_cfg = config.get("model") or {}
    doc_only = bool(model_cfg.get("doc_only", False))
    query_pooling = model_cfg.get("query_pooling")
    doc_pooling = model_cfg.get("doc_pooling")

    if doc_only or query_pooling != doc_pooling:
        if allow_mismatch:
            return
        raise ValueError(
            "Reusing doc vectors as query vectors is not safe for this model. "
            "Expected doc_only=false and identical query_pooling/doc_pooling. "
            f"Found doc_only={doc_only}, query_pooling={query_pooling}, doc_pooling={doc_pooling}."
        )


def _select_topk(
    indices: np.ndarray,
    values: np.ndarray,
    *,
    top_k: int | None,
    min_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if min_weight > 0.0:
        mask = values > min_weight
        if not bool(mask.all()):
            indices = indices[mask]
            values = values[mask]
    else:
        mask = values > 0.0
        if not bool(mask.all()):
            indices = indices[mask]
            values = values[mask]
    if indices.size == 0:
        return indices, values
    if top_k is not None and indices.size > top_k:
        keep_pos = np.argpartition(values, -top_k)[-top_k:]
        indices = indices[keep_pos]
        values = values[keep_pos]
    if indices.size > 1:
        order = np.argsort(indices)
        indices = indices[order]
        values = values[order]
    return indices, values


def _quantize(
    indices: np.ndarray,
    values: np.ndarray,
    *,
    quantization_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    if indices.size == 0:
        return indices.astype(np.int32, copy=False), np.zeros((0,), dtype=np.int32)
    quantized = np.rint(values.astype(np.float32, copy=False) * quantization_factor)
    quantized = quantized.astype(np.int32, copy=False)
    mask = quantized > 0
    if not bool(mask.all()):
        indices = indices[mask]
        quantized = quantized[mask]
    return indices.astype(np.int32, copy=False), quantized


def _vector_dict(
    indices: np.ndarray,
    counts: np.ndarray,
    *,
    id_to_token: Sequence[str],
    fallback_token: str,
) -> dict[str, int]:
    if indices.size == 0:
        return {fallback_token: 1}
    vector = {}
    for term_id, count in zip(indices.tolist(), counts.tolist()):
        if count > 0:
            vector[id_to_token[int(term_id)]] = int(count)
    if not vector:
        vector[fallback_token] = 1
    return vector


def _query_string(
    indices: np.ndarray,
    counts: np.ndarray,
    *,
    id_to_token: Sequence[str],
    fallback_token: str,
) -> str:
    if indices.size == 0:
        return fallback_token
    terms: list[str] = []
    for term_id, count in zip(indices.tolist(), counts.tolist()):
        if count <= 0:
            continue
        terms.extend([id_to_token[int(term_id)]] * int(count))
    if not terms:
        return fallback_token
    return " ".join(terms)


def _iter_shards(
    shard_infos: Sequence[ShardInfo], start_shard_index: int, max_shards: int | None
) -> Iterable[ShardInfo]:
    start = max(0, int(start_shard_index))
    selected = shard_infos[start:]
    if max_shards is None:
        return selected
    return selected[: max(0, int(max_shards))]


def main() -> None:
    args = parse_args()

    shard_infos, metadata = load_shard_manifest(args.encode_dir)
    min_weight = (
        float(metadata.get("min_weight") or 0.0)
        if args.min_weight is None
        else float(args.min_weight)
    )
    id_to_token = _load_token_id_to_token(args.tokenizer)
    if args.fallback_token_id < 0 or args.fallback_token_id >= len(id_to_token):
        raise ValueError(
            f"fallback token id {args.fallback_token_id} is out of range for vocab size {len(id_to_token)}"
        )
    fallback_token = id_to_token[int(args.fallback_token_id)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = args.output_dir / "docs"
    queries_path = args.output_dir / "queries.tsv"

    query_ids: set[str] | None = None
    remaining_query_ids: set[str] | None = None
    query_count = 0
    if args.export_queries:
        _validate_query_export_compatibility(
            args.encode_dir, allow_mismatch=bool(args.allow_query_doc_mismatch)
        )
        query_ids = _load_id_set(args.query_ids_path, args.query_id_column)
        remaining_query_ids = set(query_ids)

    source_doc_count = 0
    exported_doc_count = 0
    exported_query_count = 0
    shard_records: list[dict[str, object]] = []

    query_handle = None
    try:
        if args.export_docs:
            docs_dir.mkdir(parents=True, exist_ok=True)
        if args.export_queries:
            query_handle = queries_path.open("w", encoding="utf-8")

        for shard in _iter_shards(
            shard_infos, args.start_shard_index, args.max_shards
        ):
            if (
                not args.export_docs
                and remaining_query_ids is not None
                and not remaining_query_ids
            ):
                break
            indptr = np.load(shard.indptr_path)
            indices = np.load(shard.indices_path)
            values = np.load(shard.values_path)
            with shard.doc_ids_path.open("r", encoding="utf-8") as doc_file:
                doc_ids: list[str] = json.load(doc_file)

            shard_jsonl_path = None
            shard_handle = None
            shard_doc_count = 0
            if args.export_docs:
                shard_jsonl_path = (
                    docs_dir / f"rank{shard.rank:02d}_shard{shard.shard_id:06d}.jsonl"
                )
                shard_handle = shard_jsonl_path.open("w", encoding="utf-8")

            try:
                for row_idx, doc_id in enumerate(doc_ids):
                    start = int(indptr[row_idx])
                    end = int(indptr[row_idx + 1])
                    row_indices = indices[start:end]
                    row_values = values[start:end]
                    source_doc_count += 1

                    if args.export_docs:
                        d_indices, d_values = _select_topk(
                            row_indices,
                            row_values,
                            top_k=args.doc_top_k,
                            min_weight=min_weight,
                        )
                        d_indices, d_counts = _quantize(
                            d_indices,
                            d_values,
                            quantization_factor=float(args.quantization_factor),
                        )
                        payload = {
                            "id": str(doc_id),
                            "content": "",
                            "vector": _vector_dict(
                                d_indices,
                                d_counts,
                                id_to_token=id_to_token,
                                fallback_token=fallback_token,
                            ),
                        }
                        shard_handle.write(json.dumps(payload) + "\n")
                        shard_doc_count += 1
                        exported_doc_count += 1

                    if remaining_query_ids is not None and str(doc_id) in remaining_query_ids:
                        q_indices, q_values = _select_topk(
                            row_indices,
                            row_values,
                            top_k=args.query_top_k,
                            min_weight=min_weight,
                        )
                        q_indices, q_counts = _quantize(
                            q_indices,
                            q_values,
                            quantization_factor=float(args.quantization_factor),
                        )
                        query_text = _query_string(
                            q_indices,
                            q_counts,
                            id_to_token=id_to_token,
                            fallback_token=fallback_token,
                        )
                        query_handle.write(f"{doc_id}\t{query_text}\n")
                        exported_query_count += 1
                        query_count += 1
                        remaining_query_ids.discard(str(doc_id))
                        if not args.export_docs and not remaining_query_ids:
                            break

                    if args.max_docs is not None and source_doc_count >= int(args.max_docs):
                        break
                    progress_every = int(args.progress_every_docs)
                    if progress_every > 0 and source_doc_count % progress_every == 0:
                        print(
                            json.dumps(
                                {
                                    "processed_docs": int(source_doc_count),
                                    "exported_docs": int(exported_doc_count),
                                    "exported_queries": int(exported_query_count),
                                    "current_rank": int(shard.rank),
                                    "current_shard_id": int(shard.shard_id),
                                }
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                if shard_jsonl_path is not None:
                    shard_records.append(
                        {
                            "rank": shard.rank,
                            "shard_id": shard.shard_id,
                            "path": str(shard_jsonl_path),
                            "doc_count": shard_doc_count,
                        }
                    )
                print(
                    json.dumps(
                        {
                            "completed_rank": int(shard.rank),
                            "completed_shard_id": int(shard.shard_id),
                            "shard_docs": int(shard_doc_count),
                            "processed_docs": int(source_doc_count),
                            "exported_docs": int(exported_doc_count),
                            "exported_queries": int(exported_query_count),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                if shard_handle is not None:
                    shard_handle.close()

            if args.max_docs is not None and source_doc_count >= int(args.max_docs):
                break
    finally:
        if query_handle is not None:
            query_handle.close()

    metadata_out = {
        "source_encode_dir": str(args.encode_dir),
        "source_top_k": metadata.get("top_k"),
        "source_min_weight": metadata.get("min_weight"),
        "quantization_factor": float(args.quantization_factor),
        "doc_top_k": int(args.doc_top_k),
        "query_top_k": int(args.query_top_k),
        "effective_min_weight": float(min_weight),
        "fallback_token_id": int(args.fallback_token_id),
        "fallback_token": fallback_token,
        "docs_exported": int(exported_doc_count),
        "queries_exported": int(exported_query_count),
        "missing_query_ids": sorted(remaining_query_ids) if remaining_query_ids else [],
        "doc_files": shard_records,
    }
    if not args.skip_metadata:
        with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata_out, handle, indent=2)

    print(json.dumps(metadata_out, indent=2))


if __name__ == "__main__":
    main()
