import argparse
import glob
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.patent_text import format_patent_document_text, normalize_patent_text
from src.model.pl_module.utils import build_retrieval_model_with_checkpoint
from src.model.retriever.dense.neural.hf_dense import DenseEncoder, DenseRetrievalModel
from src.utils.transformers import build_tokenizer


DEFAULT_VARIANT_PAIRS: tuple[str, ...] = (
    "artifact:full",
    "abstract:abstract",
    "title_abstract:title_abstract",
    "full:full",
    "abstract:full",
    "title_abstract:full",
    "plain_abstract:plain_abstract",
    "plain_title_abstract:plain_title_abstract",
    "plain_full:plain_full",
    "plain_abstract:plain_full",
)

DEFAULT_POOLINGS: tuple[str, ...] = ("pooler", "cls", "mean")
DEFAULT_SIMILARITIES: tuple[str, ...] = ("dot", "cosine")
K_LIST: tuple[int, ...] = (1, 5, 10, 16, 32)


@dataclass(frozen=True)
class SampledPair:
    query_id: str
    positive_doc_id: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose dense patent retrieval mismatches with sampled qrel pairs and "
            "small candidate-pool ablations."
        )
    )
    parser.add_argument(
        "--model-config",
        default="config/model/dpr_bilingual_negative1_ko_en.yaml",
        help="Dense model preset to diagnose.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=(
            "data/model/dpr_biencoder_negative1_ko_en_20251202_filtered/"
            "dpr_biencoder.9.146"
        ),
        help="Checkpoint used for dense retrieval.",
    )
    parser.add_argument(
        "--queries-path",
        default="data/eval/patent_us_small/queries.parquet",
        help="Evaluation queries parquet path.",
    )
    parser.add_argument(
        "--qrels-path",
        default="data/eval/patent_us_small/qrels.parquet",
        help="Evaluation qrels parquet path.",
    )
    parser.add_argument(
        "--corpus-glob",
        default=".cache/hf/patent-us-corpus-small/data/*.parquet",
        help="Glob for corpus parquet files used to render patent texts.",
    )
    parser.add_argument(
        "--doc-ids-json",
        default=(
            "data/index/dpr_bilingual_negative1_ko_en/"
            "dpr_patent_us_small_full_gpu_20260403/doc_ids.json"
        ),
        help="JSON list of doc IDs available in the encoded/indexed corpus.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=64,
        help="Number of distinct qrel query-doc pairs to sample.",
    )
    parser.add_argument(
        "--negative-pool-size",
        type=int,
        default=256,
        help="Number of random negative documents shared across sampled queries.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size for diagnostics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling queries and negatives.",
    )
    parser.add_argument(
        "--variant-pairs",
        nargs="+",
        default=list(DEFAULT_VARIANT_PAIRS),
        help=(
            "Pairs of query_variant:doc_variant to evaluate. "
            f"Default: {' '.join(DEFAULT_VARIANT_PAIRS)}"
        ),
    )
    parser.add_argument(
        "--poolings",
        nargs="+",
        default=list(DEFAULT_POOLINGS),
        help=f"Token pooling modes to test. Default: {' '.join(DEFAULT_POOLINGS)}",
    )
    parser.add_argument(
        "--similarities",
        nargs="+",
        default=list(DEFAULT_SIMILARITIES),
        help=(
            f"Similarity functions to test. Default: {' '.join(DEFAULT_SIMILARITIES)}"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for diagnosis, for example cuda or cpu.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="How many top and bottom configurations to print.",
    )
    parser.add_argument(
        "--output-json",
        default="log/dense_patent_diagnostics/diagnostic_results.json",
        help="Where to save the full diagnostic report JSON.",
    )
    return parser.parse_args()


def _load_model_cfg(model_config_path: str) -> Any:
    base_cfg = OmegaConf.load("config/model/_base.yaml")
    preset_cfg = OmegaConf.load(model_config_path)
    model_cfg = OmegaConf.merge(base_cfg, preset_cfg)
    return OmegaConf.create({"model": model_cfg})


def _render_patent_variant(row: dict[str, Any], variant: str) -> str:
    normalized_variant: str = str(variant).strip().lower()
    if normalized_variant == "full":
        return format_patent_document_text(row)

    title: str = normalize_patent_text(row.get("title"))
    abstract: str = normalize_patent_text(row.get("abstract"))
    claims: str = normalize_patent_text(row.get("claims"))
    description: str = normalize_patent_text(row.get("description"))
    if normalized_variant == "artifact":
        return normalize_patent_text(row.get("artifact_text"))

    plain: bool = normalized_variant.startswith("plain_")
    suffix: str = normalized_variant[6:] if plain else normalized_variant
    labeled_fields: list[str] = []
    plain_fields: list[str] = []

    def add_field(label: str, value: str) -> None:
        if not value:
            return
        labeled_fields.append(f"{label}: {value}")
        plain_fields.append(value)

    if suffix == "abstract":
        add_field("Abstract", abstract)
    elif suffix == "title_abstract":
        add_field("Title", title)
        add_field("Abstract", abstract)
    elif suffix == "title_abstract_claims":
        add_field("Title", title)
        add_field("Abstract", abstract)
        add_field("Claims", claims)
    elif suffix == "title_abstract_description":
        add_field("Title", title)
        add_field("Abstract", abstract)
        add_field("Description", description)
    elif suffix == "full":
        add_field("Title", title)
        add_field("Abstract", abstract)
        add_field("Claims", claims)
        add_field("Description", description)
    else:
        raise ValueError(f"Unsupported text variant: {variant!r}")
    parts: list[str] = plain_fields if plain else labeled_fields
    return "\n".join(parts).strip()


def _get_encoder(model: DenseRetrievalModel, *, is_query: bool) -> DenseEncoder:
    if is_query:
        encoder = getattr(model, "query_encoder", None)
        if isinstance(encoder, DenseEncoder):
            return encoder
    else:
        encoder = getattr(model, "ctx_encoder", None)
        if isinstance(encoder, DenseEncoder):
            return encoder
    fallback = getattr(model, "encoder", None)
    if isinstance(fallback, DenseEncoder):
        return fallback
    raise TypeError("Dense diagnostic expected a DenseEncoder-compatible model.")


def _encode_texts(
    *,
    texts: list[str],
    tokenizer: Any,
    encoder: DenseEncoder,
    pooling: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    start: int
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokenized = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with torch.no_grad():
            batch_embeddings = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pooling=pooling,
            )
        embeddings.append(batch_embeddings.float().cpu().numpy())
    if not embeddings:
        return np.zeros((0, int(encoder.embedding_dim)), dtype=np.float32)
    return np.concatenate(embeddings, axis=0)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def _score_matrix(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    *,
    similarity: str,
) -> np.ndarray:
    if similarity == "cosine":
        query_embeddings = _normalize_embeddings(query_embeddings)
        doc_embeddings = _normalize_embeddings(doc_embeddings)
    return query_embeddings @ doc_embeddings.T


def _resolve_variant_pairs(raw_pairs: Iterable[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    raw_pair: str
    for raw_pair in raw_pairs:
        left, sep, right = str(raw_pair).partition(":")
        if not sep:
            raise ValueError(
                f"Variant pair must have query:doc form, got {raw_pair!r}."
            )
        pairs.append((left.strip(), right.strip()))
    if not pairs:
        raise ValueError("At least one query:doc variant pair is required.")
    return pairs


def _load_doc_id_set(doc_ids_path: str) -> tuple[list[str], set[str]]:
    with open(doc_ids_path, encoding="utf-8") as handle:
        doc_ids: list[str] = [str(doc_id) for doc_id in json.load(handle)]
    return doc_ids, set(doc_ids)


def _sample_pairs(
    *,
    qrels_df: pd.DataFrame,
    indexed_doc_ids: set[str],
    sample_size: int,
    seed: int,
) -> list[SampledPair]:
    rng = random.Random(seed)
    grouped = (
        qrels_df.assign(
            query_id=qrels_df["query_id"].astype(str),
            doc_id=qrels_df["doc_id"].astype(str),
        )
        .groupby("query_id")["doc_id"]
        .apply(list)
        .to_dict()
    )
    all_query_ids: list[str] = list(grouped.keys())
    rng.shuffle(all_query_ids)
    sampled: list[SampledPair] = []
    query_id: str
    for query_id in all_query_ids:
        if query_id not in indexed_doc_ids:
            continue
        positive_doc_id: str | None = None
        for doc_id in grouped[query_id]:
            if doc_id in indexed_doc_ids:
                positive_doc_id = doc_id
                break
        if positive_doc_id is None:
            continue
        sampled.append(
            SampledPair(query_id=str(query_id), positive_doc_id=str(positive_doc_id))
        )
        if len(sampled) >= sample_size:
            break
    return sampled


def _load_corpus_rows_and_sample_negatives(
    *,
    corpus_glob: str,
    target_doc_ids: set[str],
    excluded_doc_ids: set[str],
    negative_pool_size: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[str], int]:
    parquet_files: list[str] = sorted(glob.glob(corpus_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No corpus parquet files matched: {corpus_glob}")
    rng = random.Random(seed)
    matched_files: list[str] = []
    target_remaining: set[str] = set(target_doc_ids)
    negative_doc_ids: list[str] = []
    negative_seen: set[str] = set()
    parquet_file: str
    for parquet_file in parquet_files:
        doc_id_frame = pd.read_parquet(parquet_file, columns=["doc_id"])
        file_doc_ids: list[str] = [str(doc_id) for doc_id in doc_id_frame["doc_id"].tolist()]
        doc_ids_in_file: set[str] = set(file_doc_ids)
        matched_target_ids: set[str] = doc_ids_in_file & target_remaining
        if not matched_target_ids:
            continue
        matched_files.append(parquet_file)
        target_remaining -= matched_target_ids
        if len(negative_doc_ids) < negative_pool_size:
            shuffled_doc_ids: list[str] = list(file_doc_ids)
            rng.shuffle(shuffled_doc_ids)
            doc_id: str
            for doc_id in shuffled_doc_ids:
                if doc_id in excluded_doc_ids or doc_id in negative_seen:
                    continue
                negative_seen.add(doc_id)
                negative_doc_ids.append(doc_id)
                if len(negative_doc_ids) >= negative_pool_size:
                    break

    if not matched_files:
        return {}, [], 0

    if len(negative_doc_ids) < negative_pool_size:
        raise ValueError(
            "Unable to source enough negatives from files that contain sampled "
            f"queries/positives. Requested {negative_pool_size}, found {len(negative_doc_ids)}."
        )

    needed_doc_ids: set[str] = set(target_doc_ids) | set(negative_doc_ids)
    frames: list[pd.DataFrame] = []
    for parquet_file in matched_files:
        frame = pd.read_parquet(
            parquet_file,
            columns=["doc_id", "title", "abstract", "claims", "description"],
        )
        filtered = frame[frame["doc_id"].astype(str).isin(needed_doc_ids)]
        if not filtered.empty:
            frames.append(filtered)

    if not frames:
        return {}, [], len(matched_files)
    frame = pd.concat(frames, ignore_index=True)
    rows: dict[str, dict[str, Any]] = {}
    row: Any
    for row in frame.itertuples(index=False):
        row_dict: dict[str, Any] = row._asdict()
        rows[str(row_dict["doc_id"])] = row_dict
    return rows, negative_doc_ids, len(matched_files)


def _build_query_rows(
    *,
    query_artifacts_df: pd.DataFrame,
    corpus_rows: dict[str, dict[str, Any]],
    sampled_pairs: list[SampledPair],
) -> list[dict[str, Any]]:
    artifact_text_by_query_id: dict[str, str] = {
        str(row.query_id): str(row.text)
        for row in query_artifacts_df.itertuples(index=False)
    }
    query_rows: list[dict[str, Any]] = []
    pair: SampledPair
    for pair in sampled_pairs:
        base_row: dict[str, Any] | None = corpus_rows.get(pair.query_id)
        if base_row is None:
            raise KeyError(f"Missing corpus row for query_id={pair.query_id}")
        row = dict(base_row)
        row["artifact_text"] = artifact_text_by_query_id.get(pair.query_id, "")
        query_rows.append(row)
    return query_rows


def _build_doc_rows(
    *,
    corpus_rows: dict[str, dict[str, Any]],
    doc_ids: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    doc_id: str
    for doc_id in doc_ids:
        row = corpus_rows.get(doc_id)
        if row is None:
            raise KeyError(f"Missing corpus row for doc_id={doc_id}")
        rows.append(dict(row))
    return rows


def _collect_texts_by_variant(
    *,
    rows: list[dict[str, Any]],
    variants: Iterable[str],
) -> dict[str, list[str]]:
    texts: dict[str, list[str]] = {}
    variant: str
    for variant in variants:
        texts[str(variant)] = [_render_patent_variant(row, str(variant)) for row in rows]
    return texts


def _evaluate_configuration(
    *,
    query_embeddings: np.ndarray,
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
    similarity: str,
) -> dict[str, Any]:
    positive_scores: np.ndarray = np.sum(
        _score_matrix(query_embeddings, positive_embeddings, similarity=similarity)
        * np.eye(query_embeddings.shape[0], dtype=np.float32),
        axis=1,
    )
    negative_scores: np.ndarray = _score_matrix(
        query_embeddings,
        negative_embeddings,
        similarity=similarity,
    )
    negative_greater: np.ndarray = (negative_scores > positive_scores[:, None]).sum(axis=1)
    ranks: np.ndarray = negative_greater + 1
    result: dict[str, Any] = {
        "sample_count": int(query_embeddings.shape[0]),
        "negative_pool_size": int(negative_embeddings.shape[0]),
        "positive_score_mean": float(positive_scores.mean()),
        "positive_score_median": float(np.median(positive_scores)),
        "negative_score_mean": float(negative_scores.mean()),
        "negative_score_median": float(np.median(negative_scores)),
        "positive_gt_mean_negative_frac": float(
            (positive_scores > negative_scores.mean(axis=1)).mean()
        ),
        "positive_gt_all_negatives_frac": float((negative_greater == 0).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "rank_mean": float(ranks.mean()),
        "rank_median": float(np.median(ranks)),
        "rank_max": int(ranks.max()),
    }
    k: int
    for k in K_LIST:
        result[f"recall_{k}"] = float((ranks <= k).mean())
    return result


def _format_metric(result: dict[str, Any], key: str) -> str:
    value: Any = result[key]
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("dense_patent_diagnostics")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = _load_model_cfg(args.model_config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_cpu: bool = device.type != "cuda"
    model = build_retrieval_model_with_checkpoint(
        cfg,
        use_cpu=use_cpu,
        checkpoint_path=args.checkpoint_path,
        logger=logger,
    )
    if not isinstance(model, DenseRetrievalModel):
        raise TypeError("Dense patent diagnostics require a dense retrieval model.")
    model.eval().to(device)

    tokenizer = build_tokenizer(
        str(cfg.model.tokenizer_name),
        use_fast_tokenizer=bool(cfg.model.get("use_fast_tokenizer", True)),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        require_fast_tokenizer=bool(cfg.model.get("require_fast_tokenizer", False)),
        local_files_only=(
            None
            if cfg.model.get("local_files_only") is None
            else bool(cfg.model.get("local_files_only"))
        ),
        revision=cfg.model.get("tokenizer_revision"),
    )
    logger.info("Tokenizer loaded from %s.", str(cfg.model.tokenizer_name))

    step_start = time.time()
    query_artifacts_df = pd.read_parquet(args.queries_path)
    qrels_df = pd.read_parquet(args.qrels_path)
    logger.info(
        "Loaded query artifacts (%d) and qrels (%d) in %.2fs.",
        len(query_artifacts_df),
        len(qrels_df),
        time.time() - step_start,
    )
    step_start = time.time()
    all_doc_ids, indexed_doc_id_set = _load_doc_id_set(args.doc_ids_json)
    logger.info(
        "Loaded %d indexed doc ids in %.2fs.",
        len(all_doc_ids),
        time.time() - step_start,
    )
    step_start = time.time()
    sampled_pairs = _sample_pairs(
        qrels_df=qrels_df,
        indexed_doc_ids=indexed_doc_id_set,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    if not sampled_pairs:
        raise ValueError("No sampled qrel pairs could be resolved against the index.")
    logger.info(
        "Sampled %d qrel pairs in %.2fs.",
        len(sampled_pairs),
        time.time() - step_start,
    )

    sampled_query_ids: list[str] = [pair.query_id for pair in sampled_pairs]
    sampled_positive_ids: list[str] = [pair.positive_doc_id for pair in sampled_pairs]
    excluded_doc_ids: set[str] = set(sampled_query_ids) | set(sampled_positive_ids)
    row_lookup_start = time.time()
    target_doc_ids: set[str] = set(sampled_query_ids) | set(sampled_positive_ids)
    corpus_rows, negative_doc_ids, matched_file_count = _load_corpus_rows_and_sample_negatives(
        corpus_glob=args.corpus_glob,
        target_doc_ids=target_doc_ids,
        excluded_doc_ids=excluded_doc_ids,
        negative_pool_size=args.negative_pool_size,
        seed=args.seed + 1,
    )
    logger.info(
        "Resolved %d corpus rows from %d matched files for %d targets + %d negatives in %.2fs.",
        len(corpus_rows),
        matched_file_count,
        len(target_doc_ids),
        len(negative_doc_ids),
        time.time() - row_lookup_start,
    )
    query_rows = _build_query_rows(
        query_artifacts_df=query_artifacts_df,
        corpus_rows=corpus_rows,
        sampled_pairs=sampled_pairs,
    )
    positive_rows = _build_doc_rows(
        corpus_rows=corpus_rows,
        doc_ids=sampled_positive_ids,
    )
    negative_rows = _build_doc_rows(
        corpus_rows=corpus_rows,
        doc_ids=negative_doc_ids,
    )
    logger.info(
        "Prepared %d query rows, %d positive rows, and %d negative rows.",
        len(query_rows),
        len(positive_rows),
        len(negative_rows),
    )

    variant_pairs = _resolve_variant_pairs(args.variant_pairs)
    query_variants: list[str] = sorted({query_variant for query_variant, _ in variant_pairs})
    doc_variants: list[str] = sorted({doc_variant for _, doc_variant in variant_pairs})
    query_texts_by_variant = _collect_texts_by_variant(
        rows=query_rows,
        variants=query_variants,
    )
    positive_texts_by_variant = _collect_texts_by_variant(
        rows=positive_rows,
        variants=doc_variants,
    )
    negative_texts_by_variant = _collect_texts_by_variant(
        rows=negative_rows,
        variants=doc_variants,
    )

    max_length: int = int(
        cfg.model.get(
            "max_length",
            max(
                int(getattr(cfg.model, "max_query_length", 512)),
                int(getattr(cfg.model, "max_doc_length", 512)),
            ),
        )
    )
    query_encoder = _get_encoder(model, is_query=True)
    doc_encoder = _get_encoder(model, is_query=False)
    query_embedding_cache: dict[tuple[str, str], np.ndarray] = {}
    positive_embedding_cache: dict[tuple[str, str], np.ndarray] = {}
    negative_embedding_cache: dict[tuple[str, str], np.ndarray] = {}

    variant: str
    pooling: str
    encode_start = time.time()
    for variant in query_variants:
        for pooling in args.poolings:
            query_embedding_cache[(variant, pooling)] = _encode_texts(
                texts=query_texts_by_variant[variant],
                tokenizer=tokenizer,
                encoder=query_encoder,
                pooling=pooling,
                batch_size=args.batch_size,
                max_length=max_length,
                device=device,
            )
    logger.info(
        "Encoded %d query variant/pooling combinations in %.2fs.",
        len(query_variants) * len(args.poolings),
        time.time() - encode_start,
    )
    encode_start = time.time()
    for variant in doc_variants:
        for pooling in args.poolings:
            positive_embedding_cache[(variant, pooling)] = _encode_texts(
                texts=positive_texts_by_variant[variant],
                tokenizer=tokenizer,
                encoder=doc_encoder,
                pooling=pooling,
                batch_size=args.batch_size,
                max_length=max_length,
                device=device,
            )
            negative_embedding_cache[(variant, pooling)] = _encode_texts(
                texts=negative_texts_by_variant[variant],
                tokenizer=tokenizer,
                encoder=doc_encoder,
                pooling=pooling,
                batch_size=args.batch_size,
                max_length=max_length,
                device=device,
            )
    logger.info(
        "Encoded %d doc variant/pooling combinations in %.2fs.",
        len(doc_variants) * len(args.poolings) * 2,
        time.time() - encode_start,
    )

    score_start = time.time()
    results: list[dict[str, Any]] = []
    query_variant: str
    doc_variant: str
    similarity: str
    for query_variant, doc_variant in variant_pairs:
        for pooling in args.poolings:
            query_embeddings = query_embedding_cache[(query_variant, pooling)]
            positive_embeddings = positive_embedding_cache[(doc_variant, pooling)]
            negative_embeddings = negative_embedding_cache[(doc_variant, pooling)]
            for similarity in args.similarities:
                metrics = _evaluate_configuration(
                    query_embeddings=query_embeddings,
                    positive_embeddings=positive_embeddings,
                    negative_embeddings=negative_embeddings,
                    similarity=str(similarity),
                )
                metrics.update(
                    {
                        "query_variant": query_variant,
                        "doc_variant": doc_variant,
                        "pooling": pooling,
                        "similarity": similarity,
                    }
                )
                results.append(metrics)
    logger.info(
        "Scored %d configurations in %.2fs.",
        len(results),
        time.time() - score_start,
    )

    results.sort(
        key=lambda item: (
            float(item["mrr"]),
            float(item["recall_10"]),
            float(item["positive_gt_all_negatives_frac"]),
        ),
        reverse=True,
    )
    top_n: int = max(1, int(args.top_n))
    logger.info(
        "Resolved %d sampled pairs and %d shared negatives.",
        len(sampled_pairs),
        len(negative_doc_ids),
    )
    logger.info("Top %d configurations by sampled MRR:", top_n)
    for result in results[:top_n]:
        logger.info(
            "query=%s doc=%s pooling=%s sim=%s mrr=%s recall@10=%s "
            "recall@32=%s pos>all_neg=%s pos_mean=%s neg_mean=%s",
            result["query_variant"],
            result["doc_variant"],
            result["pooling"],
            result["similarity"],
            _format_metric(result, "mrr"),
            _format_metric(result, "recall_10"),
            _format_metric(result, "recall_32"),
            _format_metric(result, "positive_gt_all_negatives_frac"),
            _format_metric(result, "positive_score_mean"),
            _format_metric(result, "negative_score_mean"),
        )
    logger.info("Bottom %d configurations by sampled MRR:", top_n)
    for result in results[-top_n:]:
        logger.info(
            "query=%s doc=%s pooling=%s sim=%s mrr=%s recall@10=%s "
            "recall@32=%s pos>all_neg=%s pos_mean=%s neg_mean=%s",
            result["query_variant"],
            result["doc_variant"],
            result["pooling"],
            result["similarity"],
            _format_metric(result, "mrr"),
            _format_metric(result, "recall_10"),
            _format_metric(result, "recall_32"),
            _format_metric(result, "positive_gt_all_negatives_frac"),
            _format_metric(result, "positive_score_mean"),
            _format_metric(result, "negative_score_mean"),
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model_config": args.model_config,
        "checkpoint_path": args.checkpoint_path,
        "queries_path": args.queries_path,
        "qrels_path": args.qrels_path,
        "corpus_glob": args.corpus_glob,
        "doc_ids_json": args.doc_ids_json,
        "sample_size": len(sampled_pairs),
        "negative_pool_size": len(negative_doc_ids),
        "variant_pairs": list(args.variant_pairs),
        "poolings": list(args.poolings),
        "similarities": list(args.similarities),
        "top_results": results[:top_n],
        "bottom_results": results[-top_n:],
        "all_results": results,
        "sampled_pairs_preview": [
            {
                "query_id": pair.query_id,
                "positive_doc_id": pair.positive_doc_id,
            }
            for pair in sampled_pairs[:10]
        ],
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved diagnostic report to %s", output_path)


if __name__ == "__main__":
    main()
