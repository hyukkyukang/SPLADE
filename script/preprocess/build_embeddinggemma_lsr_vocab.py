import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

from src.prototype.embeddinggemma_lsr.data import (
    build_text_pairs,
    collect_required_ids,
    column_names_of,
    load_hf_splits,
    lookup_texts_by_ids,
    maybe_concat_datasets,
    resolve_first_present_column,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build target vocabulary for EmbeddingGemma-LSR using lemmatized DF/TF "
            "statistics and BM25-style utility scoring."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional OmegaConf YAML.")

    parser.add_argument("--meta-hf-name", type=str, default=None)
    parser.add_argument("--meta-hf-subset", type=str, default="triplets")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--allow-missing-val-split", action="store_true")
    parser.add_argument("--hf-cache-dir", type=str, default=None)

    parser.add_argument("--meta-query-id-column", type=str, default="query_id")
    parser.add_argument("--meta-positive-id-column", type=str, default="positive_id")
    parser.add_argument("--meta-query-text-column", type=str, default="query")
    parser.add_argument("--meta-positive-text-column", type=str, default="positive")

    parser.add_argument("--query-subset", type=str, default="queries")
    parser.add_argument("--query-split", type=str, default="train")
    parser.add_argument("--query-id-column", type=str, default="query_id")
    parser.add_argument("--query-text-column", type=str, default="query")

    parser.add_argument("--corpus-subset", type=str, default="corpus")
    parser.add_argument("--corpus-split", type=str, default="train")
    parser.add_argument("--corpus-id-column", type=str, default="passage_id")
    parser.add_argument("--corpus-text-column", type=str, default="passage")

    parser.add_argument("--target-size", type=int, default=30000)
    parser.add_argument("--stopword-df-ratio", type=float, default=0.15)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--include-queries", action="store_true")
    parser.add_argument("--max-meta-rows", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)

    parser.add_argument("--spacy-model", type=str, default="en_core_web_trf")
    parser.add_argument("--spacy-batch-size", type=int, default=128)
    parser.add_argument("--spacy-n-process", type=int, default=1)
    parser.add_argument(
        "--normalizer",
        type=str,
        default="spacy",
        choices=["spacy", "simple"],
        help="Token normalization backend for term extraction.",
    )
    parser.add_argument(
        "--allow-simple-fallback",
        action="store_true",
        help=(
            "When spacy backend cannot be loaded, fallback to simple regex token "
            "normalization."
        ),
    )

    parser.add_argument("--output-dir", type=str, default=None)
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


def _validate_required_args(args: argparse.Namespace) -> None:
    required_keys: tuple[str, ...] = ("meta_hf_name", "output_dir")
    key: str
    for key in required_keys:
        value: Any | None = getattr(args, key, None)
        if value is None or not str(value).strip():
            raise ValueError(
                f"Missing required argument `{key}`. "
                "Provide it directly or via --config."
            )


def _normalize_document(doc: Any) -> list[str]:
    tokens: list[str] = []
    token: Any
    for token in doc:
        if bool(token.is_space) or bool(token.is_punct):
            continue
        lemma: str = str(token.lemma_).strip()
        if not lemma or lemma == "-PRON-":
            lemma = str(token.text).strip()
        if not lemma:
            continue
        if str(token.ent_iob_) in {"B", "I"}:
            normalized: str = lemma
        else:
            normalized = lemma.lower()
        if normalized:
            tokens.append(normalized)
    return tokens


_SIMPLE_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9_]+")


def _normalize_text_simple(text: str) -> list[str]:
    return [token.lower() for token in _SIMPLE_TOKEN_PATTERN.findall(text)]


def _iter_document_tokens_simple(
    *,
    texts: Iterable[str],
    max_docs: int | None,
) -> tuple[list[list[str]], dict[str, Any]]:
    docs_tokens: list[list[str]] = []
    seen: int = 0
    text: str
    for text in texts:
        normalized: list[str] = _normalize_text_simple(str(text))
        if normalized:
            docs_tokens.append(normalized)
            seen += 1
        if max_docs is not None and seen >= int(max_docs):
            break
    stats: dict[str, Any] = {
        "docs_with_tokens": seen,
        "normalizer": "simple",
    }
    return docs_tokens, stats


def _resolve_text_corpus(
    *,
    args: argparse.Namespace,
    meta_dataset: Any,
) -> tuple[list[str], dict[str, Any]]:
    columns: list[str] = column_names_of(meta_dataset)

    query_text_col: str | None = resolve_first_present_column(
        columns,
        [args.meta_query_text_column, "query", "question", "query_text"],
    )
    positive_text_col: str | None = resolve_first_present_column(
        columns,
        [args.meta_positive_text_column, "positive", "passage", "doc", "positive_text"],
    )

    mode: str = "id_lookup"
    documents: list[str] = []

    if query_text_col is not None and positive_text_col is not None:
        mode = "direct_text"
        pairs = build_text_pairs(
            meta_dataset=meta_dataset,
            query_text_column=query_text_col,
            positive_text_column=positive_text_col,
            query_id_column=args.meta_query_id_column,
            positive_id_column=args.meta_positive_id_column,
            query_lookup=None,
            corpus_lookup=None,
            max_pairs=args.max_meta_rows,
        )
        documents.extend(pair.positive for pair in pairs)
        if bool(args.include_queries):
            documents.extend(pair.query for pair in pairs)
        return documents, {"mode": mode, "pairs_collected": len(pairs)}

    query_ids, positive_ids, rows_seen = collect_required_ids(
        meta_dataset=meta_dataset,
        query_id_column=args.meta_query_id_column,
        positive_id_column=args.meta_positive_id_column,
        max_rows=args.max_meta_rows,
    )

    query_datasets = load_hf_splits(
        hf_name=args.meta_hf_name,
        hf_subset=args.query_subset,
        splits=[args.query_split],
        cache_dir=args.hf_cache_dir,
        data_files=None,
        allow_missing_split=False,
    )
    corpus_datasets = load_hf_splits(
        hf_name=args.meta_hf_name,
        hf_subset=args.corpus_subset,
        splits=[args.corpus_split],
        cache_dir=args.hf_cache_dir,
        data_files=None,
        allow_missing_split=False,
    )

    query_dataset = maybe_concat_datasets(query_datasets)
    corpus_dataset = maybe_concat_datasets(corpus_datasets)

    query_lookup: dict[str, str] = {}
    if bool(args.include_queries):
        query_lookup = lookup_texts_by_ids(
            dataset=query_dataset,
            id_column=args.query_id_column,
            text_column=args.query_text_column,
            wanted_ids=query_ids,
        )
    corpus_lookup: dict[str, str] = lookup_texts_by_ids(
        dataset=corpus_dataset,
        id_column=args.corpus_id_column,
        text_column=args.corpus_text_column,
        wanted_ids=positive_ids,
    )

    documents.extend(corpus_lookup.values())
    if bool(args.include_queries):
        documents.extend(query_lookup.values())

    info: dict[str, Any] = {
        "mode": mode,
        "meta_rows_seen": rows_seen,
        "query_ids": len(query_ids),
        "positive_ids": len(positive_ids),
        "resolved_query_texts": len(query_lookup),
        "resolved_positive_texts": len(corpus_lookup),
    }
    return documents, info


def _iter_document_tokens(
    *,
    texts: Iterable[str],
    spacy_model: str,
    batch_size: int,
    n_process: int,
    max_docs: int | None,
    normalizer: str,
    allow_simple_fallback: bool,
) -> tuple[list[list[str]], dict[str, Any]]:
    if str(normalizer).lower() == "simple":
        return _iter_document_tokens_simple(texts=texts, max_docs=max_docs)

    try:
        import spacy
    except ImportError as exc:
        if bool(allow_simple_fallback):
            docs_tokens, stats = _iter_document_tokens_simple(
                texts=texts,
                max_docs=max_docs,
            )
            stats["fallback_reason"] = (
                "spacy import failed; using simple regex normalization."
            )
            return docs_tokens, stats
        raise RuntimeError(
            "spaCy is required. Install package `spacy` and an English model (e.g., "
            "`python -m spacy download en_core_web_trf`) or rerun with "
            "--allow-simple-fallback."
        ) from exc

    try:
        nlp = spacy.load(spacy_model)
    except Exception as exc:
        if bool(allow_simple_fallback):
            docs_tokens, stats = _iter_document_tokens_simple(
                texts=texts,
                max_docs=max_docs,
            )
            stats["fallback_reason"] = (
                f"spacy model load failed ({exc!r}); using simple regex normalization."
            )
            return docs_tokens, stats
        raise RuntimeError(
            f"Failed to load spaCy model {spacy_model!r}. "
            "Install the model or rerun with --allow-simple-fallback."
        ) from exc
    docs_tokens: list[list[str]] = []
    seen: int = 0

    for doc in nlp.pipe(texts, batch_size=int(batch_size), n_process=int(n_process)):
        normalized: list[str] = _normalize_document(doc)
        if normalized:
            docs_tokens.append(normalized)
            seen += 1
        if max_docs is not None and seen >= int(max_docs):
            break

    stats: dict[str, Any] = {
        "docs_with_tokens": seen,
        "spacy_model": spacy_model,
        "spacy_batch_size": int(batch_size),
        "spacy_n_process": int(n_process),
        "normalizer": "spacy",
    }
    return docs_tokens, stats


def _build_vocab(
    *,
    docs_tokens: list[list[str]],
    target_size: int,
    min_df: int,
    stopword_df_ratio: float,
) -> tuple[list[str], dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    df_counter: Counter[str] = Counter()
    tf_total_counter: Counter[str] = Counter()

    doc_count: int = 0
    for tokens in docs_tokens:
        doc_count += 1
        tf_doc: Counter[str] = Counter(tokens)
        tf_total_counter.update(tf_doc)
        df_counter.update(tf_doc.keys())

    if doc_count <= 0:
        raise RuntimeError("No valid documents were collected for vocabulary construction.")

    max_df_threshold: float = float(stopword_df_ratio) * float(doc_count)
    candidate_terms: list[str] = [
        term
        for term, df_value in df_counter.items()
        if int(df_value) >= int(min_df) and float(df_value) <= max_df_threshold
    ]

    utility_by_term: dict[str, float] = {}
    term: str
    for term in candidate_terms:
        df_value: int = int(df_counter[term])
        tf_total: int = int(tf_total_counter[term])
        idf: float = math.log(
            ((float(doc_count) - float(df_value) + 0.5) / (float(df_value) + 0.5)) + 1.0
        )
        utility_by_term[term] = float(tf_total) * idf

    ranked_terms: list[str] = sorted(
        candidate_terms,
        key=lambda token: (-utility_by_term[token], token),
    )

    selected_terms: list[str] = ranked_terms[: int(target_size)]
    selected_df_map: dict[str, int] = {term: int(df_counter[term]) for term in selected_terms}

    selected_stats: list[dict[str, Any]] = []
    rank: int
    for rank, term in enumerate(selected_terms, start=1):
        selected_stats.append(
            {
                "rank": rank,
                "term": term,
                "df": int(df_counter[term]),
                "tf_total": int(tf_total_counter[term]),
                "utility": float(utility_by_term[term]),
            }
        )

    summary: dict[str, Any] = {
        "doc_count": doc_count,
        "unique_terms": len(df_counter),
        "candidate_terms": len(candidate_terms),
        "target_size": int(target_size),
        "selected_terms": len(selected_terms),
        "max_df_threshold": max_df_threshold,
        "min_df": int(min_df),
    }
    return selected_terms, selected_df_map, selected_stats, summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args = _apply_config_overrides(args)
    _validate_required_args(args)

    meta_datasets = load_hf_splits(
        hf_name=args.meta_hf_name,
        hf_subset=args.meta_hf_subset,
        splits=[args.train_split, args.val_split],
        cache_dir=args.hf_cache_dir,
        data_files=None,
        allow_missing_split=bool(args.allow_missing_val_split),
    )
    meta_dataset = maybe_concat_datasets(meta_datasets)

    raw_texts, source_stats = _resolve_text_corpus(args=args, meta_dataset=meta_dataset)
    docs_tokens, spacy_stats = _iter_document_tokens(
        texts=raw_texts,
        spacy_model=args.spacy_model,
        batch_size=args.spacy_batch_size,
        n_process=args.spacy_n_process,
        max_docs=args.max_docs,
        normalizer=args.normalizer,
        allow_simple_fallback=bool(args.allow_simple_fallback),
    )

    v_target, df_map, selected_stats, summary = _build_vocab(
        docs_tokens=docs_tokens,
        target_size=args.target_size,
        min_df=args.min_df,
        stopword_df_ratio=args.stopword_df_ratio,
    )

    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "v_target.txt").write_text("\n".join(v_target) + "\n", encoding="utf-8")
    (output_dir / "df_map.json").write_text(
        json.dumps(df_map, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "vocab_stats.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "selected_terms": selected_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "source_stats": source_stats,
                "spacy_stats": spacy_stats,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved vocabulary artifacts to {output_dir}")
    print(f"Selected terms: {len(v_target)}")


if __name__ == "__main__":
    main()
