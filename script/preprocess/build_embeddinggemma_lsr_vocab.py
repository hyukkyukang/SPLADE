import argparse
import concurrent.futures as cf
import itertools
import json
import math
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset
from omegaconf import OmegaConf

from src.prototype.embeddinggemma_lsr.data import (
    build_text_pairs,
    collect_required_ids,
    column_names_of,
    load_hf_split,
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
    parser.add_argument(
        "--filter-stopwords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter stopwords during candidate construction before DF and utility "
            "ranking."
        ),
    )
    parser.add_argument(
        "--stopword-list-path",
        type=str,
        default=None,
        help=(
            "Optional newline-delimited stopword file. If omitted, use spaCy English "
            "stopwords with built-in fallback."
        ),
    )
    parser.add_argument(
        "--stopword-filter-phrases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When true, drop multi-token terms when all tokens are stopwords. "
            "Unigram stopwords are always filtered when filter_stopwords=true."
        ),
    )
    parser.add_argument(
        "--normalize-leading-determiners",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize phrase candidates by stripping leading determiners (for example "
            "'the', 'a', 'an')."
        ),
    )
    parser.add_argument(
        "--leading-determiners",
        nargs="*",
        default=["the", "a", "an"],
        help=(
            "Determiner tokens to strip from the beginning of phrase candidates when "
            "normalize_leading_determiners=true."
        ),
    )
    parser.add_argument(
        "--normalize-entity-determiners",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply leading-determiner normalization to entity phrases.",
    )
    parser.add_argument(
        "--normalize-noun-chunk-determiners",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply leading-determiner normalization to noun chunk phrases.",
    )
    parser.add_argument(
        "--filter-noise-terms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter noisy terms (mojibake, numeric-heavy, symbol-heavy, single-char) "
            "during candidate construction."
        ),
    )
    parser.add_argument(
        "--noise-max-digit-ratio",
        type=float,
        default=0.7,
        help="Drop terms whose digit ratio exceeds this threshold.",
    )
    parser.add_argument(
        "--noise-max-symbol-ratio",
        type=float,
        default=0.35,
        help="Drop terms whose symbol ratio exceeds this threshold.",
    )
    parser.add_argument(
        "--noise-drop-single-char",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop single-character terms (alphabetic or non-alphanumeric). "
            "Single-digit numeric terms are handled by numeric filtering."
        ),
    )
    parser.add_argument(
        "--noise-drop-pure-numeric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop terms that are purely numeric (including simple separators).",
    )
    parser.add_argument(
        "--noise-drop-mojibake",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop terms with mojibake patterns (for example malformed UTF-8 text).",
    )
    parser.add_argument(
        "--filter-url-like-terms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop URL/domain-like term candidates (for example weather.com, "
            "http, www, .pdf)."
        ),
    )
    parser.add_argument(
        "--filter-template-terms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop boilerplate/template term candidates (for example 'this web "
            "site', 'job posting', 'fact sheet')."
        ),
    )
    parser.add_argument(
        "--filter-function-leading-phrases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop multi-token phrases whose first token is a configurable "
            "function/quantifier word."
        ),
    )
    parser.add_argument(
        "--function-leading-words",
        nargs="*",
        default=[
            "this",
            "that",
            "these",
            "those",
            "any",
            "some",
            "many",
            "much",
            "more",
            "most",
            "less",
            "least",
            "few",
            "fewer",
            "all",
            "each",
            "every",
            "either",
            "neither",
            "another",
            "other",
            "such",
            "same",
            "no",
            "none",
            "both",
            "several",
            "various",
            "certain",
            "particular",
        ],
        help=(
            "Word list used by filter_function_leading_phrases to detect weak "
            "phrase openings."
        ),
    )
    parser.add_argument(
        "--function-leading-require-noun-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When true, apply function-leading phrase filtering only to terms "
            "that include noun_chunk provenance."
        ),
    )
    parser.add_argument(
        "--function-leading-keep-entity-backed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When true, preserve function-leading phrases that also have entity "
            "provenance."
        ),
    )
    parser.add_argument(
        "--filter-contraction-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop canonicalized split contraction/possessive artifacts (for example "
            "'it s', 'you re', 'don t', 'source s')."
        ),
    )
    parser.add_argument(
        "--filter-structured-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop additional structured artifacts after canonicalization such as "
            "HTML entity tokens, metadata score suffixes, pronoun-led phrases, "
            "and code-like letter-number phrases."
        ),
    )
    parser.add_argument(
        "--html-entity-blacklist",
        nargs="*",
        default=["amp", "nbsp", "lt", "gt"],
        help=(
            "Lowercased HTML/entity artifact tokens to drop when "
            "filter_structured_artifacts=true."
        ),
    )
    parser.add_argument(
        "--filter-html-entity-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When true, structured artifact filtering removes HTML/entity tokens, "
            "numeric apostrophe artifacts, and metadata score suffix patterns."
        ),
    )
    parser.add_argument(
        "--filter-pronoun-led-phrases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When true, drop multi-token phrases that begin with pronoun-like "
            "determiners (for example, 'your', 'our')."
        ),
    )
    parser.add_argument(
        "--pronoun-leading-words",
        nargs="*",
        default=["your", "my", "our", "their", "his", "her", "its"],
        help=(
            "Word list used by filter_pronoun_led_phrases to detect weak "
            "pronoun-led phrase openings."
        ),
    )
    parser.add_argument(
        "--filter-letter-number-phrases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop two-token code-like phrases matching letter+number patterns "
            "(for example, 'x 2') unless explicitly whitelisted."
        ),
    )
    parser.add_argument(
        "--letter-number-phrase-whitelist",
        nargs="*",
        default=[
            "w 2",
            "w 4",
            "b 12",
            "k 12",
            "i 95",
            "i 90",
            "i 80",
            "i 75",
            "i 70",
            "i 40",
            "i 35",
            "i 20",
            "i 15",
            "i 10",
            "i 9",
            "i 5",
            "i 94",
            "k 1",
            "b 1",
            "b 2",
            "b 6",
            "r 22",
            "r 3",
            "f 1",
            "v 6",
            "v 8",
            "c 1",
            "m 2",
        ],
        help=(
            "Whitelist exceptions for filter_letter_number_phrases. Values are "
            "canonicalized with the same tokenizer used for selection."
        ),
    )
    parser.add_argument(
        "--filter-closed-class-terms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop terms listed in closed_class_words after canonicalization. "
            "Useful for pruning residual function words beyond stopwords."
        ),
    )
    parser.add_argument(
        "--closed-class-words",
        nargs="*",
        default=[
            "may",
            "will",
            "must",
            "might",
            "shall",
            "could",
            "would",
            "should",
            "can",
            "cannot",
            "many",
            "much",
            "more",
            "most",
            "less",
            "least",
            "every",
            "another",
            "either",
            "neither",
            "none",
            "several",
            "various",
            "certain",
            "particular",
            "mine",
            "onto",
            "else",
            "done",
            "per",
            "im",
        ],
        help=(
            "Closed-class blacklist applied by filter_closed_class_terms. Values "
            "should be lowercased canonical forms."
        ),
    )
    parser.add_argument(
        "--filter-high-df-unigrams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop unigrams whose document-frequency ratio is above the configured "
            "high_df_unigram_ratio threshold."
        ),
    )
    parser.add_argument(
        "--high-df-unigram-ratio",
        type=float,
        default=0.08,
        help=(
            "Hard DF-ratio cap for unigrams when filter_high_df_unigrams=true."
        ),
    )
    parser.add_argument(
        "--filter-generic-unigram-blacklist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop unigram terms listed in generic_unigram_blacklist.",
    )
    parser.add_argument(
        "--generic-unigram-blacklist",
        nargs="*",
        default=[
            "use",
            "one",
            "also",
            "make",
            "get",
            "take",
            "find",
            "know",
            "like",
            "good",
            "well",
            "call",
            "work",
            "need",
            "first",
            "include",
            "just",
            "go",
        ],
        help=(
            "Generic unigram blacklist applied when "
            "filter_generic_unigram_blacklist=true."
        ),
    )
    parser.add_argument(
        "--filter-pos-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter candidate terms by POS tag before utility ranking.",
    )
    parser.add_argument(
        "--pos-gate-allowed-tags",
        nargs="*",
        default=["NOUN", "VERB", "ADJ"],
        help=(
            "Allowed universal POS tags for candidate terms when filter_pos_gate=true."
        ),
    )
    parser.add_argument(
        "--pos-gate-batch-size",
        type=int,
        default=2048,
        help="Batch size used by NLTK POS tagging during POS gate filtering.",
    )
    parser.add_argument(
        "--filter-noisy-numeric-terms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When true, apply a numeric quality gate for NUM-tagged candidates to "
            "drop noisy mixed alphanumeric forms."
        ),
    )
    parser.add_argument(
        "--numeric-term-max-tokens",
        type=int,
        default=3,
        help=(
            "Maximum token count for clean numeric terms retained by the numeric "
            "quality gate."
        ),
    )
    parser.add_argument(
        "--normalize-noun-forms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize NOUN terms to a canonical singular form using a high-precision "
            "hybrid agreement strategy before ranking."
        ),
    )
    parser.add_argument(
        "--noun-normalization-skip-entity-backed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip noun-form normalization for terms that include entity provenance."
        ),
    )
    parser.add_argument(
        "--noun-normalization-include-phrases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When true, apply noun-head singularization to multi-token phrases. "
            "Keeping this false is much faster and still normalizes unigram "
            "singular/plural redundancy."
        ),
    )
    parser.add_argument(
        "--noun-normalization-exceptions",
        nargs="*",
        default=[
            "news",
            "series",
            "species",
            "means",
            "headquarters",
            "politics",
            "economics",
            "mathematics",
            "physics",
            "ethics",
            "statistics",
            "diabetes",
            "measles",
        ],
        help=(
            "Lowercased noun forms to exempt from singular/plural normalization."
        ),
    )
    parser.add_argument(
        "--canonicalize-terms-for-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Canonicalize terms (case/punctuation normalization) and merge "
            "statistics before ranking."
        ),
    )
    parser.add_argument(
        "--canonical-strip-leading-determiners",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When canonicalization is enabled, strip leading determiners from "
            "canonical forms."
        ),
    )
    parser.add_argument(
        "--downweight-generic-unigrams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply a DF-ratio based penalty to very high-DF unigrams so they are "
            "less likely to dominate the top ranks."
        ),
    )
    parser.add_argument(
        "--generic-unigram-df-ratio-start",
        type=float,
        default=0.02,
        help=(
            "Start applying unigram DF-ratio penalty at this document-frequency ratio."
        ),
    )
    parser.add_argument(
        "--generic-unigram-min-multiplier",
        type=float,
        default=0.35,
        help=(
            "Minimum utility multiplier for generic unigram downweighting at the "
            "high-DF end."
        ),
    )
    parser.add_argument(
        "--generic-unigram-penalty-power",
        type=float,
        default=1.0,
        help=(
            "Penalty curve exponent for generic unigram downweighting. Values >1 "
            "make the penalty steeper near the high-DF end."
        ),
    )
    parser.add_argument(
        "--filter-low-cohesion-phrases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter weak multi-token phrases using a PMI-like cohesion score."
        ),
    )
    parser.add_argument(
        "--phrase-cohesion-method",
        type=str,
        choices=["npmi", "pmi"],
        default="npmi",
        help="Scoring method for phrase cohesion filtering.",
    )
    parser.add_argument(
        "--phrase-cohesion-min-score",
        type=float,
        default=-0.05,
        help="Minimum cohesion score to keep a phrase candidate.",
    )
    parser.add_argument(
        "--phrase-cohesion-min-df",
        type=int,
        default=20,
        help="Minimum phrase DF required before applying cohesion gating.",
    )
    parser.add_argument(
        "--phrase-cohesion-require-noun-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply cohesion filtering only when noun_chunk provenance is present."
        ),
    )
    parser.add_argument(
        "--phrase-cohesion-entity-exempt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip cohesion filtering for phrases with entity provenance.",
    )
    parser.add_argument(
        "--entity-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter low-quality entity-backed terms using entity source support "
            "thresholds."
        ),
    )
    parser.add_argument(
        "--entity-quality-min-source-df",
        type=int,
        default=5,
        help="Minimum entity-source DF for keeping weakly entity-backed terms.",
    )
    parser.add_argument(
        "--entity-quality-min-source-ratio",
        type=float,
        default=0.02,
        help="Minimum entity-source DF ratio for mixed-source terms.",
    )
    parser.add_argument(
        "--entity-quality-min-df-entity-only",
        type=int,
        default=30,
        help="Minimum DF for terms that are entity-only after canonicalization.",
    )
    parser.add_argument(
        "--term-stats-cache-path",
        type=str,
        default=None,
        help=(
            "Optional path for persisted extraction statistics. Defaults to "
            "<output_dir>/term_statistics.pkl."
        ),
    )
    parser.add_argument(
        "--save-term-stats-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist extracted DF/TF/source statistics for later selection-only reruns.",
    )
    parser.add_argument(
        "--selection-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip extraction and rebuild vocabulary selection from cached term "
            "statistics."
        ),
    )
    parser.add_argument("--include-queries", action="store_true")
    parser.add_argument("--max-meta-rows", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--use-all-corpus-documents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, ignore positive-id lookup from triplet/meta rows and stream "
            "every document from the configured corpus split."
        ),
    )
    parser.add_argument(
        "--map-reduce-sharding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable map-reduce sharding over the full corpus. Each shard is processed "
            "by an independent process and reduced into global DF/TF statistics."
        ),
    )
    parser.add_argument(
        "--map-reduce-num-shards",
        type=int,
        default=0,
        help=(
            "Number of corpus shards for map-reduce. When <= 0 and map-reduce is "
            "enabled, defaults to spacy_n_process."
        ),
    )
    parser.add_argument(
        "--map-reduce-num-workers",
        type=int,
        default=0,
        help=(
            "Number of worker processes for map-reduce. When <= 0 and map-reduce is "
            "enabled, defaults to map_reduce_num_shards."
        ),
    )
    parser.add_argument(
        "--map-reduce-tmp-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for per-shard map artifacts. Defaults to "
            "<output_dir>/.map_reduce_tmp."
        ),
    )
    parser.add_argument(
        "--map-reduce-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete per-shard map artifacts after reduce completes.",
    )

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
    parser.add_argument(
        "--spacy-extract-entities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using spaCy normalization, include NER spans as term candidates.",
    )
    parser.add_argument(
        "--spacy-extract-noun-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When using spaCy normalization, include noun chunks as phrase "
            "term candidates."
        ),
    )
    parser.add_argument(
        "--entity-labels",
        nargs="*",
        default=None,
        help=(
            "Optional spaCy entity labels to include (for example PERSON ORG GPE). "
            "Empty or omitted keeps all labels."
        ),
    )
    parser.add_argument("--entity-min-tokens", type=int, default=1)
    parser.add_argument("--entity-max-tokens", type=int, default=6)
    parser.add_argument("--noun-chunk-min-tokens", type=int, default=2)
    parser.add_argument("--noun-chunk-max-tokens", type=int, default=6)
    parser.add_argument(
        "--noun-chunk-normalization",
        type=str,
        choices=["surface", "lemma"],
        default="surface",
        help=(
            "Normalization strategy for noun chunks. 'surface' is lighter than "
            "full lemmatization."
        ),
    )
    parser.add_argument("--noun-chunk-max-stopword-ratio", type=float, default=0.4)
    parser.add_argument("--max-phrase-chars", type=int, default=80)

    parser.add_argument("--token-source-boost", type=float, default=1.0)
    parser.add_argument("--noun-chunk-source-boost", type=float, default=1.25)
    parser.add_argument("--entity-source-boost", type=float, default=1.5)

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
    if int(args.target_size) <= 0:
        raise ValueError("target_size must be > 0.")
    if int(args.min_df) <= 0:
        raise ValueError("min_df must be > 0.")
    if not (0.0 <= float(args.stopword_df_ratio) <= 1.0):
        raise ValueError("stopword_df_ratio must be in [0, 1].")
    if bool(args.normalize_leading_determiners):
        raw = args.leading_determiners
        det_tokens: list[str]
        if isinstance(raw, str):
            det_tokens = [part.strip().lower() for part in re.split(r"[\s,]+", raw) if part.strip()]
        elif isinstance(raw, Iterable):
            det_tokens = [str(part).strip().lower() for part in raw if str(part).strip()]
        else:
            det_tokens = []
        if not det_tokens:
            raise ValueError(
                "normalize_leading_determiners=true requires a non-empty leading_determiners list."
            )
    if bool(args.filter_function_leading_phrases):
        raw_function_words = args.function_leading_words
        function_words: list[str]
        if isinstance(raw_function_words, str):
            function_words = [
                part.strip().lower()
                for part in re.split(r"[\s,]+", raw_function_words)
                if part.strip()
            ]
        elif isinstance(raw_function_words, Iterable):
            function_words = [
                str(part).strip().lower()
                for part in raw_function_words
                if str(part).strip()
            ]
        else:
            function_words = []
        if not function_words:
            raise ValueError(
                "filter_function_leading_phrases=true requires a non-empty "
                "function_leading_words list."
            )
    if bool(args.filter_closed_class_terms):
        raw_closed_class_words = args.closed_class_words
        closed_class_words: list[str]
        if isinstance(raw_closed_class_words, str):
            closed_class_words = [
                part.strip().lower()
                for part in re.split(r"[\s,]+", raw_closed_class_words)
                if part.strip()
            ]
        elif isinstance(raw_closed_class_words, Iterable):
            closed_class_words = [
                str(part).strip().lower()
                for part in raw_closed_class_words
                if str(part).strip()
            ]
        else:
            closed_class_words = []
        if not closed_class_words:
            raise ValueError(
                "filter_closed_class_terms=true requires a non-empty "
                "closed_class_words list."
            )
    if bool(args.filter_structured_artifacts):
        if bool(args.filter_html_entity_artifacts):
            raw_html_entities = args.html_entity_blacklist
            html_entities: list[str]
            if isinstance(raw_html_entities, str):
                html_entities = [
                    part.strip().lower()
                    for part in re.split(r"[\s,]+", raw_html_entities)
                    if part.strip()
                ]
            elif isinstance(raw_html_entities, Iterable):
                html_entities = [
                    str(part).strip().lower()
                    for part in raw_html_entities
                    if str(part).strip()
                ]
            else:
                html_entities = []
            if not html_entities:
                raise ValueError(
                    "filter_html_entity_artifacts=true requires a non-empty "
                    "html_entity_blacklist list."
                )
        if bool(args.filter_pronoun_led_phrases):
            raw_pronoun_words = args.pronoun_leading_words
            pronoun_words: list[str]
            if isinstance(raw_pronoun_words, str):
                pronoun_words = [
                    part.strip().lower()
                    for part in re.split(r"[\s,]+", raw_pronoun_words)
                    if part.strip()
                ]
            elif isinstance(raw_pronoun_words, Iterable):
                pronoun_words = [
                    str(part).strip().lower()
                    for part in raw_pronoun_words
                    if str(part).strip()
                ]
            else:
                pronoun_words = []
            if not pronoun_words:
                raise ValueError(
                    "filter_pronoun_led_phrases=true requires a non-empty "
                    "pronoun_leading_words list."
                )
    if bool(args.filter_high_df_unigrams) and not (
        0.0 <= float(args.high_df_unigram_ratio) <= 1.0
    ):
        raise ValueError("high_df_unigram_ratio must be in [0, 1].")
    if bool(args.filter_generic_unigram_blacklist):
        raw_generic_blacklist = args.generic_unigram_blacklist
        generic_blacklist: list[str]
        if isinstance(raw_generic_blacklist, str):
            generic_blacklist = [
                part.strip().lower()
                for part in re.split(r"[\s,]+", raw_generic_blacklist)
                if part.strip()
            ]
        elif isinstance(raw_generic_blacklist, Iterable):
            generic_blacklist = [
                str(part).strip().lower()
                for part in raw_generic_blacklist
                if str(part).strip()
            ]
        else:
            generic_blacklist = []
        if not generic_blacklist:
            raise ValueError(
                "filter_generic_unigram_blacklist=true requires a non-empty "
                "generic_unigram_blacklist list."
            )
    if int(args.pos_gate_batch_size) <= 0:
        raise ValueError("pos_gate_batch_size must be > 0.")
    if int(args.numeric_term_max_tokens) <= 0:
        raise ValueError("numeric_term_max_tokens must be > 0.")
    if bool(args.filter_pos_gate):
        raw_pos_tags = args.pos_gate_allowed_tags
        pos_tags: list[str]
        if isinstance(raw_pos_tags, str):
            pos_tags = [
                part.strip().upper()
                for part in re.split(r"[\s,]+", raw_pos_tags)
                if part.strip()
            ]
        elif isinstance(raw_pos_tags, Iterable):
            pos_tags = [
                str(part).strip().upper()
                for part in raw_pos_tags
                if str(part).strip()
            ]
        else:
            pos_tags = []
        if not pos_tags:
            raise ValueError(
                "filter_pos_gate=true requires a non-empty pos_gate_allowed_tags list."
            )
        supported_tags: set[str] = {
            "ADJ",
            "ADP",
            "ADV",
            "CONJ",
            "DET",
            "NOUN",
            "NUM",
            "PRON",
            "PRT",
            "VERB",
            "X",
            ".",
        }
        unknown_tags: list[str] = sorted(
            {tag for tag in pos_tags if tag not in supported_tags}
        )
        if unknown_tags:
            raise ValueError(
                "pos_gate_allowed_tags contains unsupported universal tags: "
                f"{unknown_tags}"
            )
    if not (0.0 <= float(args.generic_unigram_df_ratio_start) <= 1.0):
        raise ValueError("generic_unigram_df_ratio_start must be in [0, 1].")
    if not (0.0 < float(args.generic_unigram_min_multiplier) <= 1.0):
        raise ValueError("generic_unigram_min_multiplier must be in (0, 1].")
    if float(args.generic_unigram_penalty_power) <= 0.0:
        raise ValueError("generic_unigram_penalty_power must be > 0.")
    if bool(args.downweight_generic_unigrams) and float(
        args.generic_unigram_df_ratio_start
    ) >= float(args.stopword_df_ratio):
        raise ValueError(
            "generic_unigram_df_ratio_start must be < stopword_df_ratio when "
            "downweight_generic_unigrams=true."
        )
    if int(args.phrase_cohesion_min_df) <= 0:
        raise ValueError("phrase_cohesion_min_df must be > 0.")
    if int(args.entity_quality_min_source_df) <= 0:
        raise ValueError("entity_quality_min_source_df must be > 0.")
    if int(args.entity_quality_min_df_entity_only) <= 0:
        raise ValueError("entity_quality_min_df_entity_only must be > 0.")
    if not (0.0 <= float(args.entity_quality_min_source_ratio) <= 1.0):
        raise ValueError("entity_quality_min_source_ratio must be in [0, 1].")
    if args.stopword_list_path is not None:
        stopword_list_path = Path(str(args.stopword_list_path))
        if not stopword_list_path.exists():
            raise ValueError(
                f"stopword_list_path does not exist: {stopword_list_path}"
            )
        if not stopword_list_path.is_file():
            raise ValueError(
                f"stopword_list_path must be a file: {stopword_list_path}"
            )
    if not (0.0 <= float(args.noise_max_digit_ratio) <= 1.0):
        raise ValueError("noise_max_digit_ratio must be in [0, 1].")
    if not (0.0 <= float(args.noise_max_symbol_ratio) <= 1.0):
        raise ValueError("noise_max_symbol_ratio must be in [0, 1].")
    if int(args.entity_min_tokens) <= 0 or int(args.entity_max_tokens) <= 0:
        raise ValueError("entity min/max tokens must be > 0.")
    if int(args.entity_min_tokens) > int(args.entity_max_tokens):
        raise ValueError("entity_min_tokens must be <= entity_max_tokens.")
    if int(args.noun_chunk_min_tokens) <= 0 or int(args.noun_chunk_max_tokens) <= 0:
        raise ValueError("noun-chunk min/max tokens must be > 0.")
    if int(args.noun_chunk_min_tokens) > int(args.noun_chunk_max_tokens):
        raise ValueError("noun_chunk_min_tokens must be <= noun_chunk_max_tokens.")
    if not (0.0 <= float(args.noun_chunk_max_stopword_ratio) <= 1.0):
        raise ValueError("noun_chunk_max_stopword_ratio must be in [0, 1].")
    if int(args.max_phrase_chars) <= 0:
        raise ValueError("max_phrase_chars must be > 0.")
    if float(args.token_source_boost) <= 0.0:
        raise ValueError("token_source_boost must be > 0.")
    if float(args.noun_chunk_source_boost) <= 0.0:
        raise ValueError("noun_chunk_source_boost must be > 0.")
    if float(args.entity_source_boost) <= 0.0:
        raise ValueError("entity_source_boost must be > 0.")
    if int(args.map_reduce_num_shards) < 0:
        raise ValueError("map_reduce_num_shards must be >= 0.")
    if int(args.map_reduce_num_workers) < 0:
        raise ValueError("map_reduce_num_workers must be >= 0.")
    if bool(args.map_reduce_sharding):
        if not bool(args.use_all_corpus_documents):
            raise ValueError(
                "map_reduce_sharding requires use_all_corpus_documents=true."
            )
        if bool(args.include_queries):
            raise ValueError(
                "map_reduce_sharding currently does not support include_queries=true."
            )


_SOURCE_TOKEN: str = "token"
_SOURCE_ENTITY: str = "entity"
_SOURCE_NOUN_CHUNK: str = "noun_chunk"
_NOUN_CHUNK_ROOT_POS: set[str] = {"NOUN", "PROPN"}
_SIMPLE_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9_]+")
_PURE_NUMERIC_TERM_PATTERN: re.Pattern[str] = re.compile(
    r"[0-9]+(?:[.,:/-][0-9]+)*"
)
_C1_CONTROL_PATTERN: re.Pattern[str] = re.compile(r"[\x80-\x9f]")
_MOJIBAKE_UTF8_LATIN1_PATTERN: re.Pattern[str] = re.compile(
    r"[\u00c2\u00c3\u00e2][\x80-\xbf]"
)
_CANONICAL_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9]+")
_URL_SCHEME_PATTERN: re.Pattern[str] = re.compile(r"\bhttps?\b", re.IGNORECASE)
_URL_DOT_SUFFIX_PATTERN: re.Pattern[str] = re.compile(
    r"\.(?:com|org|net|gov|edu|io|co|ai|app|info|biz|uk|de|fr|jp|cn|ru|br|it|es)\b",
    re.IGNORECASE,
)
_URL_TOKEN_HINTS: frozenset[str] = frozenset(
    {
        "http",
        "https",
        "www",
        "com",
        "org",
        "net",
        "gov",
        "edu",
        "io",
        "co",
        "pdf",
        "html",
        "htm",
        "php",
        "aspx",
        "asp",
    }
)
_TEMPLATE_PHRASE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:web\s+site|zip\s+code|job\s+posting|fact\s+sheet|template\s+message|"
    r"dictionary\s+definition\s+resource|video\s+clip|medication\s+guide)\b",
    re.IGNORECASE,
)
_NUMERIC_DIGIT_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"^[0-9]+$")
_NUMERIC_ORDINAL_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"^[0-9]+(?:st|nd|rd|th)$",
    re.IGNORECASE,
)
_NUMERIC_ROMAN_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"^[ivxlcdm]+$",
    re.IGNORECASE,
)
_ALPHA_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"^[a-z]+$")
_LETTER_NUMBER_PHRASE_PATTERN: re.Pattern[str] = re.compile(r"^[a-z]\s+[0-9]{1,4}[a-z]?$")
_NUMERIC_WORD_TOKENS: frozenset[str] = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "trillion",
    }
)
_IRREGULAR_NOUN_SINGULAR_MAP: dict[str, str] = {
    "children": "child",
    "men": "man",
    "women": "woman",
    "mice": "mouse",
    "geese": "goose",
    "teeth": "tooth",
    "feet": "foot",
    "oxen": "ox",
    "people": "person",
}
_FALLBACK_EN_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

_WORDNET_LEMMATIZER_CACHE: Any | None = None
_WORDNET_LEMMATIZER_AVAILABLE_CACHE: bool | None = None


def _parse_entity_labels(raw_labels: Any) -> set[str] | None:
    if raw_labels is None:
        return None
    labels: list[str] = []
    if isinstance(raw_labels, str):
        text: str = str(raw_labels).strip()
        if not text:
            return None
        labels = [part.strip() for part in re.split(r"[\s,]+", text) if part.strip()]
    elif isinstance(raw_labels, Iterable):
        labels = [str(part).strip() for part in raw_labels if str(part).strip()]
    else:
        return None
    normalized: set[str] = {label.upper() for label in labels}
    return normalized or None


def _parse_leading_determiners(raw_determiners: Any) -> set[str]:
    if raw_determiners is None:
        return set()
    values: list[str] = []
    if isinstance(raw_determiners, str):
        text: str = str(raw_determiners).strip()
        if not text:
            return set()
        values = [part.strip() for part in re.split(r"[\s,]+", text) if part.strip()]
    elif isinstance(raw_determiners, Iterable):
        values = [str(part).strip() for part in raw_determiners if str(part).strip()]
    else:
        return set()
    return {value.lower() for value in values if value.strip()}


def _parse_function_leading_words(raw_words: Any) -> set[str]:
    if raw_words is None:
        return set()
    values: list[str] = []
    if isinstance(raw_words, str):
        text: str = str(raw_words).strip()
        if not text:
            return set()
        values = [part.strip() for part in re.split(r"[\s,]+", text) if part.strip()]
    elif isinstance(raw_words, Iterable):
        values = [str(part).strip() for part in raw_words if str(part).strip()]
    else:
        return set()
    return {value.lower() for value in values if value.strip()}


def _parse_closed_class_words(raw_words: Any) -> set[str]:
    if raw_words is None:
        return set()
    values: list[str] = []
    if isinstance(raw_words, str):
        text: str = str(raw_words).strip()
        if not text:
            return set()
        values = [part.strip() for part in re.split(r"[\s,]+", text) if part.strip()]
    elif isinstance(raw_words, Iterable):
        values = [str(part).strip() for part in raw_words if str(part).strip()]
    else:
        return set()
    return {value.lower() for value in values if value.strip()}


def _normalize_phrase_for_filter(value: str) -> str:
    parts: list[str] = [
        part.lower() for part in _CANONICAL_TOKEN_PATTERN.findall(str(value)) if part
    ]
    return " ".join(parts)


def _parse_html_entity_blacklist(raw_words: Any) -> set[str]:
    return _parse_closed_class_words(raw_words)


def _parse_pronoun_leading_words(raw_words: Any) -> set[str]:
    return _parse_closed_class_words(raw_words)


def _parse_letter_number_phrase_whitelist(raw_values: Any) -> set[str]:
    if raw_values is None:
        return set()
    values: list[str] = []
    if isinstance(raw_values, str):
        text: str = str(raw_values).strip()
        if not text:
            return set()
        values = [part.strip() for part in re.split(r"[,;\n]+", text) if part.strip()]
    elif isinstance(raw_values, Iterable):
        values = [str(part).strip() for part in raw_values if str(part).strip()]
    else:
        return set()
    normalized_values: set[str] = set()
    value: str
    for value in values:
        normalized: str = _normalize_phrase_for_filter(value)
        if normalized:
            normalized_values.add(normalized)
    return normalized_values


def _parse_generic_unigram_blacklist(raw_words: Any) -> set[str]:
    return _parse_closed_class_words(raw_words)


def _parse_noun_normalization_exceptions(raw_words: Any) -> set[str]:
    return _parse_closed_class_words(raw_words)


def _parse_pos_gate_tags(raw_tags: Any) -> set[str]:
    if raw_tags is None:
        return set()
    values: list[str] = []
    if isinstance(raw_tags, str):
        text: str = str(raw_tags).strip()
        if not text:
            return set()
        values = [part.strip() for part in re.split(r"[\s,]+", text) if part.strip()]
    elif isinstance(raw_tags, Iterable):
        values = [str(part).strip() for part in raw_tags if str(part).strip()]
    else:
        return set()
    return {value.upper() for value in values if value.strip()}


def _load_stopwords(
    *,
    stopword_list_path: str | None,
) -> tuple[set[str], str]:
    if stopword_list_path is not None:
        path: Path = Path(str(stopword_list_path))
        stopwords: set[str] = {
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        if not stopwords:
            raise ValueError(
                f"stopword list file is empty after parsing: {path}"
            )
        return stopwords, str(path)

    try:
        from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
    except Exception:
        return set(_FALLBACK_EN_STOPWORDS), "built_in_default"

    stopwords = {str(token).strip().lower() for token in SPACY_STOP_WORDS if str(token).strip()}
    if not stopwords:
        return set(_FALLBACK_EN_STOPWORDS), "built_in_default"
    return stopwords, "spacy.lang.en.stop_words"


def _is_url_like_term(term: str) -> bool:
    normalized: str = str(term).strip()
    if not normalized:
        return False
    lowered: str = normalized.lower()
    if _URL_SCHEME_PATTERN.search(lowered) is not None:
        return True
    if _URL_DOT_SUFFIX_PATTERN.search(lowered) is not None:
        return True
    if "www." in lowered:
        return True
    tokens: list[str] = _CANONICAL_TOKEN_PATTERN.findall(lowered)
    if not tokens:
        return False
    token_set: set[str] = set(tokens)
    if "www" in token_set or "http" in token_set or "https" in token_set:
        return True
    if any(token in _URL_TOKEN_HINTS for token in tokens):
        if "." in lowered or "/" in lowered:
            return True
        if len(tokens) <= 2:
            return True
    return False


def _is_template_like_term(term: str) -> bool:
    return _TEMPLATE_PHRASE_PATTERN.search(str(term).strip()) is not None


def _is_stopword_term(
    *,
    term: str,
    stopwords: set[str],
    filter_phrases: bool,
) -> bool:
    normalized: str = str(term).strip().lower()
    if not normalized:
        return False
    parts: list[str] = [part for part in normalized.split() if part]
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in stopwords
    if not bool(filter_phrases):
        return False
    return all(part in stopwords for part in parts)


def _noise_term_reason(
    *,
    term: str,
    max_digit_ratio: float,
    max_symbol_ratio: float,
    drop_single_char: bool,
    drop_pure_numeric: bool,
    drop_mojibake: bool,
    drop_url_like: bool,
    drop_template_like: bool,
) -> str | None:
    normalized: str = str(term).strip()
    if not normalized:
        return "empty"

    compact: str = "".join(normalized.split())
    if not compact:
        return "empty"

    if bool(drop_mojibake):
        if "\ufffd" in normalized:
            return "mojibake_replacement_char"
        if _C1_CONTROL_PATTERN.search(normalized) is not None:
            return "mojibake_c1_control"
        if _MOJIBAKE_UTF8_LATIN1_PATTERN.search(normalized) is not None:
            return "mojibake_utf8_latin1"
    if bool(drop_url_like) and _is_url_like_term(normalized):
        return "url_like"
    if bool(drop_template_like) and _is_template_like_term(normalized):
        return "template_like"

    if bool(drop_single_char) and len(compact) == 1:
        if compact.isalpha():
            return "single_char_alpha"
        if not compact.isalnum():
            return "single_char_non_alnum"

    if bool(drop_pure_numeric) and _PURE_NUMERIC_TERM_PATTERN.fullmatch(compact):
        return "pure_numeric"

    digit_count: int = sum(1 for char in compact if char.isdigit())
    symbol_count: int = sum(1 for char in compact if not char.isalnum())
    compact_len: int = len(compact)
    if compact_len <= 0:
        return "empty"

    if digit_count > 0:
        digit_ratio: float = float(digit_count) / float(compact_len)
        if digit_ratio > float(max_digit_ratio):
            return "digit_heavy"

    if symbol_count > 0:
        alnum_count: int = compact_len - symbol_count
        if alnum_count <= 0:
            return "symbol_only"
        symbol_ratio: float = float(symbol_count) / float(compact_len)
        if symbol_ratio > float(max_symbol_ratio):
            return "symbol_heavy"

    return None


def _canonicalize_term_for_selection(
    *,
    term: str,
    strip_leading_determiners: bool,
    leading_determiners: set[str],
) -> str | None:
    text: str = str(term).strip()
    if not text:
        return None
    text = (
        text.replace("&", " and ")
        .replace("’", "'")
        .replace("`", "'")
        .replace("/", " ")
        .replace("\\", " ")
        .replace("_", " ")
        .replace("-", " ")
    )
    pieces: list[str] = [
        part.lower() for part in _CANONICAL_TOKEN_PATTERN.findall(text) if part
    ]
    if not pieces:
        return None
    if bool(strip_leading_determiners):
        while pieces and pieces[0] in leading_determiners:
            pieces = pieces[1:]
    if not pieces:
        return None
    return " ".join(pieces)


def _is_function_leading_phrase(
    *,
    term: str,
    sources: set[str],
    function_leading_words: set[str],
    require_noun_chunk_source: bool,
    keep_entity_backed: bool,
) -> bool:
    normalized: str = str(term).strip().lower()
    if not normalized:
        return False
    parts: list[str] = [part for part in normalized.split() if part]
    if len(parts) < 2:
        return False
    if bool(require_noun_chunk_source) and _SOURCE_NOUN_CHUNK not in sources:
        return False
    if bool(keep_entity_backed) and _SOURCE_ENTITY in sources:
        return False
    return parts[0] in function_leading_words


def _structured_artifact_reason(
    *,
    term: str,
    filter_html_entity_artifacts: bool,
    html_entity_blacklist: set[str],
    filter_pronoun_led_phrases: bool,
    pronoun_leading_words: set[str],
    filter_letter_number_phrases: bool,
    letter_number_phrase_whitelist: set[str],
) -> str | None:
    normalized: str = _normalize_phrase_for_filter(term)
    if not normalized:
        return None
    parts: list[str] = [part for part in normalized.split() if part]
    if not parts:
        return None

    if (
        bool(filter_html_entity_artifacts)
        and len(parts) == 1
        and normalized in html_entity_blacklist
    ):
        return "html_entity_token"
    if (
        bool(filter_html_entity_artifacts)
        and re.search(r"\bit\s+39\s+s\b", normalized) is not None
    ):
        return "numeric_apostrophe_artifact"
    if (
        bool(filter_html_entity_artifacts)
        and len(parts) >= 3
        and "0 00" in normalized
        and any(part.isalpha() for part in parts)
    ):
        return "metadata_score_suffix"

    if (
        bool(filter_pronoun_led_phrases)
        and len(parts) >= 2
        and parts[0] in pronoun_leading_words
    ):
        return "pronoun_led_phrase"

    if (
        bool(filter_letter_number_phrases)
        and len(parts) == 2
        and _LETTER_NUMBER_PHRASE_PATTERN.fullmatch(normalized) is not None
        and normalized not in letter_number_phrase_whitelist
    ):
        return "letter_number_phrase"
    return None


def _contraction_artifact_reason(*, term: str) -> str | None:
    normalized: str = str(term).strip().lower()
    if not normalized:
        return None
    parts: list[str] = [part for part in normalized.split() if part]
    if len(parts) != 2:
        return None
    first: str = parts[0]
    second: str = parts[1]
    if not first.isalpha() or not second.isalpha():
        return None

    pronoun_heads: set[str] = {
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "it",
        "there",
        "here",
        "what",
        "who",
        "that",
        "let",
    }
    if second in {"s", "re", "ve", "ll", "d", "m"}:
        if first in pronoun_heads:
            return "split_contraction"
        if second == "s" and len(first) >= 2:
            return "split_possessive"
        return None

    if second == "t":
        negation_heads: set[str] = {
            "can",
            "don",
            "doesn",
            "didn",
            "isn",
            "aren",
            "wasn",
            "weren",
            "won",
            "wouldn",
            "couldn",
            "shouldn",
            "mustn",
            "hasn",
            "hadn",
            "haven",
        }
        if first in negation_heads:
            return "split_negation"
    return None


def _is_closed_class_blacklisted_term(
    *,
    term: str,
    closed_class_words: set[str],
) -> bool:
    normalized: str = str(term).strip().lower()
    if not normalized:
        return False
    return normalized in closed_class_words


def _infer_phrase_head_universal_pos(tags: list[tuple[str, str]]) -> str:
    if not tags:
        return "X"
    for target in ("NOUN", "VERB", "ADJ", "ADV", "NUM"):
        for _token, pos in reversed(tags):
            if pos == target:
                return pos
    for _token, pos in reversed(tags):
        if pos != ".":
            return pos
    return tags[-1][1]


def _tag_terms_universal_pos(
    *,
    terms: list[str],
    batch_size: int,
) -> dict[str, str]:
    if not terms:
        return {}
    try:
        import nltk
    except Exception as exc:
        raise RuntimeError(
            "POS gate requires NLTK. Install it with `python -m pip install nltk`."
        ) from exc

    unigrams: list[str] = []
    phrases: list[str] = []
    term: str
    for term in terms:
        pieces: list[str] = [part for part in str(term).split() if part]
        if len(pieces) <= 1:
            unigrams.append(str(term))
        else:
            phrases.append(str(term))

    pos_by_term: dict[str, str] = {}
    if unigrams:
        start: int
        for start in range(0, len(unigrams), int(batch_size)):
            unigram_batch: list[str] = unigrams[start : start + int(batch_size)]
            unigram_sequences: list[list[str]] = [[token] for token in unigram_batch]
            try:
                tagged_unigrams = nltk.pos_tag_sents(
                    unigram_sequences,
                    tagset="universal",
                )
            except LookupError as exc:
                raise RuntimeError(
                    "NLTK POS resources are missing. Run:\n"
                    "python - <<'PY'\n"
                    "import nltk\n"
                    "nltk.download('averaged_perceptron_tagger', quiet=True)\n"
                    "nltk.download('averaged_perceptron_tagger_eng', quiet=True)\n"
                    "nltk.download('universal_tagset', quiet=True)\n"
                    "PY"
                ) from exc
            sequence: list[tuple[str, str]]
            token: str
            for token, sequence in zip(unigram_batch, tagged_unigrams):
                if not sequence:
                    pos_by_term[token] = "X"
                else:
                    pos_by_term[token] = str(sequence[0][1]).upper()

    if phrases:
        start = 0
        for start in range(0, len(phrases), int(batch_size)):
            phrase_batch: list[str] = phrases[start : start + int(batch_size)]
            phrase_sequences: list[list[str]] = [
                [part for part in phrase.split() if part] for phrase in phrase_batch
            ]
            try:
                tagged_phrases = nltk.pos_tag_sents(
                    phrase_sequences,
                    tagset="universal",
                )
            except LookupError as exc:
                raise RuntimeError(
                    "NLTK POS resources are missing. Run:\n"
                    "python - <<'PY'\n"
                    "import nltk\n"
                    "nltk.download('averaged_perceptron_tagger', quiet=True)\n"
                    "nltk.download('averaged_perceptron_tagger_eng', quiet=True)\n"
                    "nltk.download('universal_tagset', quiet=True)\n"
                    "PY"
                ) from exc
            phrase: str
            phrase_tags: list[tuple[str, str]]
            for phrase, phrase_tags in zip(phrase_batch, tagged_phrases):
                pos_by_term[phrase] = _infer_phrase_head_universal_pos(
                    [(str(tok), str(pos).upper()) for tok, pos in phrase_tags]
                )

    return pos_by_term


def _apply_pos_gate_to_terms(
    *,
    terms: list[str],
    allowed_tags: set[str],
    batch_size: int,
) -> tuple[list[str], Counter[str], Counter[str], dict[str, str]]:
    if not terms:
        return [], Counter(), Counter(), {}
    pos_by_term: dict[str, str] = _tag_terms_universal_pos(
        terms=terms,
        batch_size=int(batch_size),
    )
    kept_terms: list[str] = []
    kept_pos_counts: Counter[str] = Counter()
    filtered_pos_counts: Counter[str] = Counter()
    term: str
    for term in terms:
        pos_tag: str = str(pos_by_term.get(term, "X")).upper()
        if pos_tag in allowed_tags:
            kept_terms.append(term)
            kept_pos_counts[pos_tag] += 1
        else:
            filtered_pos_counts[pos_tag] += 1
    return kept_terms, kept_pos_counts, filtered_pos_counts, pos_by_term


def _is_clean_numeric_token(token: str) -> bool:
    normalized: str = str(token).strip().lower()
    if not normalized:
        return False
    if normalized in _NUMERIC_WORD_TOKENS:
        return True
    if _NUMERIC_DIGIT_TOKEN_PATTERN.fullmatch(normalized) is not None:
        return True
    if _NUMERIC_ORDINAL_TOKEN_PATTERN.fullmatch(normalized) is not None:
        return True
    if (
        _NUMERIC_ROMAN_TOKEN_PATTERN.fullmatch(normalized) is not None
        and len(normalized) <= 6
    ):
        return True
    return False


def _numeric_term_quality_reason(
    *,
    term: str,
    max_tokens: int,
) -> str | None:
    parts: list[str] = [part for part in str(term).split() if part]
    if not parts:
        return "empty"
    if len(parts) > int(max_tokens):
        return "too_many_tokens"
    if all(_is_clean_numeric_token(part) for part in parts):
        return None
    if any(char.isdigit() for char in str(term)):
        return "mixed_alphanumeric"
    return "noncanonical_numeric_phrase"


def _is_plausible_plural_surface_form(token: str) -> bool:
    value: str = str(token).strip().lower()
    if not value:
        return False
    if value in _IRREGULAR_NOUN_SINGULAR_MAP:
        return True
    if _ALPHA_TOKEN_PATTERN.fullmatch(value) is None:
        return False
    if len(value) <= 3:
        return False
    if value.endswith(("ss", "us", "is")):
        return False
    return value.endswith("s")


def _get_wordnet_lemmatizer() -> Any | None:
    global _WORDNET_LEMMATIZER_CACHE, _WORDNET_LEMMATIZER_AVAILABLE_CACHE
    if _WORDNET_LEMMATIZER_AVAILABLE_CACHE is False:
        return None
    if _WORDNET_LEMMATIZER_CACHE is not None:
        return _WORDNET_LEMMATIZER_CACHE
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.data.find("corpora/wordnet.zip")
        _WORDNET_LEMMATIZER_CACHE = WordNetLemmatizer()
        _WORDNET_LEMMATIZER_AVAILABLE_CACHE = True
        return _WORDNET_LEMMATIZER_CACHE
    except Exception:
        _WORDNET_LEMMATIZER_AVAILABLE_CACHE = False
        _WORDNET_LEMMATIZER_CACHE = None
        return None


def _rule_based_noun_singular(token: str) -> str:
    value: str = str(token).strip().lower()
    if not value:
        return value
    irregular: str | None = _IRREGULAR_NOUN_SINGULAR_MAP.get(value)
    if irregular is not None:
        return irregular
    if len(value) <= 3:
        return value
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if re.search(r"(xes|zes|ches|shes|sses)$", value) is not None and len(value) > 4:
        return value[:-2]
    if value.endswith("s") and not value.endswith(("ss", "us", "is")) and len(value) > 3:
        return value[:-1]
    return value


def _wordnet_noun_singular(token: str) -> str | None:
    lemmatizer: Any | None = _get_wordnet_lemmatizer()
    if lemmatizer is None:
        return None
    try:
        value: str = str(lemmatizer.lemmatize(str(token), "n")).strip().lower()
    except Exception:
        return None
    return value if value else None


def _hybrid_singularize_noun_token(
    *,
    token: str,
    exception_words: set[str],
) -> tuple[str, str]:
    normalized: str = str(token).strip().lower()
    if not normalized:
        return normalized, "empty"
    if normalized in exception_words:
        return normalized, "exception"
    if _ALPHA_TOKEN_PATTERN.fullmatch(normalized) is None:
        return normalized, "non_alpha"
    rule_value: str = _rule_based_noun_singular(normalized)
    wordnet_value: str | None = _wordnet_noun_singular(normalized)
    if wordnet_value is None:
        return normalized, "wordnet_unavailable"
    if not rule_value or not wordnet_value:
        return normalized, "invalid_candidate"
    if rule_value != wordnet_value:
        return normalized, "method_disagreement"
    if rule_value == normalized:
        return normalized, "already_canonical"
    if len(rule_value) < 2:
        return normalized, "too_short"
    return rule_value, "normalized"


def _normalize_noun_forms_with_hybrid_agreement(
    *,
    terms: list[str],
    term_sources: dict[str, set[str]],
    pos_batch_size: int,
    skip_entity_backed: bool,
    include_phrases: bool,
    exception_words: set[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    if not terms:
        return {}, {
            "enabled": True,
            "total_terms": 0,
            "normalized_terms": 0,
            "normalized_unigrams": 0,
            "normalized_phrases": 0,
            "skip_entity_backed": bool(skip_entity_backed),
            "include_phrases": bool(include_phrases),
            "exception_count": int(len(exception_words)),
            "wordnet_available": bool(_get_wordnet_lemmatizer() is not None),
            "reason_counts": {},
        }
    try:
        import nltk
    except Exception as exc:
        raise RuntimeError(
            "Noun-form normalization requires NLTK. Install it with "
            "`python -m pip install nltk`."
        ) from exc

    reason_counts: Counter[str] = Counter()
    normalized_map: dict[str, str] = {}
    unigram_candidates: list[str] = []
    phrase_candidates: list[str] = []
    term: str
    for term in terms:
        sources_for_term: set[str] = set(term_sources.get(term, {_SOURCE_TOKEN}))
        has_non_entity_support: bool = bool(
            {_SOURCE_TOKEN, _SOURCE_NOUN_CHUNK}.intersection(sources_for_term)
        )
        if (
            bool(skip_entity_backed)
            and _SOURCE_ENTITY in sources_for_term
            and not has_non_entity_support
        ):
            reason_counts["skip_entity_backed"] += 1
            continue
        pieces: list[str] = [piece for piece in str(term).split() if piece]
        if not pieces:
            reason_counts["empty"] += 1
            continue
        if len(pieces) == 1:
            if _is_plausible_plural_surface_form(pieces[0]):
                unigram_candidates.append(term)
            else:
                reason_counts["surface_not_plural"] += 1
        else:
            if not bool(include_phrases):
                reason_counts["phrase_disabled"] += 1
                continue
            if _is_plausible_plural_surface_form(pieces[-1]) or any(
                piece.lower() in _IRREGULAR_NOUN_SINGULAR_MAP for piece in pieces
            ):
                phrase_candidates.append(term)
            else:
                reason_counts["phrase_surface_not_plural"] += 1

    if unigram_candidates:
        start: int
        for start in range(0, len(unigram_candidates), int(pos_batch_size)):
            unigram_batch: list[str] = unigram_candidates[
                start : start + int(pos_batch_size)
            ]
            sequences: list[list[str]] = [[value] for value in unigram_batch]
            try:
                tagged_batch = nltk.pos_tag_sents(sequences, tagset="universal")
            except LookupError as exc:
                raise RuntimeError(
                    "NLTK POS resources are missing. Run:\n"
                    "python - <<'PY'\n"
                    "import nltk\n"
                    "nltk.download('averaged_perceptron_tagger', quiet=True)\n"
                    "nltk.download('averaged_perceptron_tagger_eng', quiet=True)\n"
                    "nltk.download('universal_tagset', quiet=True)\n"
                    "nltk.download('wordnet', quiet=True)\n"
                    "nltk.download('omw-1.4', quiet=True)\n"
                    "PY"
                ) from exc
            tagged: list[tuple[str, str]]
            unigram_term: str
            for unigram_term, tagged in zip(unigram_batch, tagged_batch):
                if not tagged:
                    reason_counts["pos_missing"] += 1
                    continue
                if str(tagged[0][1]).upper() != "NOUN":
                    reason_counts["non_noun_unigram"] += 1
                    continue
                if not _is_plausible_plural_surface_form(str(tagged[0][0])):
                    reason_counts["unigram_not_plural_surface"] += 1
                    continue
                normalized_token, reason = _hybrid_singularize_noun_token(
                    token=str(tagged[0][0]),
                    exception_words=exception_words,
                )
                reason_counts[reason] += 1
                if reason == "normalized":
                    normalized_map[unigram_term] = normalized_token

    if phrase_candidates:
        start = 0
        for start in range(0, len(phrase_candidates), int(pos_batch_size)):
            phrase_batch: list[str] = phrase_candidates[
                start : start + int(pos_batch_size)
            ]
            sequences = [[piece for piece in value.split() if piece] for value in phrase_batch]
            try:
                tagged_batch = nltk.pos_tag_sents(sequences, tagset="universal")
            except LookupError as exc:
                raise RuntimeError(
                    "NLTK POS resources are missing. Run:\n"
                    "python - <<'PY'\n"
                    "import nltk\n"
                    "nltk.download('averaged_perceptron_tagger', quiet=True)\n"
                    "nltk.download('averaged_perceptron_tagger_eng', quiet=True)\n"
                    "nltk.download('universal_tagset', quiet=True)\n"
                    "nltk.download('wordnet', quiet=True)\n"
                    "nltk.download('omw-1.4', quiet=True)\n"
                    "PY"
                ) from exc
            phrase_term: str
            tagged: list[tuple[str, str]]
            for phrase_term, tagged in zip(phrase_batch, tagged_batch):
                if not tagged:
                    reason_counts["pos_missing"] += 1
                    continue
                phrase_tags: list[tuple[str, str]] = [
                    (str(tok), str(pos).upper()) for tok, pos in tagged
                ]
                head_index: int | None = None
                index: int
                token_value: str
                pos_value: str
                for index in range(len(phrase_tags) - 1, -1, -1):
                    token_value, pos_value = phrase_tags[index]
                    if pos_value == "NOUN":
                        head_index = index
                        break
                if head_index is None:
                    reason_counts["non_noun_phrase"] += 1
                    continue
                head_token: str = str(phrase_tags[head_index][0]).lower()
                if not _is_plausible_plural_surface_form(head_token):
                    reason_counts["phrase_head_not_plural_surface"] += 1
                    continue
                normalized_head, reason = _hybrid_singularize_noun_token(
                    token=head_token,
                    exception_words=exception_words,
                )
                reason_counts[f"phrase_{reason}"] += 1
                if reason != "normalized":
                    continue
                phrase_parts: list[str] = [piece for piece in phrase_term.split() if piece]
                if head_index >= len(phrase_parts):
                    reason_counts["phrase_head_index_mismatch"] += 1
                    continue
                phrase_parts[head_index] = normalized_head
                normalized_map[phrase_term] = " ".join(phrase_parts)

    normalized_unigrams: int = sum(
        1 for key in normalized_map.keys() if len(str(key).split()) == 1
    )
    normalized_phrases: int = int(len(normalized_map) - normalized_unigrams)
    stats: dict[str, Any] = {
        "enabled": True,
        "total_terms": int(len(terms)),
        "normalized_terms": int(len(normalized_map)),
        "normalized_unigrams": int(normalized_unigrams),
        "normalized_phrases": int(normalized_phrases),
        "skip_entity_backed": bool(skip_entity_backed),
        "include_phrases": bool(include_phrases),
        "exception_count": int(len(exception_words)),
        "wordnet_available": bool(_get_wordnet_lemmatizer() is not None),
        "reason_counts": {
            key: int(value) for key, value in sorted(reason_counts.items())
        },
    }
    return normalized_map, stats


def _generic_unigram_df_penalty(
    *,
    term: str,
    term_df: int,
    doc_count: int,
    start_ratio: float,
    end_ratio: float,
    min_multiplier: float,
    penalty_power: float,
) -> float:
    if int(doc_count) <= 0:
        return 1.0
    parts: list[str] = [part for part in str(term).split() if part]
    if len(parts) != 1:
        return 1.0
    ratio: float = float(term_df) / float(doc_count)
    if ratio <= float(start_ratio):
        return 1.0
    if float(end_ratio) <= float(start_ratio):
        return float(min_multiplier)
    scaled: float = (ratio - float(start_ratio)) / (float(end_ratio) - float(start_ratio))
    if scaled < 0.0:
        scaled = 0.0
    elif scaled > 1.0:
        scaled = 1.0
    retained: float = (1.0 - scaled) ** float(penalty_power)
    multiplier: float = float(min_multiplier) + (1.0 - float(min_multiplier)) * retained
    if multiplier < float(min_multiplier):
        multiplier = float(min_multiplier)
    if multiplier > 1.0:
        multiplier = 1.0
    return multiplier


def _phrase_cohesion_score(
    *,
    term: str,
    df_counter: Counter[str],
    doc_count: int,
    method: str,
) -> float | None:
    if int(doc_count) <= 0:
        return None
    parts: list[str] = [part for part in str(term).split() if part]
    if len(parts) < 2:
        return None

    phrase_df: int = int(df_counter.get(term, 0))
    if phrase_df <= 0:
        return None

    p_phrase: float = max(float(phrase_df) / float(doc_count), 1.0 / float(doc_count))
    log_p_phrase: float = math.log(p_phrase)
    sum_log_parts: float = 0.0
    part: str
    for part in parts:
        part_df: int = int(df_counter.get(part, 0))
        if part_df <= 0:
            part_df = 1
        p_part: float = max(float(part_df) / float(doc_count), 1.0 / float(doc_count))
        sum_log_parts += math.log(p_part)

    pmi: float = log_p_phrase - sum_log_parts
    if str(method).lower() == "pmi":
        return pmi
    denom: float = -log_p_phrase
    if denom <= 0.0:
        return None
    return pmi / denom


def _normalize_leading_determiners(
    *,
    phrase: str,
    leading_determiners: set[str],
    min_tokens: int,
) -> tuple[str | None, bool, bool]:
    text: str = str(phrase).strip()
    if not text:
        return None, False, False
    parts: list[str] = [part for part in text.split() if part]
    if not parts:
        return None, False, False

    original_count: int = len(parts)
    while parts and parts[0].lower() in leading_determiners:
        parts = parts[1:]

    if len(parts) == original_count:
        return text, False, False
    if len(parts) < int(min_tokens):
        return None, True, True
    return " ".join(parts), True, False


def _normalize_text_simple(text: str) -> list[str]:
    return [token.lower() for token in _SIMPLE_TOKEN_PATTERN.findall(text)]


def _register_doc_term_source(
    *,
    term: str,
    source: str,
    term_sources: dict[str, set[str]],
) -> None:
    source_set: set[str] | None = term_sources.get(term)
    if source_set is None:
        term_sources[term] = {source}
        return
    source_set.add(source)


def _normalize_span_tokens(
    span: Any,
    *,
    prefer_lemma: bool,
    lowercase: bool,
    min_tokens: int,
    max_tokens: int,
    max_phrase_chars: int,
    max_stopword_ratio: float | None = None,
) -> str | None:
    fragments: list[str] = []
    token_count: int = 0
    stopword_count: int = 0
    token: Any
    for token in span:
        if bool(token.is_space) or bool(token.is_punct):
            continue
        token_count += 1
        if bool(token.is_stop):
            stopword_count += 1
        raw_text: str
        if prefer_lemma:
            raw_text = str(token.lemma_).strip()
            if not raw_text or raw_text == "-PRON-":
                raw_text = str(token.text).strip()
        else:
            raw_text = str(token.text).strip()
        if not raw_text:
            continue
        term_fragment: str
        for term_fragment in _SIMPLE_TOKEN_PATTERN.findall(raw_text):
            normalized_fragment: str = (
                term_fragment.lower() if lowercase else term_fragment
            )
            if normalized_fragment:
                fragments.append(normalized_fragment)
    fragment_count: int = len(fragments)
    if fragment_count < int(min_tokens) or fragment_count > int(max_tokens):
        return None
    if (
        max_stopword_ratio is not None
        and token_count > 0
        and float(stopword_count) / float(token_count) > float(max_stopword_ratio)
    ):
        return None
    phrase: str = " ".join(fragments).strip()
    if not phrase:
        return None
    if len(phrase) > int(max_phrase_chars):
        return None
    return phrase


def _normalize_document(
    doc: Any,
    *,
    extract_entities: bool,
    extract_noun_chunks: bool,
    allowed_entity_labels: set[str] | None,
    entity_min_tokens: int,
    entity_max_tokens: int,
    noun_chunk_min_tokens: int,
    noun_chunk_max_tokens: int,
    noun_chunk_normalization: str,
    noun_chunk_max_stopword_ratio: float,
    max_phrase_chars: int,
    normalize_leading_determiners: bool,
    leading_determiners: set[str],
    normalize_entity_determiners: bool,
    normalize_noun_chunk_determiners: bool,
) -> tuple[list[str], dict[str, set[str]], dict[str, int]]:
    terms: list[str] = []
    term_sources: dict[str, set[str]] = {}
    stats: dict[str, int] = {
        "token_terms": 0,
        "entity_spans_seen": 0,
        "entity_terms": 0,
        "noun_chunks_seen": 0,
        "noun_chunk_terms": 0,
        "noun_chunk_errors": 0,
        "determiner_normalized_entity_terms": 0,
        "determiner_normalized_noun_chunk_terms": 0,
        "determiner_dropped_short_entity_terms": 0,
        "determiner_dropped_short_noun_chunk_terms": 0,
    }

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
        if not normalized:
            continue
        terms.append(normalized)
        _register_doc_term_source(
            term=normalized,
            source=_SOURCE_TOKEN,
            term_sources=term_sources,
        )
        stats["token_terms"] += 1

    if bool(extract_entities):
        entity_span: Any
        for entity_span in doc.ents:
            stats["entity_spans_seen"] += 1
            entity_label: str = str(entity_span.label_).upper()
            if (
                allowed_entity_labels is not None
                and entity_label not in allowed_entity_labels
            ):
                continue
            phrase: str | None = _normalize_span_tokens(
                entity_span,
                prefer_lemma=False,
                lowercase=True,
                min_tokens=int(entity_min_tokens),
                max_tokens=int(entity_max_tokens),
                max_phrase_chars=int(max_phrase_chars),
            )
            if phrase is None:
                continue
            if bool(normalize_leading_determiners) and bool(normalize_entity_determiners):
                phrase, changed, dropped_short = _normalize_leading_determiners(
                    phrase=phrase,
                    leading_determiners=leading_determiners,
                    min_tokens=int(entity_min_tokens),
                )
                if changed:
                    stats["determiner_normalized_entity_terms"] += 1
                if dropped_short:
                    stats["determiner_dropped_short_entity_terms"] += 1
                    continue
                if phrase is None:
                    continue
            terms.append(phrase)
            _register_doc_term_source(
                term=phrase,
                source=_SOURCE_ENTITY,
                term_sources=term_sources,
            )
            stats["entity_terms"] += 1

    if bool(extract_noun_chunks):
        try:
            noun_chunk: Any
            for noun_chunk in doc.noun_chunks:
                stats["noun_chunks_seen"] += 1
                root_pos: str = str(noun_chunk.root.pos_).upper()
                if root_pos not in _NOUN_CHUNK_ROOT_POS:
                    continue
                phrase = _normalize_span_tokens(
                    noun_chunk,
                    prefer_lemma=(str(noun_chunk_normalization).lower() == "lemma"),
                    lowercase=True,
                    min_tokens=int(noun_chunk_min_tokens),
                    max_tokens=int(noun_chunk_max_tokens),
                    max_phrase_chars=int(max_phrase_chars),
                    max_stopword_ratio=float(noun_chunk_max_stopword_ratio),
                )
                if phrase is None:
                    continue
                if bool(normalize_leading_determiners) and bool(
                    normalize_noun_chunk_determiners
                ):
                    phrase, changed, dropped_short = _normalize_leading_determiners(
                        phrase=phrase,
                        leading_determiners=leading_determiners,
                        min_tokens=int(noun_chunk_min_tokens),
                    )
                    if changed:
                        stats["determiner_normalized_noun_chunk_terms"] += 1
                    if dropped_short:
                        stats["determiner_dropped_short_noun_chunk_terms"] += 1
                        continue
                    if phrase is None:
                        continue
                terms.append(phrase)
                _register_doc_term_source(
                    term=phrase,
                    source=_SOURCE_NOUN_CHUNK,
                    term_sources=term_sources,
                )
                stats["noun_chunk_terms"] += 1
        except Exception:
            stats["noun_chunk_errors"] += 1

    return terms, term_sources, stats


def _iter_document_tokens_simple(
    *,
    texts: Iterable[str],
    max_docs: int | None,
) -> tuple[list[list[str]], list[dict[str, set[str]]], dict[str, Any]]:
    docs_tokens: list[list[str]] = []
    docs_term_sources: list[dict[str, set[str]]] = []
    seen: int = 0
    text: str
    for text in texts:
        normalized: list[str] = _normalize_text_simple(str(text))
        if normalized:
            term_sources: dict[str, set[str]] = {}
            term: str
            for term in normalized:
                _register_doc_term_source(
                    term=term,
                    source=_SOURCE_TOKEN,
                    term_sources=term_sources,
                )
            docs_tokens.append(normalized)
            docs_term_sources.append(term_sources)
            seen += 1
        if max_docs is not None and seen >= int(max_docs):
            break
    stats: dict[str, Any] = {
        "docs_with_tokens": seen,
        "normalizer": "simple",
        "source_term_totals": {
            _SOURCE_TOKEN: int(sum(len(tokens) for tokens in docs_tokens)),
            _SOURCE_ENTITY: 0,
            _SOURCE_NOUN_CHUNK: 0,
        },
    }
    return docs_tokens, docs_term_sources, stats


def _resolve_text_corpus(
    *,
    args: argparse.Namespace,
    meta_dataset: Any | None,
) -> tuple[Iterable[str], dict[str, Any]]:
    def _iter_dataset_texts(dataset: Any, text_column: str) -> Iterable[str]:
        row: dict[str, Any]
        for row in dataset:
            text_value: Any | None = row.get(text_column)
            if text_value is None:
                continue
            text: str = str(text_value).strip()
            if text:
                yield text

    if bool(args.use_all_corpus_documents):
        corpus_datasets = load_hf_splits(
            hf_name=args.meta_hf_name,
            hf_subset=args.corpus_subset,
            splits=[args.corpus_split],
            cache_dir=args.hf_cache_dir,
            data_files=None,
            allow_missing_split=False,
        )
        corpus_dataset = maybe_concat_datasets(corpus_datasets)
        corpus_columns: list[str] = column_names_of(corpus_dataset)
        corpus_text_col: str | None = resolve_first_present_column(
            corpus_columns,
            [args.corpus_text_column, "passage", "text", "doc", "document", "contents"],
        )
        if corpus_text_col is None:
            raise ValueError(
                "Could not resolve a corpus text column while use_all_corpus_documents=true."
            )
        corpus_iter: Iterable[str] = _iter_dataset_texts(
            corpus_dataset,
            corpus_text_col,
        )
        documents: Iterable[str] = corpus_iter
        info: dict[str, Any] = {
            "mode": "full_corpus",
            "corpus_text_column": corpus_text_col,
            "include_queries": bool(args.include_queries),
        }
        if bool(args.include_queries):
            query_datasets = load_hf_splits(
                hf_name=args.meta_hf_name,
                hf_subset=args.query_subset,
                splits=[args.query_split],
                cache_dir=args.hf_cache_dir,
                data_files=None,
                allow_missing_split=False,
            )
            query_dataset = maybe_concat_datasets(query_datasets)
            query_columns: list[str] = column_names_of(query_dataset)
            query_text_col: str | None = resolve_first_present_column(
                query_columns,
                [args.query_text_column, "query", "question", "text"],
            )
            if query_text_col is None:
                raise ValueError(
                    "Could not resolve a query text column while include_queries=true."
                )
            query_iter: Iterable[str] = _iter_dataset_texts(
                query_dataset,
                query_text_col,
            )
            documents = itertools.chain(corpus_iter, query_iter)
            info["query_text_column"] = query_text_col
        return documents, info

    if meta_dataset is None:
        raise ValueError("meta_dataset must be provided unless use_all_corpus_documents=true.")

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
    extract_entities: bool,
    extract_noun_chunks: bool,
    allowed_entity_labels: set[str] | None,
    entity_min_tokens: int,
    entity_max_tokens: int,
    noun_chunk_min_tokens: int,
    noun_chunk_max_tokens: int,
    noun_chunk_normalization: str,
    noun_chunk_max_stopword_ratio: float,
    max_phrase_chars: int,
    normalize_leading_determiners: bool,
    leading_determiners: set[str],
    normalize_entity_determiners: bool,
    normalize_noun_chunk_determiners: bool,
) -> tuple[list[list[str]], list[dict[str, set[str]]], dict[str, Any]]:
    if str(normalizer).lower() == "simple":
        return _iter_document_tokens_simple(texts=texts, max_docs=max_docs)

    try:
        import spacy
    except ImportError as exc:
        if bool(allow_simple_fallback):
            docs_tokens, docs_term_sources, stats = _iter_document_tokens_simple(
                texts=texts,
                max_docs=max_docs,
            )
            stats["fallback_reason"] = (
                "spacy import failed; using simple regex normalization."
            )
            return docs_tokens, docs_term_sources, stats
        raise RuntimeError(
            "spaCy is required. Install package `spacy` and an English model (e.g., "
            "`python -m spacy download en_core_web_trf`) or rerun with "
            "--allow-simple-fallback."
        ) from exc

    try:
        nlp = spacy.load(spacy_model)
    except Exception as exc:
        if bool(allow_simple_fallback):
            docs_tokens, docs_term_sources, stats = _iter_document_tokens_simple(
                texts=texts,
                max_docs=max_docs,
            )
            stats["fallback_reason"] = (
                f"spacy model load failed ({exc!r}); using simple regex normalization."
            )
            return docs_tokens, docs_term_sources, stats
        raise RuntimeError(
            f"Failed to load spaCy model {spacy_model!r}. "
            "Install the model or rerun with --allow-simple-fallback."
        ) from exc
    docs_tokens: list[list[str]] = []
    docs_term_sources: list[dict[str, set[str]]] = []
    seen: int = 0
    source_term_totals: Counter[str] = Counter()
    doc_normalization_stats: Counter[str] = Counter()

    for doc in nlp.pipe(texts, batch_size=int(batch_size), n_process=int(n_process)):
        normalized, term_sources, doc_stats = _normalize_document(
            doc,
            extract_entities=bool(extract_entities),
            extract_noun_chunks=bool(extract_noun_chunks),
            allowed_entity_labels=allowed_entity_labels,
            entity_min_tokens=int(entity_min_tokens),
            entity_max_tokens=int(entity_max_tokens),
            noun_chunk_min_tokens=int(noun_chunk_min_tokens),
            noun_chunk_max_tokens=int(noun_chunk_max_tokens),
            noun_chunk_normalization=str(noun_chunk_normalization),
            noun_chunk_max_stopword_ratio=float(noun_chunk_max_stopword_ratio),
            max_phrase_chars=int(max_phrase_chars),
            normalize_leading_determiners=bool(normalize_leading_determiners),
            leading_determiners=leading_determiners,
            normalize_entity_determiners=bool(normalize_entity_determiners),
            normalize_noun_chunk_determiners=bool(normalize_noun_chunk_determiners),
        )
        if normalized:
            docs_tokens.append(normalized)
            docs_term_sources.append(term_sources)
            seen += 1
            source_term_totals[_SOURCE_TOKEN] += int(doc_stats["token_terms"])
            source_term_totals[_SOURCE_ENTITY] += int(doc_stats["entity_terms"])
            source_term_totals[_SOURCE_NOUN_CHUNK] += int(doc_stats["noun_chunk_terms"])
        doc_normalization_stats.update(doc_stats)
        if max_docs is not None and seen >= int(max_docs):
            break

    stats: dict[str, Any] = {
        "docs_with_tokens": seen,
        "spacy_model": spacy_model,
        "spacy_batch_size": int(batch_size),
        "spacy_n_process": int(n_process),
        "normalizer": "spacy",
        "entity_label_filter": (
            sorted(allowed_entity_labels) if allowed_entity_labels is not None else None
        ),
        "extraction": {
            "extract_entities": bool(extract_entities),
            "extract_noun_chunks": bool(extract_noun_chunks),
            "entity_min_tokens": int(entity_min_tokens),
            "entity_max_tokens": int(entity_max_tokens),
            "noun_chunk_min_tokens": int(noun_chunk_min_tokens),
            "noun_chunk_max_tokens": int(noun_chunk_max_tokens),
            "noun_chunk_normalization": str(noun_chunk_normalization),
            "noun_chunk_max_stopword_ratio": float(noun_chunk_max_stopword_ratio),
            "max_phrase_chars": int(max_phrase_chars),
            "normalize_leading_determiners": bool(normalize_leading_determiners),
            "leading_determiners": sorted(leading_determiners),
            "normalize_entity_determiners": bool(normalize_entity_determiners),
            "normalize_noun_chunk_determiners": bool(normalize_noun_chunk_determiners),
        },
        "source_term_totals": {
            _SOURCE_TOKEN: int(source_term_totals.get(_SOURCE_TOKEN, 0)),
            _SOURCE_ENTITY: int(source_term_totals.get(_SOURCE_ENTITY, 0)),
            _SOURCE_NOUN_CHUNK: int(source_term_totals.get(_SOURCE_NOUN_CHUNK, 0)),
        },
        "doc_normalization_stats": {
            key: int(value) for key, value in sorted(doc_normalization_stats.items())
        },
    }
    return docs_tokens, docs_term_sources, stats


def _resolve_term_source_boost(
    *,
    term: str,
    source_df_counter: dict[str, Counter[str]],
    source_boosts: dict[str, float],
) -> float:
    weighted_boost: float = 0.0
    contribution_total: int = 0
    source: str
    for source, boost in source_boosts.items():
        source_df: int = int(source_df_counter.get(source, Counter()).get(term, 0))
        if source_df <= 0:
            continue
        weighted_boost += float(source_df) * float(boost)
        contribution_total += int(source_df)
    if contribution_total <= 0:
        return float(source_boosts.get(_SOURCE_TOKEN, 1.0))
    return weighted_boost / float(contribution_total)


def _update_term_statistics(
    *,
    tokens: list[str],
    doc_term_sources: dict[str, set[str]],
    df_counter: Counter[str],
    tf_total_counter: Counter[str],
    term_sources: dict[str, set[str]],
    source_df_counter: dict[str, Counter[str]],
    source_doc_hits: Counter[str],
    source_tf_hits: Counter[str],
) -> None:
    tf_doc: Counter[str] = Counter(tokens)
    tf_total_counter.update(tf_doc)
    df_counter.update(tf_doc.keys())
    term: str
    tf_value: int
    for term, tf_value in tf_doc.items():
        sources: set[str] = set(doc_term_sources.get(term, {_SOURCE_TOKEN}))
        if not sources:
            sources = {_SOURCE_TOKEN}
        known_sources: set[str] | None = term_sources.get(term)
        if known_sources is None:
            term_sources[term] = set(sources)
        else:
            known_sources.update(sources)
        source: str
        for source in sources:
            counter: Counter[str] = source_df_counter.setdefault(source, Counter())
            counter[term] += 1
            source_doc_hits[source] += 1
            source_tf_hits[source] += int(tf_value)


def _collect_term_statistics(
    *,
    texts: Iterable[str],
    spacy_model: str,
    batch_size: int,
    n_process: int,
    max_docs: int | None,
    normalizer: str,
    allow_simple_fallback: bool,
    extract_entities: bool,
    extract_noun_chunks: bool,
    allowed_entity_labels: set[str] | None,
    entity_min_tokens: int,
    entity_max_tokens: int,
    noun_chunk_min_tokens: int,
    noun_chunk_max_tokens: int,
    noun_chunk_normalization: str,
    noun_chunk_max_stopword_ratio: float,
    max_phrase_chars: int,
    normalize_leading_determiners: bool,
    leading_determiners: set[str],
    normalize_entity_determiners: bool,
    normalize_noun_chunk_determiners: bool,
) -> tuple[
    Counter[str],
    Counter[str],
    dict[str, set[str]],
    dict[str, Counter[str]],
    Counter[str],
    Counter[str],
    int,
    dict[str, Any],
]:
    df_counter: Counter[str] = Counter()
    tf_total_counter: Counter[str] = Counter()
    term_sources: dict[str, set[str]] = {}
    source_df_counter: dict[str, Counter[str]] = {}
    source_doc_hits: Counter[str] = Counter()
    source_tf_hits: Counter[str] = Counter()
    docs_with_tokens: int = 0

    def _process_simple_stream(
        stream: Iterable[str],
    ) -> tuple[int, dict[str, Any]]:
        seen: int = 0
        source_term_totals: Counter[str] = Counter()
        text: str
        for text in stream:
            normalized: list[str] = _normalize_text_simple(str(text))
            if normalized:
                doc_term_sources: dict[str, set[str]] = {}
                term: str
                for term in normalized:
                    _register_doc_term_source(
                        term=term,
                        source=_SOURCE_TOKEN,
                        term_sources=doc_term_sources,
                    )
                _update_term_statistics(
                    tokens=normalized,
                    doc_term_sources=doc_term_sources,
                    df_counter=df_counter,
                    tf_total_counter=tf_total_counter,
                    term_sources=term_sources,
                    source_df_counter=source_df_counter,
                    source_doc_hits=source_doc_hits,
                    source_tf_hits=source_tf_hits,
                )
                source_term_totals[_SOURCE_TOKEN] += int(len(normalized))
                seen += 1
            if max_docs is not None and seen >= int(max_docs):
                break
        stats: dict[str, Any] = {
            "docs_with_tokens": seen,
            "normalizer": "simple",
            "source_term_totals": {
                _SOURCE_TOKEN: int(source_term_totals.get(_SOURCE_TOKEN, 0)),
                _SOURCE_ENTITY: 0,
                _SOURCE_NOUN_CHUNK: 0,
            },
        }
        return seen, stats

    if str(normalizer).lower() == "simple":
        docs_with_tokens, stats = _process_simple_stream(texts)
        return (
            df_counter,
            tf_total_counter,
            term_sources,
            source_df_counter,
            source_doc_hits,
            source_tf_hits,
            docs_with_tokens,
            stats,
        )

    try:
        import spacy
    except ImportError as exc:
        if bool(allow_simple_fallback):
            docs_with_tokens, stats = _process_simple_stream(texts)
            stats["fallback_reason"] = (
                "spacy import failed; using simple regex normalization."
            )
            return (
                df_counter,
                tf_total_counter,
                term_sources,
                source_df_counter,
                source_doc_hits,
                source_tf_hits,
                docs_with_tokens,
                stats,
            )
        raise RuntimeError(
            "spaCy is required. Install package `spacy` and an English model (e.g., "
            "`python -m spacy download en_core_web_trf`) or rerun with "
            "--allow-simple-fallback."
        ) from exc

    try:
        nlp = spacy.load(spacy_model)
    except Exception as exc:
        if bool(allow_simple_fallback):
            docs_with_tokens, stats = _process_simple_stream(texts)
            stats["fallback_reason"] = (
                f"spacy model load failed ({exc!r}); using simple regex normalization."
            )
            return (
                df_counter,
                tf_total_counter,
                term_sources,
                source_df_counter,
                source_doc_hits,
                source_tf_hits,
                docs_with_tokens,
                stats,
            )
        raise RuntimeError(
            f"Failed to load spaCy model {spacy_model!r}. "
            "Install the model or rerun with --allow-simple-fallback."
        ) from exc

    source_term_totals: Counter[str] = Counter()
    doc_normalization_stats: Counter[str] = Counter()
    doc: Any
    for doc in nlp.pipe(texts, batch_size=int(batch_size), n_process=int(n_process)):
        normalized, doc_term_sources, doc_stats = _normalize_document(
            doc,
            extract_entities=bool(extract_entities),
            extract_noun_chunks=bool(extract_noun_chunks),
            allowed_entity_labels=allowed_entity_labels,
            entity_min_tokens=int(entity_min_tokens),
            entity_max_tokens=int(entity_max_tokens),
            noun_chunk_min_tokens=int(noun_chunk_min_tokens),
            noun_chunk_max_tokens=int(noun_chunk_max_tokens),
            noun_chunk_normalization=str(noun_chunk_normalization),
            noun_chunk_max_stopword_ratio=float(noun_chunk_max_stopword_ratio),
            max_phrase_chars=int(max_phrase_chars),
            normalize_leading_determiners=bool(normalize_leading_determiners),
            leading_determiners=leading_determiners,
            normalize_entity_determiners=bool(normalize_entity_determiners),
            normalize_noun_chunk_determiners=bool(normalize_noun_chunk_determiners),
        )
        doc_normalization_stats.update(doc_stats)
        if normalized:
            _update_term_statistics(
                tokens=normalized,
                doc_term_sources=doc_term_sources,
                df_counter=df_counter,
                tf_total_counter=tf_total_counter,
                term_sources=term_sources,
                source_df_counter=source_df_counter,
                source_doc_hits=source_doc_hits,
                source_tf_hits=source_tf_hits,
            )
            docs_with_tokens += 1
            source_term_totals[_SOURCE_TOKEN] += int(doc_stats["token_terms"])
            source_term_totals[_SOURCE_ENTITY] += int(doc_stats["entity_terms"])
            source_term_totals[_SOURCE_NOUN_CHUNK] += int(doc_stats["noun_chunk_terms"])
        if max_docs is not None and docs_with_tokens >= int(max_docs):
            break

    stats = {
        "docs_with_tokens": docs_with_tokens,
        "spacy_model": spacy_model,
        "spacy_batch_size": int(batch_size),
        "spacy_n_process": int(n_process),
        "normalizer": "spacy",
        "entity_label_filter": (
            sorted(allowed_entity_labels) if allowed_entity_labels is not None else None
        ),
        "extraction": {
            "extract_entities": bool(extract_entities),
            "extract_noun_chunks": bool(extract_noun_chunks),
            "entity_min_tokens": int(entity_min_tokens),
            "entity_max_tokens": int(entity_max_tokens),
            "noun_chunk_min_tokens": int(noun_chunk_min_tokens),
            "noun_chunk_max_tokens": int(noun_chunk_max_tokens),
            "noun_chunk_normalization": str(noun_chunk_normalization),
            "noun_chunk_max_stopword_ratio": float(noun_chunk_max_stopword_ratio),
            "max_phrase_chars": int(max_phrase_chars),
            "normalize_leading_determiners": bool(normalize_leading_determiners),
            "leading_determiners": sorted(leading_determiners),
            "normalize_entity_determiners": bool(normalize_entity_determiners),
            "normalize_noun_chunk_determiners": bool(normalize_noun_chunk_determiners),
        },
        "source_term_totals": {
            _SOURCE_TOKEN: int(source_term_totals.get(_SOURCE_TOKEN, 0)),
            _SOURCE_ENTITY: int(source_term_totals.get(_SOURCE_ENTITY, 0)),
            _SOURCE_NOUN_CHUNK: int(source_term_totals.get(_SOURCE_NOUN_CHUNK, 0)),
        },
        "doc_normalization_stats": {
            key: int(value) for key, value in sorted(doc_normalization_stats.items())
        },
    }
    return (
        df_counter,
        tf_total_counter,
        term_sources,
        source_df_counter,
        source_doc_hits,
        source_tf_hits,
        docs_with_tokens,
        stats,
    )


def _resolve_corpus_text_column_for_map_reduce(args: argparse.Namespace) -> str:
    corpus_dataset: Any = load_hf_split(
        hf_name=args.meta_hf_name,
        hf_subset=args.corpus_subset,
        split=args.corpus_split,
        cache_dir=args.hf_cache_dir,
        data_files=None,
    )
    columns: list[str] = column_names_of(corpus_dataset)
    corpus_text_col: str | None = resolve_first_present_column(
        columns,
        [args.corpus_text_column, "passage", "text", "doc", "document", "contents"],
    )
    if corpus_text_col is None:
        raise ValueError(
            "Could not resolve a corpus text column while map_reduce_sharding=true."
        )
    return corpus_text_col


def _iter_dataset_texts_with_cap(
    *,
    dataset: Dataset,
    text_column: str,
    max_docs: int | None,
) -> Iterable[str]:
    seen: int = 0
    row: dict[str, Any]
    for row in dataset:
        text_value: Any | None = row.get(text_column)
        if text_value is None:
            continue
        text: str = str(text_value).strip()
        if not text:
            continue
        yield text
        seen += 1
        if max_docs is not None and seen >= int(max_docs):
            break


def _run_map_reduce_shard(payload: dict[str, Any]) -> dict[str, Any]:
    # Avoid oversubscription inside worker subprocesses.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    shard_index: int = int(payload["shard_index"])
    shard_count: int = int(payload["shard_count"])
    artifact_path: Path = Path(str(payload["artifact_path"]))
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_any: Any = load_hf_split(
        hf_name=str(payload["meta_hf_name"]),
        hf_subset=str(payload["corpus_subset"]),
        split=str(payload["corpus_split"]),
        cache_dir=payload.get("hf_cache_dir"),
        data_files=None,
    )
    if not isinstance(dataset_any, Dataset):
        raise RuntimeError(
            "map_reduce_sharding requires a map-style Dataset for corpus split."
        )
    shard_dataset: Dataset = dataset_any.shard(
        num_shards=int(shard_count),
        index=int(shard_index),
        contiguous=True,
    )

    text_column: str = str(payload["corpus_text_column"])
    max_docs: int | None = payload.get("max_docs_per_shard")
    if max_docs is not None:
        max_docs = int(max_docs)

    normalizer: str = str(payload["normalizer"]).lower()
    allow_simple_fallback: bool = bool(payload["allow_simple_fallback"])
    extract_entities: bool = bool(payload["extract_entities"])
    extract_noun_chunks: bool = bool(payload["extract_noun_chunks"])
    allowed_entity_labels_raw: list[str] | None = payload.get("allowed_entity_labels")
    allowed_entity_labels: set[str] | None = (
        None
        if allowed_entity_labels_raw is None
        else {str(label).upper() for label in allowed_entity_labels_raw}
    )

    entity_min_tokens: int = int(payload["entity_min_tokens"])
    entity_max_tokens: int = int(payload["entity_max_tokens"])
    noun_chunk_min_tokens: int = int(payload["noun_chunk_min_tokens"])
    noun_chunk_max_tokens: int = int(payload["noun_chunk_max_tokens"])
    noun_chunk_normalization: str = str(payload["noun_chunk_normalization"])
    noun_chunk_max_stopword_ratio: float = float(payload["noun_chunk_max_stopword_ratio"])
    max_phrase_chars: int = int(payload["max_phrase_chars"])
    normalize_leading_determiners: bool = bool(payload["normalize_leading_determiners"])
    leading_determiners_raw: list[str] | None = payload.get("leading_determiners")
    leading_determiners: set[str] = (
        set()
        if leading_determiners_raw is None
        else {str(value).strip().lower() for value in leading_determiners_raw if str(value).strip()}
    )
    normalize_entity_determiners: bool = bool(payload["normalize_entity_determiners"])
    normalize_noun_chunk_determiners: bool = bool(
        payload["normalize_noun_chunk_determiners"]
    )
    spacy_model: str = str(payload["spacy_model"])
    spacy_batch_size: int = int(payload["spacy_batch_size"])

    df_counter: Counter[str] = Counter()
    tf_total_counter: Counter[str] = Counter()
    term_sources: dict[str, set[str]] = {}
    source_df_counter: dict[str, Counter[str]] = {}
    source_doc_hits: Counter[str] = Counter()
    source_tf_hits: Counter[str] = Counter()
    docs_with_tokens: int = 0
    source_term_totals: Counter[str] = Counter()
    doc_normalization_stats: Counter[str] = Counter()
    normalizer_used: str = normalizer
    fallback_reason: str | None = None

    texts: Iterable[str] = _iter_dataset_texts_with_cap(
        dataset=shard_dataset,
        text_column=text_column,
        max_docs=max_docs,
    )

    if normalizer == "simple":
        text: str
        for text in texts:
            normalized: list[str] = _normalize_text_simple(text)
            if not normalized:
                continue
            doc_term_sources: dict[str, set[str]] = {}
            term: str
            for term in normalized:
                _register_doc_term_source(
                    term=term,
                    source=_SOURCE_TOKEN,
                    term_sources=doc_term_sources,
                )
            _update_term_statistics(
                tokens=normalized,
                doc_term_sources=doc_term_sources,
                df_counter=df_counter,
                tf_total_counter=tf_total_counter,
                term_sources=term_sources,
                source_df_counter=source_df_counter,
                source_doc_hits=source_doc_hits,
                source_tf_hits=source_tf_hits,
            )
            source_term_totals[_SOURCE_TOKEN] += int(len(normalized))
            docs_with_tokens += 1
    else:
        nlp: Any
        try:
            import spacy

            nlp = spacy.load(spacy_model)
        except Exception as exc:
            if not allow_simple_fallback:
                raise RuntimeError(
                    f"Failed to load spaCy model {spacy_model!r} in shard {shard_index}."
                ) from exc
            normalizer_used = "simple"
            fallback_reason = (
                f"spacy model load failed in shard {shard_index} ({exc!r}); "
                "using simple regex normalization."
            )
            text = ""
            for text in texts:
                normalized = _normalize_text_simple(text)
                if not normalized:
                    continue
                doc_term_sources = {}
                term = ""
                for term in normalized:
                    _register_doc_term_source(
                        term=term,
                        source=_SOURCE_TOKEN,
                        term_sources=doc_term_sources,
                    )
                _update_term_statistics(
                    tokens=normalized,
                    doc_term_sources=doc_term_sources,
                    df_counter=df_counter,
                    tf_total_counter=tf_total_counter,
                    term_sources=term_sources,
                    source_df_counter=source_df_counter,
                    source_doc_hits=source_doc_hits,
                    source_tf_hits=source_tf_hits,
                )
                source_term_totals[_SOURCE_TOKEN] += int(len(normalized))
                docs_with_tokens += 1
        else:
            doc: Any
            for doc in nlp.pipe(texts, batch_size=spacy_batch_size, n_process=1):
                normalized, doc_term_sources, doc_stats = _normalize_document(
                    doc,
                    extract_entities=extract_entities,
                    extract_noun_chunks=extract_noun_chunks,
                    allowed_entity_labels=allowed_entity_labels,
                    entity_min_tokens=entity_min_tokens,
                    entity_max_tokens=entity_max_tokens,
                    noun_chunk_min_tokens=noun_chunk_min_tokens,
                    noun_chunk_max_tokens=noun_chunk_max_tokens,
                    noun_chunk_normalization=noun_chunk_normalization,
                    noun_chunk_max_stopword_ratio=noun_chunk_max_stopword_ratio,
                    max_phrase_chars=max_phrase_chars,
                    normalize_leading_determiners=normalize_leading_determiners,
                    leading_determiners=leading_determiners,
                    normalize_entity_determiners=normalize_entity_determiners,
                    normalize_noun_chunk_determiners=normalize_noun_chunk_determiners,
                )
                doc_normalization_stats.update(doc_stats)
                if not normalized:
                    continue
                _update_term_statistics(
                    tokens=normalized,
                    doc_term_sources=doc_term_sources,
                    df_counter=df_counter,
                    tf_total_counter=tf_total_counter,
                    term_sources=term_sources,
                    source_df_counter=source_df_counter,
                    source_doc_hits=source_doc_hits,
                    source_tf_hits=source_tf_hits,
                )
                docs_with_tokens += 1
                source_term_totals[_SOURCE_TOKEN] += int(doc_stats["token_terms"])
                source_term_totals[_SOURCE_ENTITY] += int(doc_stats["entity_terms"])
                source_term_totals[_SOURCE_NOUN_CHUNK] += int(
                    doc_stats["noun_chunk_terms"]
                )

    shard_payload: dict[str, Any] = {
        "df_counter": df_counter,
        "tf_total_counter": tf_total_counter,
        "term_sources": term_sources,
        "source_df_counter": source_df_counter,
        "source_doc_hits": source_doc_hits,
        "source_tf_hits": source_tf_hits,
    }
    with artifact_path.open("wb") as fout:
        pickle.dump(shard_payload, fout, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "shard_index": shard_index,
        "artifact_path": str(artifact_path),
        "docs_with_tokens": int(docs_with_tokens),
        "normalizer": normalizer_used,
        "fallback_reason": fallback_reason,
        "source_term_totals": {
            _SOURCE_TOKEN: int(source_term_totals.get(_SOURCE_TOKEN, 0)),
            _SOURCE_ENTITY: int(source_term_totals.get(_SOURCE_ENTITY, 0)),
            _SOURCE_NOUN_CHUNK: int(source_term_totals.get(_SOURCE_NOUN_CHUNK, 0)),
        },
        "doc_normalization_stats": {
            key: int(value) for key, value in sorted(doc_normalization_stats.items())
        },
    }


def _collect_term_statistics_map_reduce(
    *,
    args: argparse.Namespace,
    allowed_entity_labels: set[str] | None,
) -> tuple[
    Counter[str],
    Counter[str],
    dict[str, set[str]],
    dict[str, Counter[str]],
    Counter[str],
    Counter[str],
    int,
    dict[str, Any],
]:
    shard_count: int = int(args.map_reduce_num_shards)
    if shard_count <= 0:
        shard_count = max(int(args.spacy_n_process), 1)
    worker_count: int = int(args.map_reduce_num_workers)
    if worker_count <= 0:
        worker_count = shard_count
    worker_count = max(1, min(worker_count, shard_count))

    corpus_text_column: str = _resolve_corpus_text_column_for_map_reduce(args)
    temp_dir: Path
    if args.map_reduce_tmp_dir is None:
        temp_dir = Path(args.output_dir) / ".map_reduce_tmp"
    else:
        temp_dir = Path(str(args.map_reduce_tmp_dir))
    temp_dir.mkdir(parents=True, exist_ok=True)
    leading_determiners: set[str] = _parse_leading_determiners(args.leading_determiners)

    max_docs_per_shard: int | None = None
    if args.max_docs is not None:
        max_docs_per_shard = int(math.ceil(float(int(args.max_docs)) / float(shard_count)))

    payloads: list[dict[str, Any]] = []
    shard_index: int
    for shard_index in range(shard_count):
        payloads.append(
            {
                "shard_index": int(shard_index),
                "shard_count": int(shard_count),
                "artifact_path": str(temp_dir / f"shard_{shard_index:05d}.pkl"),
                "meta_hf_name": str(args.meta_hf_name),
                "corpus_subset": str(args.corpus_subset),
                "corpus_split": str(args.corpus_split),
                "hf_cache_dir": args.hf_cache_dir,
                "corpus_text_column": corpus_text_column,
                "normalizer": str(args.normalizer),
                "allow_simple_fallback": bool(args.allow_simple_fallback),
                "extract_entities": bool(args.spacy_extract_entities),
                "extract_noun_chunks": bool(args.spacy_extract_noun_chunks),
                "allowed_entity_labels": (
                    None
                    if allowed_entity_labels is None
                    else sorted(allowed_entity_labels)
                ),
                "entity_min_tokens": int(args.entity_min_tokens),
                "entity_max_tokens": int(args.entity_max_tokens),
                "noun_chunk_min_tokens": int(args.noun_chunk_min_tokens),
                "noun_chunk_max_tokens": int(args.noun_chunk_max_tokens),
                "noun_chunk_normalization": str(args.noun_chunk_normalization),
                "noun_chunk_max_stopword_ratio": float(args.noun_chunk_max_stopword_ratio),
                "max_phrase_chars": int(args.max_phrase_chars),
                "normalize_leading_determiners": bool(args.normalize_leading_determiners),
                "leading_determiners": sorted(leading_determiners),
                "normalize_entity_determiners": bool(args.normalize_entity_determiners),
                "normalize_noun_chunk_determiners": bool(
                    args.normalize_noun_chunk_determiners
                ),
                "spacy_model": str(args.spacy_model),
                "spacy_batch_size": int(args.spacy_batch_size),
                "max_docs_per_shard": max_docs_per_shard,
            }
        )

    print(
        f"[map-reduce] Starting shard map phase: shards={shard_count}, "
        f"workers={worker_count}, tmp_dir={temp_dir}"
    )
    shard_results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_by_shard: dict[cf.Future[dict[str, Any]], int] = {}
        for payload in payloads:
            future: cf.Future[dict[str, Any]] = executor.submit(
                _run_map_reduce_shard, payload
            )
            future_by_shard[future] = int(payload["shard_index"])
        completed: int = 0
        total: int = len(payloads)
        future: cf.Future[dict[str, Any]]
        for future in cf.as_completed(future_by_shard):
            shard_index = future_by_shard[future]
            result: dict[str, Any] = future.result()
            shard_results.append(result)
            completed += 1
            print(
                f"[map-reduce] Completed shard {shard_index} "
                f"({completed}/{total}) docs_with_tokens={result['docs_with_tokens']}"
            )

    print("[map-reduce] Starting reduce phase.")
    df_counter: Counter[str] = Counter()
    tf_total_counter: Counter[str] = Counter()
    term_sources: dict[str, set[str]] = {}
    source_df_counter: dict[str, Counter[str]] = {}
    source_doc_hits: Counter[str] = Counter()
    source_tf_hits: Counter[str] = Counter()
    docs_with_tokens: int = 0
    source_term_totals: Counter[str] = Counter()
    doc_normalization_stats: Counter[str] = Counter()
    fallback_reasons: list[str] = []
    normalizer_effective: str = str(args.normalizer)

    shard_result: dict[str, Any]
    for shard_result in sorted(shard_results, key=lambda item: int(item["shard_index"])):
        docs_with_tokens += int(shard_result["docs_with_tokens"])
        source_term_totals.update(
            {
                key: int(value)
                for key, value in dict(shard_result["source_term_totals"]).items()
            }
        )
        doc_normalization_stats.update(
            {
                key: int(value)
                for key, value in dict(shard_result["doc_normalization_stats"]).items()
            }
        )
        if shard_result.get("fallback_reason") is not None:
            fallback_reasons.append(str(shard_result["fallback_reason"]))
            normalizer_effective = "simple"

        artifact_path = Path(str(shard_result["artifact_path"]))
        with artifact_path.open("rb") as fin:
            shard_payload: dict[str, Any] = pickle.load(fin)
        df_counter.update(Counter(shard_payload["df_counter"]))
        tf_total_counter.update(Counter(shard_payload["tf_total_counter"]))
        source_doc_hits.update(Counter(shard_payload["source_doc_hits"]))
        source_tf_hits.update(Counter(shard_payload["source_tf_hits"]))

        shard_term_sources_raw: dict[str, set[str]] = dict(shard_payload["term_sources"])
        term: str
        sources: set[str]
        for term, sources in shard_term_sources_raw.items():
            known_sources: set[str] | None = term_sources.get(term)
            if known_sources is None:
                term_sources[term] = set(sources)
            else:
                known_sources.update(set(sources))

        shard_source_df_raw: dict[str, Counter[str]] = dict(shard_payload["source_df_counter"])
        source: str
        source_counter: Counter[str]
        for source, source_counter in shard_source_df_raw.items():
            aggregate_counter: Counter[str] = source_df_counter.setdefault(source, Counter())
            aggregate_counter.update(Counter(source_counter))

        if bool(args.map_reduce_cleanup):
            artifact_path.unlink(missing_ok=True)

    if bool(args.map_reduce_cleanup):
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    stats: dict[str, Any] = {
        "docs_with_tokens": int(docs_with_tokens),
        "spacy_model": str(args.spacy_model),
        "spacy_batch_size": int(args.spacy_batch_size),
        "spacy_n_process": int(args.spacy_n_process),
        "normalizer": normalizer_effective,
        "entity_label_filter": (
            sorted(allowed_entity_labels) if allowed_entity_labels is not None else None
        ),
        "extraction": {
            "extract_entities": bool(args.spacy_extract_entities),
            "extract_noun_chunks": bool(args.spacy_extract_noun_chunks),
            "entity_min_tokens": int(args.entity_min_tokens),
            "entity_max_tokens": int(args.entity_max_tokens),
            "noun_chunk_min_tokens": int(args.noun_chunk_min_tokens),
            "noun_chunk_max_tokens": int(args.noun_chunk_max_tokens),
            "noun_chunk_normalization": str(args.noun_chunk_normalization),
            "noun_chunk_max_stopword_ratio": float(args.noun_chunk_max_stopword_ratio),
            "max_phrase_chars": int(args.max_phrase_chars),
            "normalize_leading_determiners": bool(args.normalize_leading_determiners),
            "leading_determiners": sorted(leading_determiners),
            "normalize_entity_determiners": bool(args.normalize_entity_determiners),
            "normalize_noun_chunk_determiners": bool(
                args.normalize_noun_chunk_determiners
            ),
        },
        "source_term_totals": {
            _SOURCE_TOKEN: int(source_term_totals.get(_SOURCE_TOKEN, 0)),
            _SOURCE_ENTITY: int(source_term_totals.get(_SOURCE_ENTITY, 0)),
            _SOURCE_NOUN_CHUNK: int(source_term_totals.get(_SOURCE_NOUN_CHUNK, 0)),
        },
        "doc_normalization_stats": {
            key: int(value) for key, value in sorted(doc_normalization_stats.items())
        },
        "map_reduce": {
            "enabled": True,
            "shard_count": int(shard_count),
            "worker_count": int(worker_count),
            "tmp_dir": str(temp_dir),
            "cleanup": bool(args.map_reduce_cleanup),
            "fallback_count": len(fallback_reasons),
        },
    }
    if fallback_reasons:
        stats["fallback_reasons"] = fallback_reasons

    return (
        df_counter,
        tf_total_counter,
        term_sources,
        source_df_counter,
        source_doc_hits,
        source_tf_hits,
        docs_with_tokens,
        stats,
    )


def _select_vocab_from_statistics(
    *,
    df_counter: Counter[str],
    tf_total_counter: Counter[str],
    term_sources: dict[str, set[str]],
    source_df_counter: dict[str, Counter[str]],
    source_doc_hits: Counter[str],
    source_tf_hits: Counter[str],
    doc_count: int,
    target_size: int,
    min_df: int,
    stopword_df_ratio: float,
    filter_stopwords: bool,
    stopwords: set[str] | None,
    stopword_filter_phrases: bool,
    stopword_list_source: str | None,
    filter_noise_terms: bool,
    noise_max_digit_ratio: float,
    noise_max_symbol_ratio: float,
    noise_drop_single_char: bool,
    noise_drop_pure_numeric: bool,
    noise_drop_mojibake: bool,
    filter_url_like_terms: bool,
    filter_template_terms: bool,
    filter_function_leading_phrases: bool,
    function_leading_words: set[str],
    function_leading_require_noun_chunk: bool,
    function_leading_keep_entity_backed: bool,
    filter_contraction_artifacts: bool,
    filter_structured_artifacts: bool,
    filter_html_entity_artifacts: bool,
    html_entity_blacklist: set[str],
    filter_pronoun_led_phrases: bool,
    pronoun_leading_words: set[str],
    filter_letter_number_phrases: bool,
    letter_number_phrase_whitelist: set[str],
    filter_closed_class_terms: bool,
    closed_class_words: set[str],
    filter_high_df_unigrams: bool,
    high_df_unigram_ratio: float,
    filter_generic_unigram_blacklist: bool,
    generic_unigram_blacklist: set[str],
    filter_pos_gate: bool,
    pos_gate_allowed_tags: set[str],
    pos_gate_batch_size: int,
    filter_noisy_numeric_terms: bool,
    numeric_term_max_tokens: int,
    canonicalize_terms_for_selection: bool,
    canonical_strip_leading_determiners: bool,
    canonical_leading_determiners: set[str],
    normalize_noun_forms: bool,
    noun_normalization_skip_entity_backed: bool,
    noun_normalization_include_phrases: bool,
    noun_normalization_exceptions: set[str],
    downweight_generic_unigrams: bool,
    generic_unigram_df_ratio_start: float,
    generic_unigram_min_multiplier: float,
    generic_unigram_penalty_power: float,
    filter_low_cohesion_phrases: bool,
    phrase_cohesion_method: str,
    phrase_cohesion_min_score: float,
    phrase_cohesion_min_df: int,
    phrase_cohesion_require_noun_chunk: bool,
    phrase_cohesion_entity_exempt: bool,
    entity_quality_gate: bool,
    entity_quality_min_source_df: int,
    entity_quality_min_source_ratio: float,
    entity_quality_min_df_entity_only: int,
    source_boosts: dict[str, float],
) -> tuple[list[str], dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    if doc_count <= 0:
        raise RuntimeError("No valid documents were collected for vocabulary construction.")
    if bool(filter_stopwords) and not stopwords:
        raise ValueError("filter_stopwords=true requires a non-empty stopword set.")

    terms_after_stopword_filter: list[str] = []
    stopword_filtered_terms: int = 0
    term: str
    for term in df_counter.keys():
        if bool(filter_stopwords) and _is_stopword_term(
            term=term,
            stopwords=stopwords or set(),
            filter_phrases=bool(stopword_filter_phrases),
        ):
            stopword_filtered_terms += 1
            continue
        terms_after_stopword_filter.append(term)

    terms_after_noise_filter: list[str] = []
    noise_filtered_terms: int = 0
    noise_filtered_reasons: Counter[str] = Counter()
    for term in terms_after_stopword_filter:
        if bool(filter_noise_terms):
            noise_reason: str | None = _noise_term_reason(
                term=term,
                max_digit_ratio=float(noise_max_digit_ratio),
                max_symbol_ratio=float(noise_max_symbol_ratio),
                drop_single_char=bool(noise_drop_single_char),
                drop_pure_numeric=bool(noise_drop_pure_numeric),
                drop_mojibake=bool(noise_drop_mojibake),
                drop_url_like=bool(filter_url_like_terms),
                drop_template_like=bool(filter_template_terms),
            )
            if noise_reason is not None:
                noise_filtered_terms += 1
                noise_filtered_reasons[noise_reason] += 1
                continue
        terms_after_noise_filter.append(term)

    terms_after_function_filter: list[str] = []
    function_leading_filtered_terms: int = 0
    for term in terms_after_noise_filter:
        sources_for_term: set[str] = set(term_sources.get(term, {_SOURCE_TOKEN}))
        if bool(filter_function_leading_phrases) and _is_function_leading_phrase(
            term=term,
            sources=sources_for_term,
            function_leading_words=function_leading_words,
            require_noun_chunk_source=bool(function_leading_require_noun_chunk),
            keep_entity_backed=bool(function_leading_keep_entity_backed),
        ):
            function_leading_filtered_terms += 1
            continue
        terms_after_function_filter.append(term)

    selection_df_counter: Counter[str]
    selection_tf_total_counter: Counter[str]
    selection_term_sources: dict[str, set[str]]
    selection_source_df_counter: dict[str, Counter[str]]
    selection_variant_count: Counter[str]
    terms_after_canonicalization: list[str]
    canonical_variant_count: Counter[str] = Counter()
    canonical_dropped_empty: int = 0
    canonical_merged_terms: int = 0

    if bool(canonicalize_terms_for_selection):
        selection_df_counter = Counter()
        selection_tf_total_counter = Counter()
        selection_term_sources = {}
        selection_source_df_counter = {
            source: Counter() for source in source_df_counter.keys()
        }
        for term in terms_after_function_filter:
            canonical_term: str | None = _canonicalize_term_for_selection(
                term=term,
                strip_leading_determiners=bool(canonical_strip_leading_determiners),
                leading_determiners=canonical_leading_determiners,
            )
            if canonical_term is None:
                canonical_dropped_empty += 1
                continue
            canonical_variant_count[canonical_term] += 1

            aggregated_df: int = (
                int(selection_df_counter[canonical_term]) + int(df_counter[term])
            )
            if aggregated_df > int(doc_count):
                aggregated_df = int(doc_count)
            selection_df_counter[canonical_term] = aggregated_df
            selection_tf_total_counter[canonical_term] += int(tf_total_counter[term])

            known_sources: set[str] | None = selection_term_sources.get(canonical_term)
            term_source_values: set[str] = set(term_sources.get(term, {_SOURCE_TOKEN}))
            if known_sources is None:
                selection_term_sources[canonical_term] = set(term_source_values)
            else:
                known_sources.update(term_source_values)

            source: str
            source_counter: Counter[str]
            for source, source_counter in source_df_counter.items():
                source_df_value: int = int(source_counter.get(term, 0))
                if source_df_value <= 0:
                    continue
                merged_source_counter: Counter[str] = selection_source_df_counter.setdefault(
                    source, Counter()
                )
                merged_source_df: int = (
                    int(merged_source_counter[canonical_term]) + source_df_value
                )
                if merged_source_df > int(doc_count):
                    merged_source_df = int(doc_count)
                merged_source_counter[canonical_term] = merged_source_df

        terms_after_canonicalization = list(selection_df_counter.keys())
        canonical_merged_terms = int(
            len(terms_after_function_filter) - len(terms_after_canonicalization)
        )
        selection_variant_count = Counter(canonical_variant_count)
    else:
        selection_df_counter = df_counter
        selection_tf_total_counter = tf_total_counter
        selection_term_sources = term_sources
        selection_source_df_counter = source_df_counter
        terms_after_canonicalization = list(terms_after_function_filter)
        selection_variant_count = Counter(
            {term: 1 for term in terms_after_canonicalization}
        )

    terms_after_postcanonical_filter: list[str] = []
    postcanonical_stopword_filtered_terms: int = 0
    postcanonical_noise_filtered_terms: int = 0
    postcanonical_noise_filtered_reasons: Counter[str] = Counter()
    postcanonical_function_filtered_terms: int = 0
    for term in terms_after_canonicalization:
        if bool(filter_stopwords) and _is_stopword_term(
            term=term,
            stopwords=stopwords or set(),
            filter_phrases=bool(stopword_filter_phrases),
        ):
            postcanonical_stopword_filtered_terms += 1
            continue
        if bool(filter_noise_terms):
            postcanonical_noise_reason: str | None = _noise_term_reason(
                term=term,
                max_digit_ratio=float(noise_max_digit_ratio),
                max_symbol_ratio=float(noise_max_symbol_ratio),
                drop_single_char=bool(noise_drop_single_char),
                drop_pure_numeric=bool(noise_drop_pure_numeric),
                drop_mojibake=bool(noise_drop_mojibake),
                drop_url_like=bool(filter_url_like_terms),
                drop_template_like=bool(filter_template_terms),
            )
            if postcanonical_noise_reason is not None:
                postcanonical_noise_filtered_terms += 1
                postcanonical_noise_filtered_reasons[postcanonical_noise_reason] += 1
                continue
        postcanonical_sources: set[str] = set(
            selection_term_sources.get(term, {_SOURCE_TOKEN})
        )
        if bool(filter_function_leading_phrases) and _is_function_leading_phrase(
            term=term,
            sources=postcanonical_sources,
            function_leading_words=function_leading_words,
            require_noun_chunk_source=bool(function_leading_require_noun_chunk),
            keep_entity_backed=bool(function_leading_keep_entity_backed),
        ):
            postcanonical_function_filtered_terms += 1
            continue
        terms_after_postcanonical_filter.append(term)

    terms_after_noun_normalization: list[str] = list(terms_after_postcanonical_filter)
    noun_normalization_dropped_empty_terms: int = 0
    noun_normalization_merged_terms: int = 0
    noun_normalization_stats: dict[str, Any] = {
        "enabled": bool(normalize_noun_forms),
        "total_terms": int(len(terms_after_postcanonical_filter)),
        "normalized_terms": 0,
        "normalized_unigrams": 0,
        "normalized_phrases": 0,
        "skip_entity_backed": bool(noun_normalization_skip_entity_backed),
        "include_phrases": bool(noun_normalization_include_phrases),
        "exception_count": int(len(noun_normalization_exceptions)),
        "wordnet_available": bool(_get_wordnet_lemmatizer() is not None),
        "reason_counts": {},
        "dropped_empty_terms": 0,
        "merged_terms": 0,
        "terms_after_normalization": int(len(terms_after_postcanonical_filter)),
    }
    if bool(normalize_noun_forms):
        noun_normalized_map, noun_normalization_stats = (
            _normalize_noun_forms_with_hybrid_agreement(
                terms=terms_after_postcanonical_filter,
                term_sources=selection_term_sources,
                pos_batch_size=int(pos_gate_batch_size),
                skip_entity_backed=bool(noun_normalization_skip_entity_backed),
                include_phrases=bool(noun_normalization_include_phrases),
                exception_words=noun_normalization_exceptions,
            )
        )
        merged_df_counter: Counter[str] = Counter()
        merged_tf_counter: Counter[str] = Counter()
        merged_term_sources: dict[str, set[str]] = {}
        merged_source_df_counter: dict[str, Counter[str]] = {
            source: Counter() for source in selection_source_df_counter.keys()
        }
        merged_variant_count: Counter[str] = Counter()
        for term in terms_after_postcanonical_filter:
            normalized_term: str = str(noun_normalized_map.get(term, term)).strip()
            if not normalized_term:
                noun_normalization_dropped_empty_terms += 1
                continue
            merged_df: int = int(merged_df_counter[normalized_term]) + int(
                selection_df_counter.get(term, 0)
            )
            if merged_df > int(doc_count):
                merged_df = int(doc_count)
            merged_df_counter[normalized_term] = merged_df
            merged_tf_counter[normalized_term] += int(selection_tf_total_counter.get(term, 0))

            known_sources: set[str] | None = merged_term_sources.get(normalized_term)
            term_source_values: set[str] = set(
                selection_term_sources.get(term, {_SOURCE_TOKEN})
            )
            if known_sources is None:
                merged_term_sources[normalized_term] = set(term_source_values)
            else:
                known_sources.update(term_source_values)

            merged_variant_count[normalized_term] += int(selection_variant_count.get(term, 1))

            source: str
            source_counter: Counter[str]
            for source, source_counter in selection_source_df_counter.items():
                source_df_value: int = int(source_counter.get(term, 0))
                if source_df_value <= 0:
                    continue
                merged_source_counter: Counter[str] = merged_source_df_counter.setdefault(
                    source, Counter()
                )
                merged_source_df: int = (
                    int(merged_source_counter[normalized_term]) + source_df_value
                )
                if merged_source_df > int(doc_count):
                    merged_source_df = int(doc_count)
                merged_source_counter[normalized_term] = merged_source_df

        selection_df_counter = merged_df_counter
        selection_tf_total_counter = merged_tf_counter
        selection_term_sources = merged_term_sources
        selection_source_df_counter = merged_source_df_counter
        selection_variant_count = merged_variant_count
        terms_after_noun_normalization = list(selection_df_counter.keys())
        noun_normalization_merged_terms = int(
            len(terms_after_postcanonical_filter)
            - int(noun_normalization_dropped_empty_terms)
            - len(terms_after_noun_normalization)
        )
        noun_normalization_stats["dropped_empty_terms"] = int(
            noun_normalization_dropped_empty_terms
        )
        noun_normalization_stats["merged_terms"] = int(noun_normalization_merged_terms)
        noun_normalization_stats["terms_after_normalization"] = int(
            len(terms_after_noun_normalization)
        )

    terms_after_artifact_filter: list[str] = []
    contraction_artifact_filtered_terms: int = 0
    contraction_artifact_filtered_reasons: Counter[str] = Counter()
    structured_artifact_filtered_terms: int = 0
    structured_artifact_filtered_reasons: Counter[str] = Counter()
    closed_class_filtered_terms: int = 0
    for term in terms_after_noun_normalization:
        if bool(filter_contraction_artifacts):
            contraction_reason: str | None = _contraction_artifact_reason(term=term)
            if contraction_reason is not None:
                contraction_artifact_filtered_terms += 1
                contraction_artifact_filtered_reasons[contraction_reason] += 1
                continue
        if bool(filter_structured_artifacts):
            structured_reason: str | None = _structured_artifact_reason(
                term=term,
                filter_html_entity_artifacts=bool(filter_html_entity_artifacts),
                html_entity_blacklist=html_entity_blacklist,
                filter_pronoun_led_phrases=bool(filter_pronoun_led_phrases),
                pronoun_leading_words=pronoun_leading_words,
                filter_letter_number_phrases=bool(filter_letter_number_phrases),
                letter_number_phrase_whitelist=letter_number_phrase_whitelist,
            )
            if structured_reason is not None:
                structured_artifact_filtered_terms += 1
                structured_artifact_filtered_reasons[structured_reason] += 1
                continue
        if bool(filter_closed_class_terms) and _is_closed_class_blacklisted_term(
            term=term,
            closed_class_words=closed_class_words,
        ):
            closed_class_filtered_terms += 1
            continue
        terms_after_artifact_filter.append(term)

    terms_after_generic_unigram_filter: list[str] = []
    high_df_unigram_filtered_terms: int = 0
    generic_unigram_blacklist_filtered_terms: int = 0
    for term in terms_after_artifact_filter:
        parts = [part for part in str(term).split() if part]
        if len(parts) != 1:
            terms_after_generic_unigram_filter.append(term)
            continue
        term_df_value: int = int(selection_df_counter.get(term, 0))
        if (
            bool(filter_high_df_unigrams)
            and int(doc_count) > 0
            and float(term_df_value) / float(doc_count) > float(high_df_unigram_ratio)
        ):
            high_df_unigram_filtered_terms += 1
            continue
        if bool(filter_generic_unigram_blacklist) and str(term).lower() in generic_unigram_blacklist:
            generic_unigram_blacklist_filtered_terms += 1
            continue
        terms_after_generic_unigram_filter.append(term)

    terms_after_entity_quality_filter: list[str] = []
    entity_quality_filtered_terms: int = 0
    entity_quality_filtered_entity_only_terms: int = 0
    entity_source_counter: Counter[str] = selection_source_df_counter.get(
        _SOURCE_ENTITY, Counter()
    )
    for term in terms_after_generic_unigram_filter:
        if not bool(entity_quality_gate):
            terms_after_entity_quality_filter.append(term)
            continue
        entity_df_value: int = int(entity_source_counter.get(term, 0))
        if entity_df_value <= 0:
            terms_after_entity_quality_filter.append(term)
            continue
        term_df_value: int = int(selection_df_counter.get(term, 0))
        if term_df_value <= 0:
            continue
        term_source_set: set[str] = set(selection_term_sources.get(term, {_SOURCE_TOKEN}))
        entity_ratio: float = float(entity_df_value) / float(term_df_value)
        is_entity_only: bool = term_source_set == {_SOURCE_ENTITY}
        if is_entity_only and (
            term_df_value < int(entity_quality_min_df_entity_only)
            or entity_df_value < int(entity_quality_min_source_df)
        ):
            entity_quality_filtered_terms += 1
            entity_quality_filtered_entity_only_terms += 1
            continue
        if (
            (not is_entity_only)
            and entity_df_value < int(entity_quality_min_source_df)
            and entity_ratio < float(entity_quality_min_source_ratio)
        ):
            entity_quality_filtered_terms += 1
            continue
        terms_after_entity_quality_filter.append(term)

    terms_after_phrase_cohesion_filter: list[str] = []
    phrase_cohesion_filtered_low_support_terms: int = 0
    phrase_cohesion_filtered_low_score_terms: int = 0
    phrase_cohesion_score_by_term: dict[str, float] = {}
    for term in terms_after_entity_quality_filter:
        parts: list[str] = [part for part in term.split() if part]
        if len(parts) < 2:
            terms_after_phrase_cohesion_filter.append(term)
            continue
        if not bool(filter_low_cohesion_phrases):
            terms_after_phrase_cohesion_filter.append(term)
            continue
        term_source_set = set(selection_term_sources.get(term, {_SOURCE_TOKEN}))
        if (
            bool(phrase_cohesion_require_noun_chunk)
            and _SOURCE_NOUN_CHUNK not in term_source_set
        ):
            terms_after_phrase_cohesion_filter.append(term)
            continue
        if bool(phrase_cohesion_entity_exempt) and _SOURCE_ENTITY in term_source_set:
            terms_after_phrase_cohesion_filter.append(term)
            continue
        phrase_df: int = int(selection_df_counter.get(term, 0))
        if phrase_df < int(phrase_cohesion_min_df):
            phrase_cohesion_filtered_low_support_terms += 1
            continue
        cohesion_score: float | None = _phrase_cohesion_score(
            term=term,
            df_counter=selection_df_counter,
            doc_count=int(doc_count),
            method=str(phrase_cohesion_method),
        )
        if cohesion_score is None:
            terms_after_phrase_cohesion_filter.append(term)
            continue
        phrase_cohesion_score_by_term[term] = float(cohesion_score)
        if float(cohesion_score) < float(phrase_cohesion_min_score):
            phrase_cohesion_filtered_low_score_terms += 1
            continue
        terms_after_phrase_cohesion_filter.append(term)

    max_df_threshold: float = float(stopword_df_ratio) * float(doc_count)
    candidate_terms_before_pos_gate: list[str] = [
        term
        for term in terms_after_phrase_cohesion_filter
        for df_value in [selection_df_counter[term]]
        if int(df_value) >= int(min_df) and float(df_value) <= max_df_threshold
    ]
    candidate_terms: list[str] = list(candidate_terms_before_pos_gate)
    pos_gate_kept_pos_counts: Counter[str] = Counter()
    pos_gate_filtered_pos_counts: Counter[str] = Counter()
    pos_gate_tag_by_term: dict[str, str] = {}
    if bool(filter_pos_gate):
        (
            candidate_terms,
            pos_gate_kept_pos_counts,
            pos_gate_filtered_pos_counts,
            pos_gate_tag_by_term,
        ) = _apply_pos_gate_to_terms(
            terms=candidate_terms_before_pos_gate,
            allowed_tags=pos_gate_allowed_tags,
            batch_size=int(pos_gate_batch_size),
        )
    candidate_terms_before_numeric_quality: list[str] = list(candidate_terms)
    numeric_quality_filtered_terms: int = 0
    numeric_quality_reason_counts: Counter[str] = Counter()
    if bool(filter_noisy_numeric_terms):
        filtered_candidate_terms: list[str] = []
        term: str
        for term in candidate_terms_before_numeric_quality:
            pos_tag: str = str(pos_gate_tag_by_term.get(term, "")).upper()
            if pos_tag == "NUM":
                numeric_reason: str | None = _numeric_term_quality_reason(
                    term=term,
                    max_tokens=int(numeric_term_max_tokens),
                )
                if numeric_reason is not None:
                    numeric_quality_filtered_terms += 1
                    numeric_quality_reason_counts[numeric_reason] += 1
                    continue
            filtered_candidate_terms.append(term)
        candidate_terms = filtered_candidate_terms
    pos_gate_filtered_terms: int = int(
        len(candidate_terms_before_pos_gate) - len(candidate_terms_before_numeric_quality)
    )

    utility_by_term: dict[str, float] = {}
    idf_by_term: dict[str, float] = {}
    utility_raw_by_term: dict[str, float] = {}
    source_boost_by_term: dict[str, float] = {}
    generic_unigram_penalty_by_term: dict[str, float] = {}
    generic_unigram_penalized_terms: int = 0
    for term in candidate_terms:
        df_value: int = int(selection_df_counter[term])
        tf_total: int = int(selection_tf_total_counter[term])
        idf: float = math.log(
            ((float(doc_count) - float(df_value) + 0.5) / (float(df_value) + 0.5)) + 1.0
        )
        idf_by_term[term] = idf
        utility_raw: float = float(tf_total) * idf
        term_boost: float = _resolve_term_source_boost(
            term=term,
            source_df_counter=selection_source_df_counter,
            source_boosts=source_boosts,
        )
        utility_raw_by_term[term] = utility_raw
        source_boost_by_term[term] = term_boost
        unigram_penalty: float = 1.0
        if bool(downweight_generic_unigrams):
            unigram_penalty = _generic_unigram_df_penalty(
                term=term,
                term_df=int(df_value),
                doc_count=int(doc_count),
                start_ratio=float(generic_unigram_df_ratio_start),
                end_ratio=float(stopword_df_ratio),
                min_multiplier=float(generic_unigram_min_multiplier),
                penalty_power=float(generic_unigram_penalty_power),
            )
            if float(unigram_penalty) < 0.999999:
                generic_unigram_penalized_terms += 1
        generic_unigram_penalty_by_term[term] = float(unigram_penalty)
        utility_by_term[term] = utility_raw * term_boost * float(unigram_penalty)

    ranked_terms: list[str] = sorted(
        candidate_terms,
        key=lambda token: (-utility_by_term[token], token),
    )

    selected_terms: list[str] = ranked_terms[: int(target_size)]
    selected_df_map: dict[str, int] = {
        term: int(selection_df_counter[term]) for term in selected_terms
    }

    selected_stats: list[dict[str, Any]] = []
    rank: int
    for rank, term in enumerate(selected_terms, start=1):
        selected_sources: list[str] = sorted(
            selection_term_sources.get(term, {_SOURCE_TOKEN})
        )
        source_df: dict[str, int] = {}
        source: str
        for source in selected_sources:
            source_df[source] = int(
                selection_source_df_counter.get(source, Counter()).get(term, 0)
            )
        selected_stats.append(
            {
                "rank": rank,
                "term": term,
                "df": int(selection_df_counter[term]),
                "tf_total": int(selection_tf_total_counter[term]),
                "idf": float(idf_by_term[term]),
                "utility_raw": float(utility_raw_by_term[term]),
                "source_boost": float(source_boost_by_term[term]),
                "generic_unigram_multiplier": float(
                    generic_unigram_penalty_by_term.get(term, 1.0)
                ),
                "utility": float(utility_by_term[term]),
                "sources": selected_sources,
                "source_df": source_df,
                "pos_tag": pos_gate_tag_by_term.get(term) if bool(filter_pos_gate) else None,
                "phrase_cohesion_score": (
                    float(phrase_cohesion_score_by_term[term])
                    if term in phrase_cohesion_score_by_term
                    else None
                ),
                "variant_count": (
                    int(selection_variant_count.get(term, 1))
                ),
            }
        )

    summary: dict[str, Any] = {
        "doc_count": int(doc_count),
        "unique_terms": len(df_counter),
        "terms_after_stopword_filter": len(terms_after_stopword_filter),
        "stopword_filtered_terms": int(stopword_filtered_terms),
        "terms_after_noise_filter": len(terms_after_noise_filter),
        "noise_filtered_terms": int(noise_filtered_terms),
        "terms_after_function_filter": len(terms_after_function_filter),
        "function_leading_filtered_terms": int(function_leading_filtered_terms),
        "terms_after_canonicalization": len(terms_after_canonicalization),
        "terms_after_postcanonical_filter": len(terms_after_postcanonical_filter),
        "terms_after_noun_normalization": len(terms_after_noun_normalization),
        "noun_normalization_dropped_empty_terms": int(
            noun_normalization_dropped_empty_terms
        ),
        "noun_normalization_merged_terms": int(noun_normalization_merged_terms),
        "terms_after_artifact_filter": len(terms_after_artifact_filter),
        "contraction_artifact_filtered_terms": int(contraction_artifact_filtered_terms),
        "structured_artifact_filtered_terms": int(structured_artifact_filtered_terms),
        "closed_class_filtered_terms": int(closed_class_filtered_terms),
        "terms_after_generic_unigram_filter": len(terms_after_generic_unigram_filter),
        "high_df_unigram_filtered_terms": int(high_df_unigram_filtered_terms),
        "generic_unigram_blacklist_filtered_terms": int(
            generic_unigram_blacklist_filtered_terms
        ),
        "terms_after_entity_quality_filter": len(terms_after_entity_quality_filter),
        "entity_quality_filtered_terms": int(entity_quality_filtered_terms),
        "entity_quality_filtered_entity_only_terms": int(
            entity_quality_filtered_entity_only_terms
        ),
        "terms_after_phrase_cohesion_filter": len(terms_after_phrase_cohesion_filter),
        "phrase_cohesion_filtered_low_support_terms": int(
            phrase_cohesion_filtered_low_support_terms
        ),
        "phrase_cohesion_filtered_low_score_terms": int(
            phrase_cohesion_filtered_low_score_terms
        ),
        "postcanonical_stopword_filtered_terms": int(
            postcanonical_stopword_filtered_terms
        ),
        "postcanonical_noise_filtered_terms": int(postcanonical_noise_filtered_terms),
        "postcanonical_function_filtered_terms": int(
            postcanonical_function_filtered_terms
        ),
        "canonicalization_dropped_empty_terms": int(canonical_dropped_empty),
        "canonicalization_merged_terms": int(canonical_merged_terms),
        "df_filtered_terms": int(
            len(terms_after_phrase_cohesion_filter) - len(candidate_terms_before_pos_gate)
        ),
        "pos_gate_filtered_terms": int(
            len(candidate_terms_before_pos_gate) - len(candidate_terms_before_numeric_quality)
        ),
        "numeric_quality_filtered_terms": int(numeric_quality_filtered_terms),
        "candidate_terms_before_pos_gate": len(candidate_terms_before_pos_gate),
        "candidate_terms_before_numeric_quality": len(
            candidate_terms_before_numeric_quality
        ),
        "candidate_terms": len(candidate_terms),
        "target_size": int(target_size),
        "selected_terms": len(selected_terms),
        "max_df_threshold": max_df_threshold,
        "min_df": int(min_df),
        "source_boost_mode": "source_df_weighted_average",
        "source_boosts": {key: float(value) for key, value in source_boosts.items()},
        "stopword_filter": {
            "enabled": bool(filter_stopwords),
            "filter_phrases": bool(stopword_filter_phrases),
            "stopword_list_source": stopword_list_source if bool(filter_stopwords) else None,
            "stopword_list_size": (
                int(len(stopwords or set())) if bool(filter_stopwords) else 0
            ),
        },
        "noise_filter": {
            "enabled": bool(filter_noise_terms),
            "max_digit_ratio": (
                float(noise_max_digit_ratio) if bool(filter_noise_terms) else None
            ),
            "max_symbol_ratio": (
                float(noise_max_symbol_ratio) if bool(filter_noise_terms) else None
            ),
            "drop_single_char": bool(noise_drop_single_char),
            "drop_pure_numeric": bool(noise_drop_pure_numeric),
            "drop_mojibake": bool(noise_drop_mojibake),
            "drop_url_like": bool(filter_url_like_terms),
            "drop_template_like": bool(filter_template_terms),
            "reason_counts": {
                key: int(value) for key, value in sorted(noise_filtered_reasons.items())
            },
        },
        "function_leading_filter": {
            "enabled": bool(filter_function_leading_phrases),
            "word_count": int(len(function_leading_words)),
            "require_noun_chunk_source": bool(function_leading_require_noun_chunk),
            "keep_entity_backed": bool(function_leading_keep_entity_backed),
            "filtered_terms": int(function_leading_filtered_terms),
        },
        "canonicalization": {
            "enabled": bool(canonicalize_terms_for_selection),
            "strip_leading_determiners": bool(canonical_strip_leading_determiners),
            "leading_determiners": (
                sorted(canonical_leading_determiners)
                if bool(canonicalize_terms_for_selection)
                else []
            ),
            "dropped_empty_terms": int(canonical_dropped_empty),
            "merged_terms": int(canonical_merged_terms),
        },
        "noun_form_normalization": noun_normalization_stats,
        "postcanonical_filtering": {
            "enabled": True,
            "stopword_filtered_terms": int(postcanonical_stopword_filtered_terms),
            "noise_filtered_terms": int(postcanonical_noise_filtered_terms),
            "noise_reason_counts": {
                key: int(value)
                for key, value in sorted(postcanonical_noise_filtered_reasons.items())
            },
            "function_filtered_terms": int(postcanonical_function_filtered_terms),
        },
        "artifact_filtering": {
            "enabled": bool(
                filter_contraction_artifacts
                or filter_structured_artifacts
                or filter_closed_class_terms
            ),
            "contraction_artifacts_enabled": bool(filter_contraction_artifacts),
            "contraction_artifact_filtered_terms": int(
                contraction_artifact_filtered_terms
            ),
            "contraction_artifact_reason_counts": {
                key: int(value)
                for key, value in sorted(contraction_artifact_filtered_reasons.items())
            },
            "structured_artifacts_enabled": bool(filter_structured_artifacts),
            "structured_artifact_filtered_terms": int(structured_artifact_filtered_terms),
            "structured_artifact_reason_counts": {
                key: int(value)
                for key, value in sorted(structured_artifact_filtered_reasons.items())
            },
            "html_entity_artifacts_enabled": bool(filter_html_entity_artifacts),
            "html_entity_blacklist": (
                sorted(html_entity_blacklist)
                if bool(filter_structured_artifacts and filter_html_entity_artifacts)
                else []
            ),
            "pronoun_led_phrases_enabled": bool(filter_pronoun_led_phrases),
            "pronoun_leading_words": (
                sorted(pronoun_leading_words)
                if bool(filter_structured_artifacts and filter_pronoun_led_phrases)
                else []
            ),
            "letter_number_phrases_enabled": bool(filter_letter_number_phrases),
            "letter_number_phrase_whitelist": (
                sorted(letter_number_phrase_whitelist)
                if bool(filter_structured_artifacts and filter_letter_number_phrases)
                else []
            ),
            "closed_class_enabled": bool(filter_closed_class_terms),
            "closed_class_filtered_terms": int(closed_class_filtered_terms),
            "closed_class_word_count": int(len(closed_class_words)),
            "closed_class_words": (
                sorted(closed_class_words) if bool(filter_closed_class_terms) else []
            ),
        },
        "generic_unigram_filtering": {
            "enabled": bool(
                filter_high_df_unigrams or filter_generic_unigram_blacklist
            ),
            "high_df_enabled": bool(filter_high_df_unigrams),
            "high_df_unigram_ratio": float(high_df_unigram_ratio),
            "high_df_unigram_filtered_terms": int(high_df_unigram_filtered_terms),
            "blacklist_enabled": bool(filter_generic_unigram_blacklist),
            "generic_unigram_blacklist_filtered_terms": int(
                generic_unigram_blacklist_filtered_terms
            ),
            "generic_unigram_blacklist_size": int(len(generic_unigram_blacklist)),
            "generic_unigram_blacklist": (
                sorted(generic_unigram_blacklist)
                if bool(filter_generic_unigram_blacklist)
                else []
            ),
        },
        "pos_gate": {
            "enabled": bool(filter_pos_gate),
            "allowed_tags": sorted(pos_gate_allowed_tags) if bool(filter_pos_gate) else [],
            "batch_size": int(pos_gate_batch_size),
            "filtered_terms": int(pos_gate_filtered_terms),
            "kept_terms": int(len(candidate_terms_before_numeric_quality)),
            "kept_pos_counts": {
                key: int(value)
                for key, value in sorted(
                    pos_gate_kept_pos_counts.items(), key=lambda item: item[0]
                )
            },
            "filtered_pos_counts": {
                key: int(value)
                for key, value in sorted(
                    pos_gate_filtered_pos_counts.items(), key=lambda item: item[0]
                )
            },
        },
        "numeric_quality_filter": {
            "enabled": bool(filter_noisy_numeric_terms),
            "max_tokens": int(numeric_term_max_tokens),
            "filtered_terms": int(numeric_quality_filtered_terms),
            "reason_counts": {
                key: int(value)
                for key, value in sorted(numeric_quality_reason_counts.items())
            },
        },
        "entity_quality_gate": {
            "enabled": bool(entity_quality_gate),
            "min_source_df": int(entity_quality_min_source_df),
            "min_source_ratio": float(entity_quality_min_source_ratio),
            "min_df_entity_only": int(entity_quality_min_df_entity_only),
            "filtered_terms": int(entity_quality_filtered_terms),
            "filtered_entity_only_terms": int(entity_quality_filtered_entity_only_terms),
        },
        "phrase_cohesion_filter": {
            "enabled": bool(filter_low_cohesion_phrases),
            "method": str(phrase_cohesion_method),
            "min_score": float(phrase_cohesion_min_score),
            "min_df": int(phrase_cohesion_min_df),
            "require_noun_chunk": bool(phrase_cohesion_require_noun_chunk),
            "entity_exempt": bool(phrase_cohesion_entity_exempt),
            "filtered_low_support_terms": int(
                phrase_cohesion_filtered_low_support_terms
            ),
            "filtered_low_score_terms": int(phrase_cohesion_filtered_low_score_terms),
            "scored_terms": int(len(phrase_cohesion_score_by_term)),
        },
        "generic_unigram_downweighting": {
            "enabled": bool(downweight_generic_unigrams),
            "df_ratio_start": float(generic_unigram_df_ratio_start),
            "df_ratio_end": float(stopword_df_ratio),
            "min_multiplier": float(generic_unigram_min_multiplier),
            "penalty_power": float(generic_unigram_penalty_power),
            "penalized_candidate_terms": int(generic_unigram_penalized_terms),
        },
        "terms_with_source": {
            _SOURCE_TOKEN: int(
                sum(1 for value in selection_term_sources.values() if _SOURCE_TOKEN in value)
            ),
            _SOURCE_ENTITY: int(
                sum(1 for value in selection_term_sources.values() if _SOURCE_ENTITY in value)
            ),
            _SOURCE_NOUN_CHUNK: int(
                sum(
                    1
                    for value in selection_term_sources.values()
                    if _SOURCE_NOUN_CHUNK in value
                )
            ),
        },
        "source_doc_hits": {
            key: int(value) for key, value in sorted(source_doc_hits.items())
        },
        "source_tf_hits": {
            key: int(value) for key, value in sorted(source_tf_hits.items())
        },
    }
    return selected_terms, selected_df_map, selected_stats, summary


def _build_vocab(
    *,
    docs_tokens: list[list[str]],
    docs_term_sources: list[dict[str, set[str]]],
    target_size: int,
    min_df: int,
    stopword_df_ratio: float,
    source_boosts: dict[str, float] | None = None,
    filter_stopwords: bool = False,
    stopwords: set[str] | None = None,
    stopword_filter_phrases: bool = False,
    stopword_list_source: str | None = None,
    filter_noise_terms: bool = True,
    noise_max_digit_ratio: float = 0.7,
    noise_max_symbol_ratio: float = 0.35,
    noise_drop_single_char: bool = True,
    noise_drop_pure_numeric: bool = True,
    noise_drop_mojibake: bool = True,
    filter_url_like_terms: bool = True,
    filter_template_terms: bool = True,
    filter_function_leading_phrases: bool = True,
    function_leading_words: set[str] | None = None,
    function_leading_require_noun_chunk: bool = True,
    function_leading_keep_entity_backed: bool = False,
    filter_contraction_artifacts: bool = True,
    filter_structured_artifacts: bool = True,
    filter_html_entity_artifacts: bool = True,
    html_entity_blacklist: set[str] | None = None,
    filter_pronoun_led_phrases: bool = True,
    pronoun_leading_words: set[str] | None = None,
    filter_letter_number_phrases: bool = True,
    letter_number_phrase_whitelist: set[str] | None = None,
    filter_closed_class_terms: bool = True,
    closed_class_words: set[str] | None = None,
    filter_high_df_unigrams: bool = True,
    high_df_unigram_ratio: float = 0.08,
    filter_generic_unigram_blacklist: bool = True,
    generic_unigram_blacklist: set[str] | None = None,
    filter_pos_gate: bool = True,
    pos_gate_allowed_tags: set[str] | None = None,
    pos_gate_batch_size: int = 2048,
    filter_noisy_numeric_terms: bool = True,
    numeric_term_max_tokens: int = 3,
    canonicalize_terms_for_selection: bool = True,
    canonical_strip_leading_determiners: bool = True,
    canonical_leading_determiners: set[str] | None = None,
    normalize_noun_forms: bool = True,
    noun_normalization_skip_entity_backed: bool = True,
    noun_normalization_include_phrases: bool = False,
    noun_normalization_exceptions: set[str] | None = None,
    downweight_generic_unigrams: bool = True,
    generic_unigram_df_ratio_start: float = 0.02,
    generic_unigram_min_multiplier: float = 0.35,
    generic_unigram_penalty_power: float = 1.0,
    filter_low_cohesion_phrases: bool = True,
    phrase_cohesion_method: str = "npmi",
    phrase_cohesion_min_score: float = -0.05,
    phrase_cohesion_min_df: int = 20,
    phrase_cohesion_require_noun_chunk: bool = True,
    phrase_cohesion_entity_exempt: bool = True,
    entity_quality_gate: bool = True,
    entity_quality_min_source_df: int = 5,
    entity_quality_min_source_ratio: float = 0.02,
    entity_quality_min_df_entity_only: int = 30,
) -> tuple[list[str], dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    if len(docs_tokens) != len(docs_term_sources):
        raise ValueError(
            "docs_tokens and docs_term_sources must have the same length."
        )
    df_counter: Counter[str] = Counter()
    tf_total_counter: Counter[str] = Counter()
    term_sources: dict[str, set[str]] = {}
    source_df_counter: dict[str, Counter[str]] = {}
    source_doc_hits: Counter[str] = Counter()
    source_tf_hits: Counter[str] = Counter()
    if source_boosts is None:
        source_boosts = {
            _SOURCE_TOKEN: 1.0,
            _SOURCE_NOUN_CHUNK: 1.25,
            _SOURCE_ENTITY: 1.5,
        }
    if function_leading_words is None:
        function_leading_words = {
            "this",
            "that",
            "these",
            "those",
            "any",
            "some",
            "many",
            "much",
            "more",
            "most",
            "less",
            "least",
            "few",
            "fewer",
            "all",
            "each",
            "every",
            "either",
            "neither",
            "another",
            "other",
            "such",
            "same",
            "no",
            "none",
            "both",
            "several",
            "various",
            "certain",
            "particular",
        }
    if canonical_leading_determiners is None:
        canonical_leading_determiners = {"the", "a", "an"}
    if html_entity_blacklist is None:
        html_entity_blacklist = {"amp", "nbsp", "lt", "gt"}
    if pronoun_leading_words is None:
        pronoun_leading_words = {"your", "my", "our", "their", "his", "her", "its"}
    if letter_number_phrase_whitelist is None:
        letter_number_phrase_whitelist = {
            "w 2",
            "w 4",
            "b 12",
            "k 12",
            "i 95",
            "i 90",
            "i 80",
            "i 75",
            "i 70",
            "i 40",
            "i 35",
            "i 20",
            "i 15",
            "i 10",
            "i 9",
            "i 5",
            "i 94",
            "k 1",
            "b 1",
            "b 2",
            "b 6",
            "r 22",
            "r 3",
            "f 1",
            "v 6",
            "v 8",
            "c 1",
            "m 2",
        }
    if closed_class_words is None:
        closed_class_words = {
            "may",
            "will",
            "must",
            "might",
            "shall",
            "could",
            "would",
            "should",
            "can",
            "cannot",
            "many",
            "much",
            "more",
            "most",
            "less",
            "least",
            "every",
            "another",
            "either",
            "neither",
            "none",
            "several",
            "various",
            "certain",
            "particular",
            "mine",
            "onto",
            "else",
            "done",
            "per",
            "im",
        }
    if generic_unigram_blacklist is None:
        generic_unigram_blacklist = {
            "use",
            "one",
            "also",
            "make",
            "get",
            "take",
            "find",
            "know",
            "like",
            "good",
            "well",
            "call",
            "work",
            "need",
            "first",
            "include",
            "just",
            "go",
        }
    if pos_gate_allowed_tags is None:
        pos_gate_allowed_tags = {"NOUN", "VERB", "ADJ"}
    if noun_normalization_exceptions is None:
        noun_normalization_exceptions = {
            "news",
            "series",
            "species",
            "means",
            "headquarters",
            "politics",
            "economics",
            "mathematics",
            "physics",
            "ethics",
            "statistics",
            "diabetes",
            "measles",
        }

    doc_count: int = 0
    tokens: list[str]
    doc_term_sources: dict[str, set[str]]
    for tokens, doc_term_sources in zip(docs_tokens, docs_term_sources):
        doc_count += 1
        tf_doc: Counter[str] = Counter(tokens)
        tf_total_counter.update(tf_doc)
        df_counter.update(tf_doc.keys())
        term: str
        tf_value: int
        for term, tf_value in tf_doc.items():
            sources: set[str] = set(doc_term_sources.get(term, {_SOURCE_TOKEN}))
            if not sources:
                sources = {_SOURCE_TOKEN}
            known_sources: set[str] | None = term_sources.get(term)
            if known_sources is None:
                term_sources[term] = set(sources)
            else:
                known_sources.update(sources)
            source: str
            for source in sources:
                counter: Counter[str] = source_df_counter.setdefault(source, Counter())
                counter[term] += 1
                source_doc_hits[source] += 1
                source_tf_hits[source] += int(tf_value)

    return _select_vocab_from_statistics(
        df_counter=df_counter,
        tf_total_counter=tf_total_counter,
        term_sources=term_sources,
        source_df_counter=source_df_counter,
        source_doc_hits=source_doc_hits,
        source_tf_hits=source_tf_hits,
        doc_count=doc_count,
        target_size=target_size,
        min_df=min_df,
        stopword_df_ratio=stopword_df_ratio,
        filter_stopwords=filter_stopwords,
        stopwords=stopwords,
        stopword_filter_phrases=stopword_filter_phrases,
        stopword_list_source=stopword_list_source,
        filter_noise_terms=filter_noise_terms,
        noise_max_digit_ratio=noise_max_digit_ratio,
        noise_max_symbol_ratio=noise_max_symbol_ratio,
        noise_drop_single_char=noise_drop_single_char,
        noise_drop_pure_numeric=noise_drop_pure_numeric,
        noise_drop_mojibake=noise_drop_mojibake,
        filter_url_like_terms=filter_url_like_terms,
        filter_template_terms=filter_template_terms,
        filter_function_leading_phrases=filter_function_leading_phrases,
        function_leading_words=function_leading_words,
        function_leading_require_noun_chunk=function_leading_require_noun_chunk,
        function_leading_keep_entity_backed=function_leading_keep_entity_backed,
        filter_contraction_artifacts=filter_contraction_artifacts,
        filter_structured_artifacts=filter_structured_artifacts,
        filter_html_entity_artifacts=filter_html_entity_artifacts,
        html_entity_blacklist=html_entity_blacklist,
        filter_pronoun_led_phrases=filter_pronoun_led_phrases,
        pronoun_leading_words=pronoun_leading_words,
        filter_letter_number_phrases=filter_letter_number_phrases,
        letter_number_phrase_whitelist=letter_number_phrase_whitelist,
        filter_closed_class_terms=filter_closed_class_terms,
        closed_class_words=closed_class_words,
        filter_high_df_unigrams=filter_high_df_unigrams,
        high_df_unigram_ratio=high_df_unigram_ratio,
        filter_generic_unigram_blacklist=filter_generic_unigram_blacklist,
        generic_unigram_blacklist=generic_unigram_blacklist,
        filter_pos_gate=filter_pos_gate,
        pos_gate_allowed_tags=pos_gate_allowed_tags,
        pos_gate_batch_size=pos_gate_batch_size,
        filter_noisy_numeric_terms=filter_noisy_numeric_terms,
        numeric_term_max_tokens=numeric_term_max_tokens,
        canonicalize_terms_for_selection=canonicalize_terms_for_selection,
        canonical_strip_leading_determiners=canonical_strip_leading_determiners,
        canonical_leading_determiners=canonical_leading_determiners,
        normalize_noun_forms=normalize_noun_forms,
        noun_normalization_skip_entity_backed=noun_normalization_skip_entity_backed,
        noun_normalization_include_phrases=noun_normalization_include_phrases,
        noun_normalization_exceptions=noun_normalization_exceptions,
        downweight_generic_unigrams=downweight_generic_unigrams,
        generic_unigram_df_ratio_start=generic_unigram_df_ratio_start,
        generic_unigram_min_multiplier=generic_unigram_min_multiplier,
        generic_unigram_penalty_power=generic_unigram_penalty_power,
        filter_low_cohesion_phrases=filter_low_cohesion_phrases,
        phrase_cohesion_method=phrase_cohesion_method,
        phrase_cohesion_min_score=phrase_cohesion_min_score,
        phrase_cohesion_min_df=phrase_cohesion_min_df,
        phrase_cohesion_require_noun_chunk=phrase_cohesion_require_noun_chunk,
        phrase_cohesion_entity_exempt=phrase_cohesion_entity_exempt,
        entity_quality_gate=entity_quality_gate,
        entity_quality_min_source_df=entity_quality_min_source_df,
        entity_quality_min_source_ratio=entity_quality_min_source_ratio,
        entity_quality_min_df_entity_only=entity_quality_min_df_entity_only,
        source_boosts=source_boosts,
    )


def _resolve_term_stats_cache_path(args: argparse.Namespace) -> Path:
    if args.term_stats_cache_path is not None and str(args.term_stats_cache_path).strip():
        return Path(str(args.term_stats_cache_path))
    return Path(str(args.output_dir)) / "term_statistics.pkl"


def _save_term_statistics_cache(
    *,
    cache_path: Path,
    df_counter: Counter[str],
    tf_total_counter: Counter[str],
    term_sources: dict[str, set[str]],
    source_df_counter: dict[str, Counter[str]],
    source_doc_hits: Counter[str],
    source_tf_hits: Counter[str],
    docs_with_tokens: int,
    source_stats: dict[str, Any],
    spacy_stats: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "df_counter": dict(df_counter),
        "tf_total_counter": dict(tf_total_counter),
        "term_sources": {
            term: sorted(sources) for term, sources in term_sources.items()
        },
        "source_df_counter": {
            source: dict(counter) for source, counter in source_df_counter.items()
        },
        "source_doc_hits": dict(source_doc_hits),
        "source_tf_hits": dict(source_tf_hits),
        "docs_with_tokens": int(docs_with_tokens),
        "source_stats": source_stats,
        "spacy_stats": spacy_stats,
    }
    with cache_path.open("wb") as fout:
        pickle.dump(payload, fout, protocol=pickle.HIGHEST_PROTOCOL)


def _load_term_statistics_cache(
    *,
    cache_path: Path,
) -> tuple[
    Counter[str],
    Counter[str],
    dict[str, set[str]],
    dict[str, Counter[str]],
    Counter[str],
    Counter[str],
    int,
    dict[str, Any],
    dict[str, Any],
]:
    if not cache_path.exists() or not cache_path.is_file():
        raise FileNotFoundError(f"term stats cache not found: {cache_path}")
    with cache_path.open("rb") as fin:
        payload: dict[str, Any] = pickle.load(fin)

    df_counter: Counter[str] = Counter(payload.get("df_counter", {}))
    tf_total_counter: Counter[str] = Counter(payload.get("tf_total_counter", {}))
    term_sources_raw: dict[str, Any] = dict(payload.get("term_sources", {}))
    term_sources: dict[str, set[str]] = {
        term: set(values) for term, values in term_sources_raw.items()
    }
    source_df_raw: dict[str, Any] = dict(payload.get("source_df_counter", {}))
    source_df_counter: dict[str, Counter[str]] = {
        source: Counter(counter) for source, counter in source_df_raw.items()
    }
    source_doc_hits: Counter[str] = Counter(payload.get("source_doc_hits", {}))
    source_tf_hits: Counter[str] = Counter(payload.get("source_tf_hits", {}))
    docs_with_tokens: int = int(payload.get("docs_with_tokens", 0))
    source_stats: dict[str, Any] = dict(payload.get("source_stats", {}))
    spacy_stats: dict[str, Any] = dict(payload.get("spacy_stats", {}))
    return (
        df_counter,
        tf_total_counter,
        term_sources,
        source_df_counter,
        source_doc_hits,
        source_tf_hits,
        docs_with_tokens,
        source_stats,
        spacy_stats,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args = _apply_config_overrides(args)
    _validate_required_args(args)

    entity_labels: set[str] | None = _parse_entity_labels(args.entity_labels)
    leading_determiners: set[str] = _parse_leading_determiners(args.leading_determiners)
    function_leading_words: set[str] = _parse_function_leading_words(
        args.function_leading_words
    )
    html_entity_blacklist: set[str] = _parse_html_entity_blacklist(
        args.html_entity_blacklist
    )
    pronoun_leading_words: set[str] = _parse_pronoun_leading_words(
        args.pronoun_leading_words
    )
    letter_number_phrase_whitelist: set[str] = _parse_letter_number_phrase_whitelist(
        args.letter_number_phrase_whitelist
    )
    closed_class_words: set[str] = _parse_closed_class_words(args.closed_class_words)
    generic_unigram_blacklist: set[str] = _parse_generic_unigram_blacklist(
        args.generic_unigram_blacklist
    )
    noun_normalization_exceptions: set[str] = _parse_noun_normalization_exceptions(
        args.noun_normalization_exceptions
    )
    pos_gate_allowed_tags: set[str] = _parse_pos_gate_tags(args.pos_gate_allowed_tags)
    source_boosts: dict[str, float] = {
        _SOURCE_TOKEN: float(args.token_source_boost),
        _SOURCE_NOUN_CHUNK: float(args.noun_chunk_source_boost),
        _SOURCE_ENTITY: float(args.entity_source_boost),
    }
    stopwords: set[str] | None = None
    stopword_list_source: str | None = None
    if bool(args.filter_stopwords):
        stopwords, stopword_list_source = _load_stopwords(
            stopword_list_path=args.stopword_list_path
        )

    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    term_stats_cache_path: Path = _resolve_term_stats_cache_path(args)

    source_stats: dict[str, Any]
    spacy_stats: dict[str, Any]
    df_counter: Counter[str]
    tf_total_counter: Counter[str]
    term_sources: dict[str, set[str]]
    source_df_counter: dict[str, Counter[str]]
    source_doc_hits: Counter[str]
    source_tf_hits: Counter[str]
    docs_with_tokens: int
    if bool(args.selection_only):
        (
            df_counter,
            tf_total_counter,
            term_sources,
            source_df_counter,
            source_doc_hits,
            source_tf_hits,
            docs_with_tokens,
            source_stats,
            spacy_stats,
        ) = _load_term_statistics_cache(cache_path=term_stats_cache_path)
        print(f"[selection-only] Loaded term statistics from {term_stats_cache_path}")
    else:
        meta_dataset: Any | None = None
        if not bool(args.use_all_corpus_documents):
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
        if bool(args.map_reduce_sharding):
            (
                df_counter,
                tf_total_counter,
                term_sources,
                source_df_counter,
                source_doc_hits,
                source_tf_hits,
                docs_with_tokens,
                spacy_stats,
            ) = _collect_term_statistics_map_reduce(
                args=args,
                allowed_entity_labels=entity_labels,
            )
        else:
            (
                df_counter,
                tf_total_counter,
                term_sources,
                source_df_counter,
                source_doc_hits,
                source_tf_hits,
                docs_with_tokens,
                spacy_stats,
            ) = _collect_term_statistics(
                texts=raw_texts,
                spacy_model=args.spacy_model,
                batch_size=args.spacy_batch_size,
                n_process=args.spacy_n_process,
                max_docs=args.max_docs,
                normalizer=args.normalizer,
                allow_simple_fallback=bool(args.allow_simple_fallback),
                extract_entities=bool(args.spacy_extract_entities),
                extract_noun_chunks=bool(args.spacy_extract_noun_chunks),
                allowed_entity_labels=entity_labels,
                entity_min_tokens=int(args.entity_min_tokens),
                entity_max_tokens=int(args.entity_max_tokens),
                noun_chunk_min_tokens=int(args.noun_chunk_min_tokens),
                noun_chunk_max_tokens=int(args.noun_chunk_max_tokens),
                noun_chunk_normalization=str(args.noun_chunk_normalization),
                noun_chunk_max_stopword_ratio=float(args.noun_chunk_max_stopword_ratio),
                max_phrase_chars=int(args.max_phrase_chars),
                normalize_leading_determiners=bool(args.normalize_leading_determiners),
                leading_determiners=leading_determiners,
                normalize_entity_determiners=bool(args.normalize_entity_determiners),
                normalize_noun_chunk_determiners=bool(args.normalize_noun_chunk_determiners),
            )
        if bool(args.save_term_stats_cache):
            _save_term_statistics_cache(
                cache_path=term_stats_cache_path,
                df_counter=df_counter,
                tf_total_counter=tf_total_counter,
                term_sources=term_sources,
                source_df_counter=source_df_counter,
                source_doc_hits=source_doc_hits,
                source_tf_hits=source_tf_hits,
                docs_with_tokens=docs_with_tokens,
                source_stats=source_stats,
                spacy_stats=spacy_stats,
            )
            print(f"[cache] Saved term statistics to {term_stats_cache_path}")

    v_target, df_map, selected_stats, summary = _select_vocab_from_statistics(
        df_counter=df_counter,
        tf_total_counter=tf_total_counter,
        term_sources=term_sources,
        source_df_counter=source_df_counter,
        source_doc_hits=source_doc_hits,
        source_tf_hits=source_tf_hits,
        doc_count=docs_with_tokens,
        target_size=args.target_size,
        min_df=args.min_df,
        stopword_df_ratio=args.stopword_df_ratio,
        filter_stopwords=bool(args.filter_stopwords),
        stopwords=stopwords,
        stopword_filter_phrases=bool(args.stopword_filter_phrases),
        stopword_list_source=stopword_list_source,
        filter_noise_terms=bool(args.filter_noise_terms),
        noise_max_digit_ratio=float(args.noise_max_digit_ratio),
        noise_max_symbol_ratio=float(args.noise_max_symbol_ratio),
        noise_drop_single_char=bool(args.noise_drop_single_char),
        noise_drop_pure_numeric=bool(args.noise_drop_pure_numeric),
        noise_drop_mojibake=bool(args.noise_drop_mojibake),
        filter_url_like_terms=bool(args.filter_url_like_terms),
        filter_template_terms=bool(args.filter_template_terms),
        filter_function_leading_phrases=bool(args.filter_function_leading_phrases),
        function_leading_words=function_leading_words,
        function_leading_require_noun_chunk=bool(
            args.function_leading_require_noun_chunk
        ),
        function_leading_keep_entity_backed=bool(
            args.function_leading_keep_entity_backed
        ),
        filter_contraction_artifacts=bool(args.filter_contraction_artifacts),
        filter_structured_artifacts=bool(args.filter_structured_artifacts),
        filter_html_entity_artifacts=bool(args.filter_structured_artifacts)
        and bool(args.filter_html_entity_artifacts),
        html_entity_blacklist=html_entity_blacklist,
        filter_pronoun_led_phrases=bool(args.filter_structured_artifacts)
        and bool(args.filter_pronoun_led_phrases),
        pronoun_leading_words=pronoun_leading_words,
        filter_letter_number_phrases=bool(args.filter_structured_artifacts)
        and bool(args.filter_letter_number_phrases),
        letter_number_phrase_whitelist=letter_number_phrase_whitelist,
        filter_closed_class_terms=bool(args.filter_closed_class_terms),
        closed_class_words=closed_class_words,
        filter_high_df_unigrams=bool(args.filter_high_df_unigrams),
        high_df_unigram_ratio=float(args.high_df_unigram_ratio),
        filter_generic_unigram_blacklist=bool(args.filter_generic_unigram_blacklist),
        generic_unigram_blacklist=generic_unigram_blacklist,
        filter_pos_gate=bool(args.filter_pos_gate),
        pos_gate_allowed_tags=pos_gate_allowed_tags,
        pos_gate_batch_size=int(args.pos_gate_batch_size),
        filter_noisy_numeric_terms=bool(args.filter_noisy_numeric_terms),
        numeric_term_max_tokens=int(args.numeric_term_max_tokens),
        canonicalize_terms_for_selection=bool(args.canonicalize_terms_for_selection),
        canonical_strip_leading_determiners=bool(
            args.canonical_strip_leading_determiners
        ),
        canonical_leading_determiners=leading_determiners,
        normalize_noun_forms=bool(args.normalize_noun_forms),
        noun_normalization_skip_entity_backed=bool(
            args.noun_normalization_skip_entity_backed
        ),
        noun_normalization_include_phrases=bool(
            args.noun_normalization_include_phrases
        ),
        noun_normalization_exceptions=noun_normalization_exceptions,
        downweight_generic_unigrams=bool(args.downweight_generic_unigrams),
        generic_unigram_df_ratio_start=float(args.generic_unigram_df_ratio_start),
        generic_unigram_min_multiplier=float(args.generic_unigram_min_multiplier),
        generic_unigram_penalty_power=float(args.generic_unigram_penalty_power),
        filter_low_cohesion_phrases=bool(args.filter_low_cohesion_phrases),
        phrase_cohesion_method=str(args.phrase_cohesion_method),
        phrase_cohesion_min_score=float(args.phrase_cohesion_min_score),
        phrase_cohesion_min_df=int(args.phrase_cohesion_min_df),
        phrase_cohesion_require_noun_chunk=bool(
            args.phrase_cohesion_require_noun_chunk
        ),
        phrase_cohesion_entity_exempt=bool(args.phrase_cohesion_entity_exempt),
        entity_quality_gate=bool(args.entity_quality_gate),
        entity_quality_min_source_df=int(args.entity_quality_min_source_df),
        entity_quality_min_source_ratio=float(args.entity_quality_min_source_ratio),
        entity_quality_min_df_entity_only=int(args.entity_quality_min_df_entity_only),
        source_boosts=source_boosts,
    )

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
                "term_stats_cache": {
                    "path": str(term_stats_cache_path),
                    "selection_only": bool(args.selection_only),
                    "saved": bool(not args.selection_only and args.save_term_stats_cache),
                },
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
