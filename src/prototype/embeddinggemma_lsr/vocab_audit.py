import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PROBES: tuple[str, ...] = (
    "i",
    "use",
    "will",
    "which",
    "more",
    "u s",
    "city of",
    "about 10 minutes",
    "sign uplog in",
)
TRAILING_FUNCTION_WORDS: frozenset[str] = frozenset(
    {"of", "and", "or", "to", "for", "in", "on", "with", "from", "by"}
)
YEAR_PATTERN: re.Pattern[str] = re.compile(r"\b(?:19\d{2}|20\d{2})\b")


def load_vocab_stats(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_vocab_stats(
    vocab_stats: dict[str, Any],
    *,
    probes: tuple[str, ...] = DEFAULT_PROBES,
) -> dict[str, Any]:
    summary: dict[str, Any] = dict(vocab_stats.get("summary", {}))
    selected_terms: list[dict[str, Any]] = list(vocab_stats.get("selected_terms", []))
    term_strings: list[str] = [str(entry.get("term", "")).strip() for entry in selected_terms]
    term_set: set[str] = {term for term in term_strings if term}

    unigrams: list[str] = [term for term in term_strings if term and " " not in term]
    phrases: list[str] = [term for term in term_strings if " " in term]
    pos_counts: Counter[str] = Counter(
        str(entry.get("pos_tag") or "UNK").upper() for entry in selected_terms
    )

    short_alpha_unigrams: list[str] = [
        term for term in unigrams if term.isalpha() and len(term) <= 2
    ]
    abbreviation_heavy_phrases: list[str] = []
    trailing_function_word_phrases: list[str] = []
    about_numeric_phrases: list[str] = []
    year_terms: list[str] = []
    single_char_token_phrases: list[str] = []

    term: str
    for term in phrases:
        parts: list[str] = [part for part in term.split() if part]
        if parts and parts[-1] in TRAILING_FUNCTION_WORDS:
            trailing_function_word_phrases.append(term)
        if parts and parts[0] == "about" and any(part.isdigit() for part in parts[1:]):
            about_numeric_phrases.append(term)
        if YEAR_PATTERN.search(term) is not None:
            year_terms.append(term)
        if any(len(part) == 1 and part.isalpha() for part in parts):
            single_char_token_phrases.append(term)
        if sum(1 for part in parts if len(part) == 1 and part.isalpha()) >= 2:
            abbreviation_heavy_phrases.append(term)

    for term in unigrams:
        if YEAR_PATTERN.search(term) is not None:
            year_terms.append(term)

    report: dict[str, Any] = {
        "doc_count": summary.get("doc_count"),
        "selected_terms": len(selected_terms),
        "candidate_terms": summary.get("candidate_terms"),
        "unigrams": len(unigrams),
        "phrases": len(phrases),
        "pos_counts": {
            key: int(value) for key, value in sorted(pos_counts.items(), key=lambda item: item[0])
        },
        "strict_post_selection_cleanup": dict(
            summary.get("strict_post_selection_cleanup", {})
        ),
        "short_alpha_unigrams": {
            "count": len(short_alpha_unigrams),
            "sample": short_alpha_unigrams[:50],
        },
        "abbreviation_heavy_phrases": {
            "count": len(abbreviation_heavy_phrases),
            "sample": abbreviation_heavy_phrases[:50],
        },
        "trailing_function_word_phrases": {
            "count": len(trailing_function_word_phrases),
            "sample": trailing_function_word_phrases[:50],
        },
        "about_numeric_phrases": {
            "count": len(about_numeric_phrases),
            "sample": about_numeric_phrases[:50],
        },
        "year_terms": {
            "count": len(year_terms),
            "sample": year_terms[:50],
        },
        "single_char_token_phrases": {
            "count": len(single_char_token_phrases),
            "sample": single_char_token_phrases[:50],
        },
        "probe_membership": {
            probe: bool(probe in term_set) for probe in probes
        },
        "top_terms": term_strings[:30],
        "tail_terms": term_strings[-30:],
    }
    return report
