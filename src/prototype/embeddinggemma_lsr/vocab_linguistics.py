import re
from collections import Counter
from typing import Any

from src.prototype.embeddinggemma_lsr.vocab_filtering import (
    SOURCE_ENTITY,
    SOURCE_NOUN_CHUNK,
)
from src.prototype.embeddinggemma_lsr.vocab_selection import SOURCE_TOKEN

NUMERIC_DIGIT_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"^[0-9]+$")
NUMERIC_ORDINAL_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"^[0-9]+(?:st|nd|rd|th)$",
    re.IGNORECASE,
)
NUMERIC_ROMAN_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"^[ivxlcdm]+$",
    re.IGNORECASE,
)
ALPHA_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"^[a-z]+$")
NUMERIC_WORD_TOKENS: frozenset[str] = frozenset(
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
IRREGULAR_NOUN_SINGULAR_MAP: dict[str, str] = {
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

_WORDNET_LEMMATIZER_CACHE: Any | None = None
_WORDNET_LEMMATIZER_AVAILABLE_CACHE: bool | None = None


def _missing_nltk_error(*, include_wordnet: bool, exc: Exception) -> RuntimeError:
    _ = exc
    message_lines: list[str] = [
        "NLTK POS resources are missing. Run:",
        "python - <<'PY'",
        "import nltk",
        "nltk.download('averaged_perceptron_tagger', quiet=True)",
        "nltk.download('averaged_perceptron_tagger_eng', quiet=True)",
        "nltk.download('universal_tagset', quiet=True)",
    ]
    if bool(include_wordnet):
        message_lines.extend(
            [
                "nltk.download('wordnet', quiet=True)",
                "nltk.download('omw-1.4', quiet=True)",
            ]
        )
    message_lines.extend(["PY"])
    return RuntimeError("\n".join(message_lines))


def infer_phrase_head_universal_pos(tags: list[tuple[str, str]]) -> str:
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


def tag_terms_universal_pos(
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
                raise _missing_nltk_error(include_wordnet=False, exc=exc)
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
                raise _missing_nltk_error(include_wordnet=False, exc=exc)
            phrase: str
            phrase_tags: list[tuple[str, str]]
            for phrase, phrase_tags in zip(phrase_batch, tagged_phrases):
                pos_by_term[phrase] = infer_phrase_head_universal_pos(
                    [(str(tok), str(pos).upper()) for tok, pos in phrase_tags]
                )

    return pos_by_term


def apply_pos_gate_to_terms(
    *,
    terms: list[str],
    allowed_tags: set[str],
    batch_size: int,
) -> tuple[list[str], Counter[str], Counter[str], dict[str, str]]:
    if not terms:
        return [], Counter(), Counter(), {}
    pos_by_term: dict[str, str] = tag_terms_universal_pos(
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
    if normalized in NUMERIC_WORD_TOKENS:
        return True
    if NUMERIC_DIGIT_TOKEN_PATTERN.fullmatch(normalized) is not None:
        return True
    if NUMERIC_ORDINAL_TOKEN_PATTERN.fullmatch(normalized) is not None:
        return True
    if (
        NUMERIC_ROMAN_TOKEN_PATTERN.fullmatch(normalized) is not None
        and len(normalized) <= 6
    ):
        return True
    return False


def numeric_term_quality_reason(
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
    if value in IRREGULAR_NOUN_SINGULAR_MAP:
        return True
    if ALPHA_TOKEN_PATTERN.fullmatch(value) is None:
        return False
    if len(value) <= 3:
        return False
    if value.endswith(("ss", "us", "is")):
        return False
    return value.endswith("s")


def get_wordnet_lemmatizer() -> Any | None:
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
    irregular: str | None = IRREGULAR_NOUN_SINGULAR_MAP.get(value)
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
    lemmatizer: Any | None = get_wordnet_lemmatizer()
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
    if ALPHA_TOKEN_PATTERN.fullmatch(normalized) is None:
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


def normalize_noun_forms_with_hybrid_agreement(
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
            "wordnet_available": bool(get_wordnet_lemmatizer() is not None),
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
        sources_for_term: set[str] = set(term_sources.get(term, {SOURCE_TOKEN}))
        has_non_entity_support: bool = bool(
            {SOURCE_TOKEN, SOURCE_NOUN_CHUNK}.intersection(sources_for_term)
        )
        if (
            bool(skip_entity_backed)
            and SOURCE_ENTITY in sources_for_term
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
                piece.lower() in IRREGULAR_NOUN_SINGULAR_MAP for piece in pieces
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
                raise _missing_nltk_error(include_wordnet=True, exc=exc)
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
                raise _missing_nltk_error(include_wordnet=True, exc=exc)
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
        "wordnet_available": bool(get_wordnet_lemmatizer() is not None),
        "reason_counts": {
            key: int(value) for key, value in sorted(reason_counts.items())
        },
    }
    return normalized_map, stats


__all__: list[str] = [
    "apply_pos_gate_to_terms",
    "get_wordnet_lemmatizer",
    "infer_phrase_head_universal_pos",
    "normalize_noun_forms_with_hybrid_agreement",
    "numeric_term_quality_reason",
    "tag_terms_universal_pos",
]
