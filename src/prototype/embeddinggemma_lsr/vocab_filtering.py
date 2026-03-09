import re

SOURCE_ENTITY: str = "entity"
SOURCE_NOUN_CHUNK: str = "noun_chunk"

PURE_NUMERIC_TERM_PATTERN: re.Pattern[str] = re.compile(
    r"[0-9]+(?:[.,:/-][0-9]+)*"
)
C1_CONTROL_PATTERN: re.Pattern[str] = re.compile(r"[\x80-\x9f]")
MOJIBAKE_UTF8_LATIN1_PATTERN: re.Pattern[str] = re.compile(
    r"[\u00c2\u00c3\u00e2][\x80-\xbf]"
)
CANONICAL_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9]+")
URL_SCHEME_PATTERN: re.Pattern[str] = re.compile(r"\bhttps?\b", re.IGNORECASE)
URL_DOT_SUFFIX_PATTERN: re.Pattern[str] = re.compile(
    r"\.(?:com|org|net|gov|edu|io|co|ai|app|info|biz|uk|de|fr|jp|cn|ru|br|it|es)\b",
    re.IGNORECASE,
)
URL_TOKEN_HINTS: frozenset[str] = frozenset(
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
TEMPLATE_PHRASE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:web\s+site|zip\s+code|job\s+posting|fact\s+sheet|template\s+message|"
    r"dictionary\s+definition\s+resource|video\s+clip|medication\s+guide)\b",
    re.IGNORECASE,
)
NUMERIC_ORDINAL_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"^[0-9]+(?:st|nd|rd|th)$",
    re.IGNORECASE,
)
LETTER_NUMBER_PHRASE_PATTERN: re.Pattern[str] = re.compile(
    r"^[a-z]\s+[0-9]{1,4}[a-z]?$"
)
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
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
    }
)
STRICT_ABOUT_PHRASE_HEAD: str = "about"
STRICT_LEADING_FUNCTION_WORDS: frozenset[str] = frozenset(
    {"the", "a", "an", "of", "and", "to", "for", "in", "on", "with", "from", "by"}
)


def normalize_phrase_for_filter(value: str) -> str:
    parts: list[str] = [
        part.lower() for part in CANONICAL_TOKEN_PATTERN.findall(str(value)) if part
    ]
    return " ".join(parts)


def is_url_like_term(term: str) -> bool:
    normalized: str = str(term).strip()
    if not normalized:
        return False
    lowered: str = normalized.lower()
    if URL_SCHEME_PATTERN.search(lowered) is not None:
        return True
    if URL_DOT_SUFFIX_PATTERN.search(lowered) is not None:
        return True
    if "www." in lowered:
        return True
    tokens: list[str] = CANONICAL_TOKEN_PATTERN.findall(lowered)
    if not tokens:
        return False
    token_set: set[str] = set(tokens)
    if "www" in token_set or "http" in token_set or "https" in token_set:
        return True
    if any(token in URL_TOKEN_HINTS for token in tokens):
        if "." in lowered or "/" in lowered:
            return True
        if len(tokens) <= 2:
            return True
    return False


def is_template_like_term(term: str) -> bool:
    return TEMPLATE_PHRASE_PATTERN.search(str(term).strip()) is not None


def is_stopword_term(
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


def noise_term_reason(
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
        if C1_CONTROL_PATTERN.search(normalized) is not None:
            return "mojibake_c1_control"
        if MOJIBAKE_UTF8_LATIN1_PATTERN.search(normalized) is not None:
            return "mojibake_utf8_latin1"
    if bool(drop_url_like) and is_url_like_term(normalized):
        return "url_like"
    if bool(drop_template_like) and is_template_like_term(normalized):
        return "template_like"

    if bool(drop_single_char) and len(compact) == 1:
        if compact.isalpha():
            return "single_char_alpha"
        if not compact.isalnum():
            return "single_char_non_alnum"

    if bool(drop_pure_numeric) and PURE_NUMERIC_TERM_PATTERN.fullmatch(compact):
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


def canonicalize_term_for_selection(
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
        part.lower() for part in CANONICAL_TOKEN_PATTERN.findall(text) if part
    ]
    if not pieces:
        return None
    if bool(strip_leading_determiners):
        while pieces and pieces[0] in leading_determiners:
            pieces = pieces[1:]
    if not pieces:
        return None
    return " ".join(pieces)


def is_function_leading_phrase(
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
    if bool(require_noun_chunk_source) and SOURCE_NOUN_CHUNK not in sources:
        return False
    if bool(keep_entity_backed) and SOURCE_ENTITY in sources:
        return False
    return parts[0] in function_leading_words


def structured_artifact_reason(
    *,
    term: str,
    filter_html_entity_artifacts: bool,
    html_entity_blacklist: set[str],
    filter_pronoun_led_phrases: bool,
    pronoun_leading_words: set[str],
    filter_letter_number_phrases: bool,
    letter_number_phrase_whitelist: set[str],
) -> str | None:
    normalized: str = normalize_phrase_for_filter(term)
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
        and LETTER_NUMBER_PHRASE_PATTERN.fullmatch(normalized) is not None
        and normalized not in letter_number_phrase_whitelist
    ):
        return "letter_number_phrase"
    return None


def contraction_artifact_reason(*, term: str) -> str | None:
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


def strict_post_selection_cleanup_reason(
    *,
    term: str,
    drop_short_alpha_unigrams: bool,
    short_alpha_unigram_max_len: int,
    short_alpha_unigram_whitelist: set[str],
    drop_about_numeric_phrases: bool,
    drop_leading_numeric_function_phrases: bool,
    drop_trailing_function_word_phrases: bool,
    trailing_function_words: set[str],
    drop_abbreviation_heavy_phrases: bool,
    abbreviation_phrase_whitelist: set[str],
    drop_artifact_substrings: bool,
    artifact_substrings: set[str],
) -> str | None:
    normalized: str = normalize_phrase_for_filter(term)
    if not normalized:
        return "empty"
    parts: list[str] = [part for part in normalized.split() if part]
    if not parts:
        return "empty"

    if bool(drop_artifact_substrings):
        snippet: str
        for snippet in artifact_substrings:
            if snippet and snippet in normalized:
                return "artifact_substring"

    if bool(drop_short_alpha_unigrams) and len(parts) == 1:
        token: str = parts[0]
        if (
            token.isalpha()
            and len(token) <= int(short_alpha_unigram_max_len)
            and token not in short_alpha_unigram_whitelist
        ):
            return "short_alpha_unigram"

    if (
        bool(drop_about_numeric_phrases)
        and len(parts) >= 2
        and parts[0] == STRICT_ABOUT_PHRASE_HEAD
        and any(
            part.isdigit()
            or bool(NUMERIC_ORDINAL_TOKEN_PATTERN.fullmatch(part))
            or part in NUMERIC_WORD_TOKENS
            for part in parts[1:]
        )
    ):
        return "about_numeric_phrase"

    if (
        bool(drop_leading_numeric_function_phrases)
        and len(parts) >= 3
        and (
            parts[0].isdigit()
            or bool(NUMERIC_ORDINAL_TOKEN_PATTERN.fullmatch(parts[0]))
            or parts[0] in NUMERIC_WORD_TOKENS
        )
        and parts[1] in STRICT_LEADING_FUNCTION_WORDS
    ):
        return "leading_numeric_function_phrase"

    if (
        bool(drop_trailing_function_word_phrases)
        and len(parts) >= 2
        and parts[-1] in trailing_function_words
    ):
        return "trailing_function_word_phrase"

    if (
        bool(drop_abbreviation_heavy_phrases)
        and len(parts) >= 2
        and normalized not in abbreviation_phrase_whitelist
    ):
        single_alpha_tokens: int = sum(
            1 for part in parts if len(part) == 1 and part.isalpha()
        )
        if single_alpha_tokens >= 2:
            return "abbreviation_heavy_phrase"

    return None
