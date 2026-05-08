from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from src.utils.normalize import normalize_optional_str

_SENTENCE_BOUNDARY_PATTERN: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+")


def _split_sentences_with_nltk(text: str) -> list[str] | None:
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
    except Exception:
        return None

    try:
        return [str(sentence) for sentence in sent_tokenize(text)]
    except LookupError:
        for resource_name in ("punkt", "punkt_tab"):
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                continue
        try:
            return [str(sentence) for sentence in sent_tokenize(text)]
        except LookupError:
            return None


def clean_patent_passage_text(value: Any) -> str:
    normalized: str | None = normalize_optional_str(value)
    if normalized is None:
        return ""
    return " ".join(str(normalized).replace("\n", " ").split())


def split_into_sentence_chunks(
    text: str,
    *,
    max_words: int = 300,
) -> list[str]:
    cleaned_text: str = clean_patent_passage_text(text)
    if not cleaned_text:
        return []

    max_words = max(1, int(max_words))
    raw_sentences: list[str] | None = _split_sentences_with_nltk(cleaned_text)
    sentence_source: list[str] = (
        raw_sentences
        if raw_sentences is not None
        else _SENTENCE_BOUNDARY_PATTERN.split(cleaned_text)
    )
    sentences: list[str] = []
    for sentence in sentence_source:
        normalized_sentence: str = " ".join(sentence.split())
        if normalized_sentence:
            sentences.append(normalized_sentence)

    if not sentences:
        return []

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len: int = 0
    sentence: str
    for sentence in sentences:
        word_count: int = len(sentence.split())
        if word_count <= 0:
            continue
        if current_chunk and current_len + word_count > max_words:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_len = word_count
            continue
        current_chunk.append(sentence)
        current_len += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    return chunks


def build_title_prefixed_claim_passage(
    *,
    title: str,
    claim_chunk: str,
    max_title_prefixed_words: int = 100,
) -> str:
    chunk_words: list[str] = clean_patent_passage_text(claim_chunk).split()
    if not chunk_words:
        return ""

    title_words: list[str] = clean_patent_passage_text(title).split()
    if not title_words:
        return " ".join(chunk_words)

    max_title_prefixed_words = max(1, int(max_title_prefixed_words))
    merged_words: list[str] = title_words + chunk_words
    if len(merged_words) <= max_title_prefixed_words:
        return " ".join(merged_words)

    space_for_title: int = max_title_prefixed_words - len(chunk_words)
    if space_for_title > 0:
        return " ".join(title_words[:space_for_title] + chunk_words)
    return " ".join(chunk_words)


def build_patent_claim_passages(
    row: Mapping[str, Any],
    *,
    doc_id_key: str = "doc_id",
    title_key: str = "title",
    claims_key: str = "claims",
    group_id_key: str | None = "parent_doc_id",
    max_claim_chunk_words: int = 300,
    max_title_prefixed_words: int = 100,
) -> list[dict[str, Any]]:
    doc_id: str = clean_patent_passage_text(row.get(doc_id_key))
    if not doc_id:
        return []

    title: str = clean_patent_passage_text(row.get(title_key))
    claims: str = clean_patent_passage_text(row.get(claims_key))
    if not claims:
        return []

    group_id: str = doc_id
    if group_id_key is not None:
        resolved_group_id: str = clean_patent_passage_text(row.get(group_id_key))
        if resolved_group_id:
            group_id = resolved_group_id

    passages: list[dict[str, Any]] = []
    for chunk_idx, claim_chunk in enumerate(
        split_into_sentence_chunks(claims, max_words=max_claim_chunk_words)
    ):
        passage_text: str = build_title_prefixed_claim_passage(
            title=title,
            claim_chunk=claim_chunk,
            max_title_prefixed_words=max_title_prefixed_words,
        )
        if not passage_text:
            continue
        passages.append(
            {
                "passage_id": f"{doc_id}&&&claim&&&{chunk_idx}",
                "parent_doc_id": group_id,
                "source_doc_id": doc_id,
                "chunk_type": "claim",
                "chunk_idx": int(chunk_idx),
                "text": passage_text,
            }
        )
    return passages


def iter_patent_claim_passages(
    rows: Iterable[Mapping[str, Any]],
    *,
    doc_id_key: str = "doc_id",
    title_key: str = "title",
    claims_key: str = "claims",
    group_id_key: str | None = "parent_doc_id",
    max_claim_chunk_words: int = 300,
    max_title_prefixed_words: int = 100,
) -> Iterable[dict[str, Any]]:
    row: Mapping[str, Any]
    for row in rows:
        yield from build_patent_claim_passages(
            row,
            doc_id_key=doc_id_key,
            title_key=title_key,
            claims_key=claims_key,
            group_id_key=group_id_key,
            max_claim_chunk_words=max_claim_chunk_words,
            max_title_prefixed_words=max_title_prefixed_words,
        )


__all__ = [
    "build_patent_claim_passages",
    "build_title_prefixed_claim_passage",
    "clean_patent_passage_text",
    "iter_patent_claim_passages",
    "split_into_sentence_chunks",
]
