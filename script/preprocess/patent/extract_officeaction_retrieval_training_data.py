"""Extract structured retrieval training data from office-action JSONL.

This pipeline is designed for retrieval supervision quality rather than for
reconstructing a legacy Hugging Face dataset. It uses:

1. office-action sections as label evidence
2. the examined patent document as the query source
3. claim-group / claim-specific rationale units as the extraction granularity

Output:

- `examples.jsonl`
- `metadata.json`
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.compute as pc
import pyarrow.dataset as ds


CLAIM_BLOCK_START_RE: re.Pattern[str] = re.compile(
    r"(?im)^\s*Claim(?:s|\(s\))?\s+[^\n]{0,500}?\brejected under\b[^\n]*"
)
CLAIM_HEADER_RE: re.Pattern[str] = re.compile(
    r"(?im)^\s*Claim(?:s|\(s\))?\s+(.+?)\s+(?:is/are|is|are)\s+rejected under\s+35\s+U\.S\.C\.\s*§?\s*(102|103)\b"
)
CLAIM_RATIONALE_START_RE: re.Pattern[str] = re.compile(
    r"(?im)^\s*(?:Regarding|As to)\s+claim(?:s)?\s+([^\n,;:.]+)"
)
CLAIMS_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?ms)(?:^|\n)\s*(\d+)\.\s+(.*?)(?=(?:\n\s*\d+\.\s)|\Z)"),
    re.compile(r"(?ms)(?:^|\n)\s*(\d+)\)\s+(.*?)(?=(?:\n\s*\d+\)\s)|\Z)"),
    re.compile(r"(?ms)(?:^|\n)\s*Claim\s+(\d+)\s*[:.]\s*(.*?)(?=(?:\n\s*Claim\s+\d+\s*[:.])|\Z)", re.I),
)
SEQUENTIAL_CLAIM_SPLIT_RE: re.Pattern[str] = re.compile(
    r"\n\s*\.?\s*(?=(?:An?|The)\b)"
)
SENTENCE_SPLIT_RE: re.Pattern[str] = re.compile(
    r"(?<=[.!?;])\s+(?=(?:\[[0-9]{4}\]|[A-Z]))"
)
PARA_REF_RE: re.Pattern[str] = re.compile(r"\bPara\.?\s*\d+(?:-\d+)?|\[\d{1,4}\]", re.I)
FIG_REF_RE: re.Pattern[str] = re.compile(r"\b(?:Figure|Fig\.?)\s*\d+[A-Za-z]?", re.I)
COL_LINE_REF_RE: re.Pattern[str] = re.compile(
    r"\bCol\.?\s*\d+(?:\s*,\s*|\s+)(?:ln\.?|line|lines)\s*\d+(?:-\d+)?",
    re.I,
)
ABSTRACT_REF_RE: re.Pattern[str] = re.compile(r"\bAbstract\b", re.I)
BOILERPLATE_PREFIX_RE: re.Pattern[str] = re.compile(
    r"(?is)^\s*Claim Rejections\s*-\s*35\s*USC\s*§?\s*(?:102|103)\s*"
    r"(?:.*?forms the basis for .*?Office action:\s*)?"
)
HEREINAFTER_ALIAS_RE: re.Pattern[str] = re.compile(r'hereinafter\s+[“"]([^”"]+)[”"]', re.I)
TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")


@dataclass(slots=True)
class CandidateReference:
    index: int
    application_id: str
    publication_id: str
    application_field: str | None
    publication_field: str | None
    raw_citation: str
    markers: list[str]
    alias: str | None = None
    role: str = "supporting"


@dataclass(slots=True)
class CandidateUnit:
    officeaction_line: int
    examined_app_id: str
    statute: str
    section: str
    block_index: int
    unit_index: int
    claim_ids: list[str]
    cpc: list[str]
    header_line: str
    block_text: str
    rationale_text: str
    references: list[CandidateReference]


@dataclass(slots=True)
class PatentRecord:
    doc_id: str
    application_id: str
    title: str
    abstract: str
    claims: str
    description: str


@dataclass(slots=True)
class BuildStats:
    rows_scanned: int = 0
    sections_with_text: int = 0
    claim_blocks: int = 0
    rationale_units: int = 0
    examples_written: int = 0
    examples_dropped: int = 0
    examples_missing_patent_record: int = 0
    examples_missing_claim_text: int = 0
    examples_without_resolved_positive_doc: int = 0
    examples_with_evidence_refs: int = 0
    gold_examples: int = 0
    silver_examples: int = 0
    bronze_examples: int = 0
    drop_examples: int = 0


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split())


def normalize_identifier(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").strip().upper())


def normalize_match_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def split_rejection_text(text: str) -> list[str]:
    raw_text = (text or "").strip()
    if not raw_text:
        return []
    matches = list(CLAIM_BLOCK_START_RE.finditer(raw_text))
    if not matches:
        return [raw_text]
    blocks: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        segment = raw_text[start:end].strip()
        if segment:
            blocks.append(segment)
    return blocks


def expand_claim_expression(text: str) -> list[str]:
    cleaned = (
        (text or "")
        .replace("–", "-")
        .replace("—", "-")
        .replace("to", "-")
        .replace("and", ",")
        .replace("or", ",")
    )
    tokens = re.findall(r"\d+\s*-\s*\d+|\d+", cleaned)
    claim_ids: list[str] = []
    for token in tokens:
        token = token.replace(" ", "")
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start <= end:
                claim_ids.extend(str(value) for value in range(start, end + 1))
            else:
                claim_ids.extend(str(value) for value in range(start, end - 1, -1))
        else:
            claim_ids.append(str(int(token)))
    return dedupe_preserve_order(claim_ids)


def extract_header_line(block_text: str) -> str:
    match = CLAIM_BLOCK_START_RE.search(block_text or "")
    if not match:
        first_line = (block_text or "").strip().splitlines()
        return first_line[0].strip() if first_line else ""
    line = match.group(0).strip()
    return normalize_whitespace(line)


def parse_claim_block_header(block_text: str) -> tuple[list[str], str | None, str]:
    header_line = extract_header_line(block_text)
    match = CLAIM_HEADER_RE.search(header_line)
    if not match:
        return [], None, header_line
    claim_ids = expand_claim_expression(match.group(1))
    statute = match.group(2)
    return claim_ids, statute, header_line


def split_claim_rationale_units(block_text: str, fallback_claim_ids: list[str]) -> list[tuple[list[str], str]]:
    matches = list(CLAIM_RATIONALE_START_RE.finditer(block_text or ""))
    if not matches:
        return [(fallback_claim_ids, block_text.strip())]
    units: list[tuple[list[str], str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(block_text)
        unit_text = block_text[start:end].strip()
        claim_ids = expand_claim_expression(match.group(1))
        units.append((claim_ids or fallback_claim_ids, unit_text))
    return units


def parse_claims_text(claims_text: str) -> dict[str, str]:
    text = (claims_text or "").strip()
    if not text:
        return {}
    best: dict[str, str] = {}
    for pattern in CLAIMS_TEXT_PATTERNS:
        found = {
            str(match.group(1)): normalize_whitespace(match.group(0))
            for match in pattern.finditer(text)
            if normalize_whitespace(match.group(0))
        }
        if len(found) > len(best):
            best = found
    if best:
        return best

    chunks = [
        normalize_whitespace(chunk)
        for chunk in SEQUENTIAL_CLAIM_SPLIT_RE.split(text)
        if normalize_whitespace(chunk)
    ]
    if len(chunks) <= 1:
        return {}
    return {
        str(index): f"{index}. {chunk}" if not chunk.startswith(f"{index}.") else chunk
        for index, chunk in enumerate(chunks, start=1)
    }


def split_description_paragraphs(text: str) -> list[str]:
    raw_text = (text or "").strip()
    if not raw_text:
        return []
    chunks: list[str]
    if "[0001]" in raw_text:
        chunks = re.split(r"(?=\[\d{4}\])", raw_text)
    else:
        chunks = re.split(r"\n\s*\n+", raw_text)
    paragraphs = [
        normalize_whitespace(chunk)
        for chunk in chunks
        if normalize_whitespace(chunk)
    ]
    normalized = [paragraph for paragraph in paragraphs if len(paragraph) >= 40]
    expanded: list[str] = []
    for paragraph in normalized:
        expanded.extend(split_long_description_chunk(paragraph))
    return expanded


def split_long_description_chunk(text: str, *, max_chars: int = 900) -> list[str]:
    paragraph = normalize_whitespace(text)
    if not paragraph:
        return []
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = [
        normalize_whitespace(chunk)
        for chunk in SENTENCE_SPLIT_RE.split(paragraph)
        if normalize_whitespace(chunk)
    ]
    if len(sentences) <= 1:
        return chunk_text_by_words(paragraph, max_chars=max_chars)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence) + (1 if current else 0)
        if current and current_len + sentence_len > max_chars:
            chunk = normalize_whitespace(" ".join(current))
            if chunk:
                chunks.append(chunk)
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += sentence_len
    if current:
        chunk = normalize_whitespace(" ".join(current))
        if chunk:
            chunks.append(chunk)
    return chunks or chunk_text_by_words(paragraph, max_chars=max_chars)


def chunk_text_by_words(text: str, *, max_chars: int = 900) -> list[str]:
    words = normalize_whitespace(text).split()
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        word_len = len(word) + (1 if current else 0)
        if current and current_len + word_len > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += word_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def tokenize_for_overlap(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "")}


def select_description_snippets(
    *,
    abstract: str,
    description: str,
    claim_texts: Sequence[str],
    rationale_text: str,
    max_snippets: int,
) -> list[str]:
    paragraphs = split_description_paragraphs(description)
    if not paragraphs:
        return [normalize_whitespace(abstract)] if normalize_whitespace(abstract) else []
    claim_tokens = tokenize_for_overlap(" ".join(claim_texts))
    rationale_tokens = tokenize_for_overlap(rationale_text)
    scored: list[tuple[int, int, str]] = []
    for index, paragraph in enumerate(paragraphs):
        tokens = tokenize_for_overlap(paragraph)
        score = (2 * len(tokens & claim_tokens)) + len(tokens & rationale_tokens)
        if score <= 0:
            continue
        scored.append((score, -index, paragraph))
    scored.sort(reverse=True)
    snippets = [paragraph for _, _, paragraph in scored[:max_snippets]]
    if snippets:
        return snippets
    abstract_text = normalize_whitespace(abstract)
    return [abstract_text] if abstract_text else paragraphs[:max_snippets]


def compose_query_text(
    *,
    title: str,
    claim_texts: Sequence[str],
    description_snippets: Sequence[str],
) -> str:
    parts: list[str] = []
    if normalize_whitespace(title):
        parts.append(f"Title: {normalize_whitespace(title)}")
    if claim_texts:
        parts.append("Claims:")
        parts.extend(claim_texts)
    if description_snippets:
        parts.append("Description:")
        parts.extend(description_snippets)
    return "\n".join(parts).strip()


def extract_evidence_refs(text: str) -> list[str]:
    refs: list[str] = []
    for pattern in (ABSTRACT_REF_RE, FIG_REF_RE, PARA_REF_RE, COL_LINE_REF_RE):
        refs.extend(match.group(0) for match in pattern.finditer(text or ""))
    return dedupe_preserve_order([normalize_whitespace(ref) for ref in refs if ref.strip()])


def _section_field_candidates(section_suffix: str) -> dict[str, str]:
    return {
        "dedup_application": f"DedupApplicationCheck{section_suffix}",
        "opensearch_application": f"OpenSearchApplicationMatches{section_suffix}",
        "search_application": f"SearchApplicationNumbers{section_suffix}",
        "cited_application": f"CitedApplicationNumbers{section_suffix}",
        "dedup_publication": f"DedupPublicationCheck{section_suffix}",
        "opensearch_publication": f"OpenSearchPublicationMatches{section_suffix}",
        "search_publication": f"SearchPublicationNumbers{section_suffix}",
        "cited_publication": f"CitedPublicationNumbers{section_suffix}",
    }


def build_reference_items(section: dict[str, Any], *, section_suffix: str) -> list[CandidateReference]:
    field_names = _section_field_candidates(section_suffix)
    arrays: dict[str, list[str]] = {}
    max_len = 0
    for field_name in field_names.values():
        raw_values = section.get(field_name)
        if isinstance(raw_values, list):
            arrays[field_name] = [str(value).strip() for value in raw_values]
            max_len = max(max_len, len(arrays[field_name]))
        else:
            arrays[field_name] = []

    items: list[CandidateReference] = []
    for index in range(max_len):
        application_id = ""
        application_field = None
        for logical_name in (
            "dedup_application",
            "opensearch_application",
            "search_application",
            "cited_application",
        ):
            field_name = field_names[logical_name]
            values = arrays.get(field_name, [])
            if index < len(values) and values[index].strip():
                application_id = normalize_identifier(values[index])
                application_field = field_name
                break

        publication_id = ""
        publication_field = None
        for logical_name in (
            "dedup_publication",
            "opensearch_publication",
            "search_publication",
            "cited_publication",
        ):
            field_name = field_names[logical_name]
            values = arrays.get(field_name, [])
            if index < len(values) and values[index].strip():
                publication_id = normalize_identifier(values[index])
                publication_field = field_name
                break

        cited_publication_values = arrays.get(field_names["cited_publication"], [])
        raw_citation = ""
        if index < len(cited_publication_values):
            raw_citation = cited_publication_values[index].strip()
        if not raw_citation:
            raw_citation = publication_id or application_id

        markers: list[str] = []
        for field_name in field_names.values():
            values = arrays.get(field_name, [])
            if index >= len(values):
                continue
            normalized = normalize_match_text(values[index])
            if normalized:
                markers.append(normalized)
        markers = dedupe_preserve_order(markers)

        if not application_id and not publication_id and not markers:
            continue

        items.append(
            CandidateReference(
                index=index,
                application_id=application_id,
                publication_id=publication_id,
                application_field=application_field,
                publication_field=publication_field,
                raw_citation=raw_citation,
                markers=markers,
            )
        )
    return items


def select_block_references(
    section: dict[str, Any],
    block_text: str,
    *,
    section_suffix: str,
) -> list[CandidateReference]:
    items = build_reference_items(section, section_suffix=section_suffix)
    if not items:
        return []
    normalized_block = normalize_match_text(block_text)
    filtered = [
        item
        for item in items
        if item.markers and any(marker in normalized_block for marker in item.markers)
    ]
    return filtered or items


def assign_reference_aliases_and_roles(
    references: Sequence[CandidateReference],
    *,
    statute: str,
    header_line: str,
) -> list[CandidateReference]:
    aliases = HEREINAFTER_ALIAS_RE.findall(header_line or "")
    enriched: list[CandidateReference] = []
    for index, reference in enumerate(references):
        alias = aliases[index].strip() if index < len(aliases) else None
        if len(references) == 1:
            role = "primary"
        elif statute == "102":
            role = "primary" if index == 0 else "supporting"
        elif " in view of " in (header_line or "").lower():
            role = "primary" if index == 0 else "supporting"
        else:
            role = "primary" if index == 0 else "supporting"
        enriched.append(
            CandidateReference(
                index=reference.index,
                application_id=reference.application_id,
                publication_id=reference.publication_id,
                application_field=reference.application_field,
                publication_field=reference.publication_field,
                raw_citation=reference.raw_citation,
                markers=list(reference.markers),
                alias=alias,
                role=role,
            )
        )
    return enriched


def select_unit_references(
    block_references: Sequence[CandidateReference],
    rationale_text: str,
) -> list[CandidateReference]:
    normalized_text = normalize_match_text(rationale_text)
    matched: list[CandidateReference] = []
    for reference in block_references:
        alias = normalize_match_text(reference.alias or "")
        alias_hit = alias and alias in normalized_text
        marker_hit = any(marker in normalized_text for marker in reference.markers)
        if alias_hit or marker_hit:
            matched.append(reference)
    return matched or list(block_references)


def reference_confidence(reference: CandidateReference) -> str:
    if reference.application_field and reference.application_field.startswith("DedupApplicationCheck"):
        return "gold"
    if reference.application_field and reference.application_field.startswith("OpenSearchApplicationMatches"):
        return "silver"
    return "bronze"


def reference_weight(reference: CandidateReference, statute: str) -> float:
    base = 1.0 if statute == "102" else 0.8
    if reference.role == "supporting":
        base -= 0.3
    confidence = reference_confidence(reference)
    if confidence == "silver":
        base -= 0.1
    elif confidence == "bronze":
        base -= 0.3
    return max(0.1, round(base, 2))


def iter_candidate_units(
    officeaction_path: Path,
    *,
    max_rows: int | None = None,
) -> Iterable[CandidateUnit]:
    with officeaction_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if max_rows is not None and line_number > max_rows:
                break
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            app_id = normalize_identifier(str(row.get("patentApplicationNumber", "")))
            cpc = [
                str(value).strip()
                for value in row.get("patentCPCList", [])
                if str(value).strip()
            ] if isinstance(row.get("patentCPCList"), list) else []
            for section_name, section_suffix in (("ClaimRejections102", "102"), ("ClaimRejections103", "103")):
                section = row.get(section_name)
                if not isinstance(section, dict):
                    continue
                section_text = str(section.get("text", "") or "").strip()
                if not section_text:
                    continue
                for block_index, block_text in enumerate(split_rejection_text(section_text)):
                    claim_ids, statute, header_line = parse_claim_block_header(block_text)
                    block_refs = select_block_references(
                        section,
                        block_text,
                        section_suffix=section_suffix,
                    )
                    block_refs = assign_reference_aliases_and_roles(
                        block_refs,
                        statute=statute or section_suffix,
                        header_line=header_line,
                    )
                    for unit_index, (unit_claim_ids, rationale_text) in enumerate(
                        split_claim_rationale_units(block_text, claim_ids)
                    ):
                        references = select_unit_references(block_refs, rationale_text)
                        yield CandidateUnit(
                            officeaction_line=line_number,
                            examined_app_id=app_id,
                            statute=statute or section_suffix,
                            section=section_suffix,
                            block_index=block_index,
                            unit_index=unit_index,
                            claim_ids=unit_claim_ids,
                            cpc=cpc,
                            header_line=header_line,
                            block_text=block_text,
                            rationale_text=rationale_text,
                            references=references,
                        )


def build_patent_record_lookup(
    corpus_paths: Sequence[str],
    target_ids: set[str],
) -> dict[str, PatentRecord]:
    if not target_ids:
        return {}
    dataset = ds.dataset(list(corpus_paths), format="parquet")
    table = dataset.to_table(
        columns=["doc_id", "application_id", "title", "abstract", "claims", "description"],
        filter=(
            pc.field("application_id").isin(sorted(target_ids))
            | pc.field("doc_id").isin(sorted(target_ids))
        ),
    )
    lookup: dict[str, PatentRecord] = {}
    for row in table.to_pylist():
        record = PatentRecord(
            doc_id=normalize_identifier(str(row.get("doc_id", "") or "")),
            application_id=normalize_identifier(str(row.get("application_id", "") or "")),
            title=str(row.get("title", "") or ""),
            abstract=str(row.get("abstract", "") or ""),
            claims=str(row.get("claims", "") or ""),
            description=str(row.get("description", "") or ""),
        )
        if record.doc_id:
            lookup.setdefault(record.doc_id, record)
        if record.application_id:
            lookup.setdefault(record.application_id, record)
    return lookup


def quality_tier(
    *,
    claim_texts: Sequence[str],
    positives: Sequence[dict[str, Any]],
    evidence_refs_present: bool,
) -> str:
    resolved_doc_positives = [positive for positive in positives if positive.get("doc_id")]
    if claim_texts and resolved_doc_positives and evidence_refs_present:
        return "gold"
    if claim_texts and resolved_doc_positives:
        return "silver"
    if positives:
        return "bronze"
    return "drop"


def build_query_id(unit: CandidateUnit) -> str:
    claims_part = "_".join(unit.claim_ids) if unit.claim_ids else "unknown"
    return (
        f"{unit.examined_app_id}__{unit.statute}__claims_{claims_part}"
        f"__block_{unit.block_index}__unit_{unit.unit_index}"
    )


def materialize_examples(
    candidate_units: Sequence[CandidateUnit],
    patent_lookup: dict[str, PatentRecord],
    *,
    max_description_snippets: int,
    require_claim_text: bool,
    min_quality_tier: str,
) -> tuple[list[dict[str, Any]], BuildStats, Counter[str]]:
    stats = BuildStats()
    tier_counter: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    claim_map_cache: dict[str, dict[str, str]] = {}
    tier_rank = {"drop": 0, "bronze": 1, "silver": 2, "gold": 3}
    min_rank = tier_rank[min_quality_tier]

    for unit in candidate_units:
        stats.rationale_units += 1
        patent = patent_lookup.get(unit.examined_app_id)
        if patent is None:
            stats.examples_missing_patent_record += 1
            stats.examples_dropped += 1
            stats.drop_examples += 1
            tier_counter["drop"] += 1
            continue

        claim_map = claim_map_cache.setdefault(unit.examined_app_id, parse_claims_text(patent.claims))
        claim_texts = [claim_map[claim_id] for claim_id in unit.claim_ids if claim_id in claim_map]
        if require_claim_text and not claim_texts:
            stats.examples_missing_claim_text += 1
            stats.examples_dropped += 1
            stats.drop_examples += 1
            tier_counter["drop"] += 1
            continue

        description_snippets = select_description_snippets(
            abstract=patent.abstract,
            description=patent.description,
            claim_texts=claim_texts,
            rationale_text=unit.rationale_text,
            max_snippets=max_description_snippets,
        )
        query_text = compose_query_text(
            title=patent.title,
            claim_texts=claim_texts,
            description_snippets=description_snippets,
        )

        evidence_refs = extract_evidence_refs(unit.rationale_text)
        positives: list[dict[str, Any]] = []
        for reference in unit.references:
            positives.append(
                {
                    "doc_id": reference.application_id or "",
                    "role": reference.role,
                    "weight": reference_weight(reference, unit.statute),
                    "confidence": reference_confidence(reference),
                    "source_field": reference.application_field or reference.publication_field,
                    "raw_citation": reference.raw_citation,
                    "publication_ids": [reference.publication_id] if reference.publication_id else [],
                    "evidence_text": normalize_whitespace(unit.rationale_text),
                    "evidence_refs": evidence_refs,
                    "alias": reference.alias,
                }
            )

        tier = quality_tier(
            claim_texts=claim_texts,
            positives=positives,
            evidence_refs_present=bool(evidence_refs),
        )
        tier_counter[tier] += 1
        if tier == "gold":
            stats.gold_examples += 1
        elif tier == "silver":
            stats.silver_examples += 1
        elif tier == "bronze":
            stats.bronze_examples += 1
        else:
            stats.drop_examples += 1
        if tier_rank[tier] < min_rank:
            stats.examples_dropped += 1
            continue

        if not any(positive.get("doc_id") for positive in positives):
            stats.examples_without_resolved_positive_doc += 1

        example = {
            "query_id": build_query_id(unit),
            "examined_app_id": unit.examined_app_id,
            "officeaction_line": unit.officeaction_line,
            "statute": unit.statute,
            "claim_ids": unit.claim_ids,
            "query_title": normalize_whitespace(patent.title),
            "query_claim_texts": claim_texts,
            "query_description_snippets": description_snippets,
            "query_text": query_text,
            "positives": positives,
            "quality_tier": tier,
            "source_header_line": unit.header_line,
            "source_rationale_text": normalize_whitespace(unit.rationale_text),
            "cpc": unit.cpc,
        }
        examples.append(example)
        stats.examples_written += 1
        if evidence_refs:
            stats.examples_with_evidence_refs += 1

    return examples, stats, tier_counter


def write_examples_jsonl(examples: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--officeaction-path",
        type=Path,
        default=Path("officeaction_102_103_20250105-20250330_cpc.jsonl"),
        help="Path to office-action JSONL.",
    )
    parser.add_argument(
        "--corpus-glob",
        default=".cache/hf/patent-us-corpus/patent_us_docs_slice*.parquet",
        help="Glob for local patent corpus parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/patent/officeaction_retrieval_training"),
        help="Output directory for examples and metadata.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max office-action rows to scan.",
    )
    parser.add_argument(
        "--max-description-snippets",
        type=int,
        default=2,
        help="Max number of description snippets to include in the query text.",
    )
    parser.add_argument(
        "--require-claim-text",
        action="store_true",
        help="Drop examples whose rejected claims cannot be resolved from the patent corpus.",
    )
    parser.add_argument(
        "--min-quality-tier",
        choices=["drop", "bronze", "silver", "gold"],
        default="silver",
        help="Minimum quality tier to keep in the final output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    officeaction_path = Path(args.officeaction_path)
    if not officeaction_path.exists():
        raise FileNotFoundError(f"Missing office-action file: {officeaction_path}")

    corpus_paths = sorted(glob(str(args.corpus_glob)))
    if not corpus_paths:
        raise FileNotFoundError(f"No parquet files matched --corpus-glob={args.corpus_glob!r}")

    candidate_units: list[CandidateUnit] = []
    row_count = 0
    section_keys: set[tuple[int, str]] = set()
    block_keys: set[tuple[int, str, int]] = set()
    for unit in iter_candidate_units(officeaction_path, max_rows=args.max_rows):
        candidate_units.append(unit)
        row_count = max(row_count, unit.officeaction_line)
        section_keys.add((unit.officeaction_line, unit.section))
        block_keys.add((unit.officeaction_line, unit.section, unit.block_index))
    target_ids = {unit.examined_app_id for unit in candidate_units if unit.examined_app_id}
    patent_lookup = build_patent_record_lookup(corpus_paths, target_ids)
    examples, materialized_stats, tier_counter = materialize_examples(
        candidate_units,
        patent_lookup,
        max_description_snippets=int(args.max_description_snippets),
        require_claim_text=bool(args.require_claim_text),
        min_quality_tier=str(args.min_quality_tier),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "examples.jsonl"
    metadata_path = output_dir / "metadata.json"
    write_examples_jsonl(examples, examples_path)

    metadata = {
        "officeaction_path": officeaction_path.as_posix(),
        "corpus_glob": str(args.corpus_glob),
        "max_rows": args.max_rows,
        "max_description_snippets": int(args.max_description_snippets),
        "require_claim_text": bool(args.require_claim_text),
        "min_quality_tier": str(args.min_quality_tier),
        "rows_scanned": row_count,
        "sections_with_text": len(section_keys),
        "claim_blocks": len(block_keys),
        "candidate_units": len(candidate_units),
        "loaded_patent_records": len(patent_lookup),
        "examples_path": examples_path.as_posix(),
        "quality_tiers": dict(tier_counter),
        "stats": asdict(materialized_stats),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
