from collections.abc import Mapping
from typing import Any

from src.data.text_prefix import TextPrefix, slice_text_prefix
from src.utils.normalize import normalize_optional_str

PATENT_DOCUMENT_TEMPLATE_NAME: str = "patent_document_v1"
_PATENT_TEMPLATE_ALIASES: frozenset[str] = frozenset(
    {
        PATENT_DOCUMENT_TEMPLATE_NAME,
        "patent_document",
        "patent_doc",
    }
)


def normalize_patent_text(value: Any) -> str:
    if value is None:
        return ""
    normalized: str | None = normalize_optional_str(value)
    return "" if normalized is None else normalized


def format_patent_document_text(
    row: Mapping[str, Any],
    *,
    title_key: str = "title",
    abstract_key: str = "abstract",
    claims_key: str = "claims",
    description_key: str = "description",
    include_description: bool = True,
) -> str:
    parts: list[str] = []
    title: str = normalize_patent_text(row.get(title_key))
    abstract: str = normalize_patent_text(row.get(abstract_key))
    claims: str = normalize_patent_text(row.get(claims_key))
    description: str = normalize_patent_text(row.get(description_key))
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if claims:
        parts.append(f"Claims: {claims}")
    if include_description and description:
        parts.append(f"Description: {description}")
    return "\n".join(parts).strip()


def format_patent_document_text_prefix(
    row: Mapping[str, Any],
    *,
    char_budget: int,
    title_key: str = "title",
    abstract_key: str = "abstract",
    claims_key: str = "claims",
    description_key: str = "description",
    include_description: bool = True,
    boundary_window: int = 256,
) -> TextPrefix:
    """Format a patent document template without materializing the full text."""
    remaining_chars: int = max(int(char_budget), 0)
    if remaining_chars <= 0:
        return TextPrefix(text="", truncated=True)

    fields: list[tuple[str, str]] = []
    title: str = normalize_patent_text(row.get(title_key))
    abstract: str = normalize_patent_text(row.get(abstract_key))
    claims: str = normalize_patent_text(row.get(claims_key))
    description: str = normalize_patent_text(row.get(description_key))
    if title:
        fields.append(("Title", title))
    if abstract:
        fields.append(("Abstract", abstract))
    if claims:
        fields.append(("Claims", claims))
    if include_description and description:
        fields.append(("Description", description))

    if not fields:
        return TextPrefix(text="", truncated=False)

    parts: list[str] = []
    consumed_all_fields: bool = True
    for field_idx, (label, field_text) in enumerate(fields):
        separator: str = "\n" if parts else ""
        header: str = f"{label}: "
        required_chars: int = len(separator) + len(header)
        if remaining_chars <= required_chars:
            consumed_all_fields = False
            break

        available_chars: int = remaining_chars - required_chars
        if len(field_text) > available_chars:
            field_prefix: TextPrefix = slice_text_prefix(
                field_text,
                char_budget=available_chars,
                boundary_window=boundary_window,
            )
            parts.append(f"{separator}{header}{field_prefix.text}")
            consumed_all_fields = False
            break

        parts.append(f"{separator}{header}{field_text}")
        remaining_chars -= required_chars + len(field_text)
        if field_idx < len(fields) - 1 and remaining_chars <= 0:
            consumed_all_fields = False
            break

    return TextPrefix(
        text="".join(parts).strip(),
        truncated=not consumed_all_fields,
    )


def format_named_text_template(template_name: str, row: Mapping[str, Any]) -> str:
    normalized_name: str = str(template_name).strip().lower()
    if normalized_name in _PATENT_TEMPLATE_ALIASES:
        return format_patent_document_text(row)
    raise ValueError(f"Unsupported corpus text template: {template_name!r}")


__all__ = [
    "PATENT_DOCUMENT_TEMPLATE_NAME",
    "format_named_text_template",
    "format_patent_document_text",
    "format_patent_document_text_prefix",
    "normalize_patent_text",
]
