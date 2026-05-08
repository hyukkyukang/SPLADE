from dataclasses import dataclass


@dataclass(frozen=True)
class TextPrefix:
    """A prefix slice of a larger text payload."""

    text: str
    truncated: bool


def slice_text_prefix(
    text: str,
    *,
    char_budget: int,
    boundary_window: int = 256,
) -> TextPrefix:
    """Slice text near a character budget while preferring word boundaries."""
    normalized_text: str = str(text)
    budget: int = max(int(char_budget), 0)
    if budget <= 0:
        return TextPrefix(text="", truncated=bool(normalized_text))
    if len(normalized_text) <= budget:
        return TextPrefix(text=normalized_text, truncated=False)

    end: int = min(len(normalized_text), budget)
    max_end: int = min(len(normalized_text), budget + max(int(boundary_window), 0))
    while end < max_end and not normalized_text[end].isspace():
        end += 1

    if end == budget:
        prefix_text: str = normalized_text[:budget]
    else:
        prefix_text = normalized_text[:end]
    return TextPrefix(text=prefix_text.rstrip(), truncated=True)


__all__ = ["TextPrefix", "slice_text_prefix"]
