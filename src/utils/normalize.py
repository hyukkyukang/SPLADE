from typing import Any


def normalize_optional_path(value: Any) -> str | None:
    """Normalize optional path-like values from configs."""
    if value is None:
        return None
    text: str = str(value).strip()
    return text if text else None


def normalize_optional_str(value: Any) -> str | None:
    """Normalize optional string values from configs."""
    if value is None:
        return None
    text: str = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null"}:
        return None
    return text


def normalize_optional_bool(value: Any) -> bool | None:
    """Normalize optional boolean values from config-friendly inputs."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text: str = str(value).strip()
    if not text:
        return None
    lowered: str = text.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    raise ValueError(
        "Boolean fields must be one of true/false, 1/0, yes/no, on/off; "
        f"got {value!r}."
    )
