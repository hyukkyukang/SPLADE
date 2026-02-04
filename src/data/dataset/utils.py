import random
from typing import Any

from omegaconf import DictConfig


def normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized: str = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
    return str(value)


def require_cfg_str(cfg: DictConfig, key: str) -> str:
    if key not in cfg:
        raise ValueError(f"Missing required dataset config key: {key}")
    value: object | None = cfg.get(key)
    if value is None:
        raise ValueError(f"Dataset config key {key} must be set.")
    value_str: str = str(value).strip()
    if not value_str:
        raise ValueError(f"Dataset config key {key} must be non-empty.")
    return value_str


def optional_cfg_str(cfg: DictConfig, key: str) -> str | None:
    if key not in cfg:
        raise ValueError(f"Missing required dataset config key: {key}")
    value: object | None = cfg.get(key)
    if value is None:
        return None
    value_str: str = str(value).strip()
    return value_str or None


def parse_triplet_line(line: str, row_idx: int) -> tuple[str, str, str, str] | None:
    stripped: str = line.strip()
    if not stripped:
        return None
    parts: list[str] = stripped.split("\t")
    if len(parts) == 3:
        query_text: str
        pos_text: str
        neg_text: str
        query_text, pos_text, neg_text = parts
        qid: str = str(row_idx)
    elif len(parts) == 4:
        qid = parts[0].strip()
        query_text = parts[1]
        pos_text = parts[2]
        neg_text = parts[3]
    else:
        return None
    return qid.strip(), query_text.strip(), pos_text.strip(), neg_text.strip()


def parse_inline_scores(score_values: Any, doc_ids: list[str]) -> list[float] | None:
    if score_values is None:
        return None
    if isinstance(score_values, (list, tuple)):
        if len(score_values) == len(doc_ids):
            return [float(score) for score in score_values]
        return None
    if isinstance(score_values, (int, float)):
        if len(doc_ids) == 1:
            return [float(score_values)]
        return None
    return None


def sample_items(items: list[Any], count: int, rng: random.Random) -> list[Any]:
    if count <= 0:
        return []
    if len(items) <= count:
        return items
    return rng.sample(items, count)
