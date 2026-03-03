import logging
from typing import Any

from omegaconf import DictConfig, open_dict

from src.utils import log_if_rank_zero


def enforce_retrieval_evaluation_isolation(
    cfg: DictConfig, *, logger: logging.Logger
) -> DictConfig:
    """Enforce retrieval-only semantics for index-based evaluation script."""
    eval_type_value: Any = cfg.evaluation.get("type", "retrieval")
    eval_type: str = str(eval_type_value).lower()
    if eval_type != "retrieval":
        raise ValueError(
            "script/evaluation.py only supports full end-to-end retrieval over an "
            "existing index. Set evaluation.type=retrieval."
        )

    if "nanobeir" in cfg and bool(cfg.nanobeir.get("enabled", False)):
        with open_dict(cfg):
            cfg.nanobeir.enabled = False
        log_if_rank_zero(
            logger,
            "Ignoring nanobeir.enabled for retrieval evaluation script; "
            "NanoBEIR belongs to validation flow.",
            level="warning",
        )

    return cfg
