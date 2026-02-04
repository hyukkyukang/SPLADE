import logging
import os
import warnings

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.utils.logging import (
    patch_hydra_argparser_for_python314,
    suppress_dataloader_workers_warning,
    suppress_httpx_logging,
    suppress_pytorch_lightning_tips,
)


def normalize_tag(tag: object | None) -> str | None:
    """Normalize tag values into a clean string or None."""
    if tag is None:
        return None
    # Treat empty or whitespace-only tags as missing.
    tag_value: str = str(tag).strip()
    if not tag_value:
        return None
    return tag_value


def _resolve_tagged_log_dir(log_dir_base: str, tag: str | None) -> str:
    """Build the log directory, appending the tag when provided."""
    tag_value: str | None = normalize_tag(tag)
    if tag_value is None:
        return log_dir_base
    return os.path.join(log_dir_base, tag_value)


def _register_tagged_log_dir_resolver() -> None:
    """Register the tagged log dir resolver for Hydra configs."""
    resolver_name: str = "tagged_log_dir"
    has_resolver: bool = False
    if hasattr(OmegaConf, "has_resolver"):
        has_resolver = OmegaConf.has_resolver(resolver_name)
    if has_resolver:
        return
    # Keep resolver registration centralized for all entrypoints.
    try:
        OmegaConf.register_new_resolver(resolver_name, _resolve_tagged_log_dir)
    except ValueError:
        # Resolver may already be registered in the current process.
        return


def configure_script_environment(
    *,
    load_env: bool,
    set_tokenizers_parallelism: bool,
    set_matmul_precision: bool,
    suppress_lightning_tips: bool,
    suppress_httpx: bool,
    suppress_dataloader_workers: bool,
) -> None:
    """Apply shared script setup for Hydra entrypoints."""
    _register_tagged_log_dir_resolver()
    # Silence noisy FutureWarning messages from dependencies.
    warnings.simplefilter(action="ignore", category=FutureWarning)
    # Route Python warnings through logging so Hydra formatting applies.
    logging.captureWarnings(True)
    # Patch Hydra's argparser early for Python 3.14+ compatibility.
    patch_hydra_argparser_for_python314()

    if load_env:
        # Load environment variables from .env when requested.
        load_dotenv()

    if set_tokenizers_parallelism:
        # Avoid tokenizer parallelism warnings in multi-process contexts.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if set_matmul_precision:
        # Prefer higher precision matmul for stability on supported hardware.
        torch.set_float32_matmul_precision("high")

    if suppress_lightning_tips:
        suppress_pytorch_lightning_tips()

    if suppress_httpx:
        suppress_httpx_logging()

    if suppress_dataloader_workers:
        suppress_dataloader_workers_warning()
