from __future__ import annotations

import os
import warnings

import torch
from dotenv import load_dotenv

from src.utils.logging import (
    patch_hydra_argparser_for_python314,
    suppress_dataloader_workers_warning,
    suppress_httpx_logging,
    suppress_pytorch_lightning_tips,
)


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
    # Silence noisy FutureWarning messages from dependencies.
    warnings.simplefilter(action="ignore", category=FutureWarning)
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
