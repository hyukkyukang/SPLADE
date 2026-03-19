import logging
import os
import warnings
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.logging import (
    log_if_rank_zero,
    patch_hydra_argparser_for_python314,
    setup_tqdm_friendly_logging,
    suppress_lightning_recommendation_tips,
    suppress_dataloader_workers_warning,
    suppress_httpx_logging,
    suppress_pytorch_lightning_tips,
)
from src.utils.normalize import (
    normalize_optional_path as _normalize_optional_path_impl,
    normalize_optional_str as _normalize_optional_str_impl,
)
from src.utils.seed import set_seed
from src.utils.trainer import (
    get_cpu_trainer_kwargs,
    get_gpu_trainer_kwargs,
    resolve_precision,
)

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        _ = args, kwargs
        return False


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
    """Build the log directory, appending {tag|no_tag} as final segment."""
    tag_value: str | None = normalize_tag(tag)
    tag_segment: str = tag_value if tag_value is not None else "no_tag"
    return os.path.join(log_dir_base, tag_segment)


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


def configure_default_entrypoint_environment(
    *,
    load_env: bool,
    set_matmul_precision: bool = True,
) -> None:
    """Configure the standard runtime environment for script entrypoints."""
    configure_script_environment(
        load_env=bool(load_env),
        set_tokenizers_parallelism=True,
        set_matmul_precision=bool(set_matmul_precision),
        suppress_lightning_tips=True,
        suppress_httpx=True,
        suppress_dataloader_workers=True,
    )


def initialize_run(
    cfg: DictConfig,
    *,
    logger: logging.Logger,
    suppress_lightning_tips: bool = True,
) -> None:
    """Common run initialization for script entrypoints."""
    setup_tqdm_friendly_logging()
    if suppress_lightning_tips:
        suppress_lightning_recommendation_tips()
    os.makedirs(cfg.log_dir, exist_ok=True)
    set_seed(cfg.seed)
    log_if_rank_zero(logger, f"Random seed set to: {cfg.seed}")


def normalize_optional_path(value: Any) -> str | None:
    """Normalize optional path-like values from configs."""
    return _normalize_optional_path_impl(value)


def normalize_optional_str(value: Any) -> str | None:
    """Normalize optional string values from configs."""
    return _normalize_optional_str_impl(value)


def resolve_trainer_settings(cfg_section: DictConfig) -> tuple[dict[str, Any], str]:
    """Return trainer kwargs and precision for the config section."""
    trainer_kwargs: dict[str, Any] = (
        get_cpu_trainer_kwargs(cfg_section)
        if bool(cfg_section.use_cpu)
        else get_gpu_trainer_kwargs(cfg_section)
    )
    precision: str = resolve_precision(cfg_section)
    return trainer_kwargs, precision


def resolve_model_source(
    cfg: DictConfig,
    *,
    logger: logging.Logger,
    set_nanobeir_flag: bool = False,
) -> DictConfig:
    """Resolve model source (HF weights vs checkpoint) for evaluation/mining."""
    testing_cfg: DictConfig = cfg.testing
    hf_model_path: str | None = normalize_optional_path(
        testing_cfg.hf_model_path
    )
    checkpoint_path: str | None = normalize_optional_path(
        testing_cfg.checkpoint_path
    )

    if hf_model_path:
        if checkpoint_path:
            raise ValueError(
                "Provide either testing.hf_model_path or "
                "testing.checkpoint_path, not both."
            )
        cfg.model.huggingface_name = hf_model_path
        if set_nanobeir_flag and hasattr(cfg, "nanobeir"):
            cfg.nanobeir.use_huggingface_model = True
        log_if_rank_zero(logger, f"Using Hugging Face model: {hf_model_path}")
        return cfg

    if not checkpoint_path:
        raise ValueError(
            "testing.checkpoint_path must be set unless "
            "testing.hf_model_path is provided."
        )
    return cfg
