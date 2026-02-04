import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Generator, Iterator, Optional, TextIO

import torch
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def log_if_rank_zero(logger: logging.Logger, message: str, level: str = "info") -> None:
    """Helper function to log only on rank 0 process.

    Uses PyTorch Lightning's @rank_zero_only decorator to ensure logging
    only happens on the main process during distributed training.
    If not distributed, logs the message as well.

    Args:
        logger: Logger instance to use for logging.
        message: The message to log.
        level: Log level to use ('info', 'debug', 'warning', 'error'). Defaults to 'info'.

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_if_rank_zero(logger, "Training started")
        >>> log_if_rank_zero(logger, "Missing config", level="warning")
    """
    level_name: str = str(level).lower()
    level_map: dict[str, int] = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logger.log(level_map.get(level_name, logging.INFO), message)


def get_global_rank() -> int:
    """Return the global rank when available, defaulting to 0."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    rank_env: str | None = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank_env is None:
        return 0
    try:
        return int(rank_env)
    except ValueError:
        return 0


def is_rank_zero() -> bool:
    """Check whether the current process is global rank zero."""
    return get_global_rank() == 0


def is_ddp_launcher_process() -> bool:
    """Identify the DDP launcher process created by Lightning."""
    launcher_flag: str | None = os.environ.get("SPLADE_DDP_LAUNCHER")
    if launcher_flag != "1":
        return False
    rank_env: str | None = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    # Launcher processes do not have rank environment variables set.
    return rank_env is None


@contextmanager
def suppress_output_if_not_rank_zero() -> Iterator[None]:
    """Silence stdout/stderr on non-zero ranks to avoid duplicate logs."""
    if is_ddp_launcher_process():
        devnull: TextIO
        # Launcher output is redundant once worker processes start.
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
        return
    if is_rank_zero():
        yield
        return
    devnull: TextIO
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def get_logger(name: str, file_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)

    # Ensure logs flow to Hydra's root handlers/formatters.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = True

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    return logger


def setup_tqdm_friendly_logging() -> None:
    logging.getLogger("tqdm").setLevel(logging.WARNING)


def patch_hydra_argparser_for_python314() -> None:
    """Patch Hydra's get_args_parser to fix Python 3.14+ compatibility.

    Python 3.14+ requires `help` argument values to implement `__contains__`,
    `__iter__`, and `__len__` methods. Hydra's `LazyCompletionHelp` class doesn't
    implement these methods, causing argparse errors. This function patches
    Hydra's `get_args_parser` with a fixed version that includes a properly
    implemented `LazyCompletionHelp` class.

    Should be called early in the script before any Hydra operations.
    """
    _logger = logging.getLogger("src.utils.logging")

    try:
        import hydra._internal.utils

        def _patched_get_args_parser() -> argparse.ArgumentParser:
            """Create Hydra's argument parser with fixed LazyCompletionHelp."""
            from hydra import __version__

            parser = argparse.ArgumentParser(add_help=False, description="Hydra")
            parser.add_argument(
                "--help", "-h", action="store_true", help="Application's help"
            )
            parser.add_argument(
                "--hydra-help", action="store_true", help="Hydra's help"
            )
            parser.add_argument(
                "--version",
                action="version",
                help="Show Hydra's version and exit",
                version=f"Hydra {__version__}",
            )
            parser.add_argument(
                "overrides",
                nargs="*",
                help="Any key=value arguments to override config values "
                "(use dots for.nested=overrides)",
            )

            parser.add_argument(
                "--cfg",
                "-c",
                choices=["job", "hydra", "all"],
                help="Show config instead of running [job|hydra|all]",
            )
            parser.add_argument(
                "--resolve",
                action="store_true",
                help="Used in conjunction with --cfg, resolve config interpolations "
                "before printing.",
            )

            parser.add_argument("--package", "-p", help="Config package to show")

            parser.add_argument("--run", "-r", action="store_true", help="Run a job")

            parser.add_argument(
                "--multirun",
                "-m",
                action="store_true",
                help="Run multiple jobs with the configured launcher and sweeper",
            )

            # Fixed LazyCompletionHelp with required methods for Python 3.14+
            class LazyCompletionHelp:
                """Help text for shell completion that satisfies Python 3.14+ requirements."""

                def __repr__(self) -> str:
                    return "Install or Uninstall shell completion"

                def __contains__(self, item: object) -> bool:
                    return False

                def __iter__(self):
                    return iter([])

                def __len__(self) -> int:
                    return 0

            parser.add_argument(
                "--shell-completion",
                "-sc",
                action="store_true",
                help=LazyCompletionHelp(),
            )

            parser.add_argument(
                "--config-path",
                "-cp",
                help="Overrides the config_path specified in hydra.main(). "
                "The config_path is absolute or relative to the Python file "
                "declaring @hydra.main()",
            )

            parser.add_argument(
                "--config-name",
                "-cn",
                help="Overrides the config_name specified in hydra.main()",
            )

            parser.add_argument(
                "--config-dir",
                "-cd",
                help="Adds an additional config dir to the config search path",
            )

            parser.add_argument(
                "--experimental-rerun",
                help="Rerun a job from a previous config pickle",
            )

            info_choices = [
                "all",
                "config",
                "defaults",
                "defaults-tree",
                "plugins",
                "searchpath",
            ]
            parser.add_argument(
                "--info",
                "-i",
                const="all",
                nargs="?",
                action="store",
                choices=info_choices,
                help=f"Print Hydra information [{'|'.join(info_choices)}]",
            )
            return parser

        # Apply the patch
        hydra._internal.utils.get_args_parser = _patched_get_args_parser

        # Also patch in hydra.main if it's already imported
        if "hydra.main" in sys.modules:
            sys.modules["hydra.main"].get_args_parser = _patched_get_args_parser
            log_if_rank_zero(
                _logger, "hydra.main.get_args_parser patched", level="debug"
            )

        log_if_rank_zero(
            _logger, "hydra._internal.utils.get_args_parser patched", level="debug"
        )

    except ImportError:
        log_if_rank_zero(
            _logger, "hydra._internal.utils not found, patch not applied", level="debug"
        )


def suppress_httpx_logging() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)


def suppress_pytorch_lightning_tips() -> None:
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


_LITLOGGER_TIP_PHRASES: tuple[str, ...] = ("💡 tip", "litlogger")
_LITMODEL_TIP_PHRASES: tuple[str, ...] = ("💡 tip", "litmodel")


class _LightningTipFilter(logging.Filter):
    """Filter out Lightning tips that mention litlogger or litmodel."""

    def filter(self, record: logging.LogRecord) -> bool:
        message: str = record.getMessage().lower()
        is_litlogger_tip: bool = all(
            phrase in message for phrase in _LITLOGGER_TIP_PHRASES
        )
        is_litmodel_tip: bool = all(
            phrase in message for phrase in _LITMODEL_TIP_PHRASES
        )
        return not (is_litlogger_tip or is_litmodel_tip)


_LIGHTNING_TIP_FILTER: _LightningTipFilter = _LightningTipFilter()


def _add_lightning_tip_filter(logger: logging.Logger) -> None:
    """Attach the Lightning tip filter when missing."""
    if _LIGHTNING_TIP_FILTER not in logger.filters:
        logger.addFilter(_LIGHTNING_TIP_FILTER)


def suppress_lightning_recommendation_tips() -> None:
    """Suppress the Lightning tips recommending litlogger or litmodels."""
    logger_names: tuple[str, ...] = (
        "pytorch_lightning.utilities.rank_zero",
        "lightning.pytorch.utilities.rank_zero",
        "lightning.fabric.utilities.rank_zero",
        "lightning_utilities.core.rank_zero",
    )
    for logger_name in logger_names:
        logger: logging.Logger = logging.getLogger(logger_name)
        _add_lightning_tip_filter(logger)

    rank_zero_module: Any | None
    try:
        import lightning_utilities.core.rank_zero as rank_zero_module
    except ImportError:
        rank_zero_module = None
    if rank_zero_module is None:
        return
    rank_zero_logger: logging.Logger = rank_zero_module.log
    _add_lightning_tip_filter(rank_zero_logger)


def suppress_dataloader_workers_warning() -> None:
    logging.getLogger("torch.utils.data").setLevel(logging.WARNING)


def suppress_accumulate_grad_stream_mismatch_warning() -> None:
    """Disable the AccumulateGrad stream mismatch warning in PyTorch."""
    graph_module: Any | None = (
        torch.autograd.graph if hasattr(torch.autograd, "graph") else None
    )
    if graph_module is None:
        return
    set_warn_fn: Any | None = (
        graph_module.set_warn_on_accumulate_grad_stream_mismatch
        if hasattr(graph_module, "set_warn_on_accumulate_grad_stream_mismatch")
        else None
    )
    if callable(set_warn_fn):
        # Avoid spamming warnings for benign DDP/compile stream mismatches.
        set_warn_fn(False)


@contextmanager
def loading_status(
    logger: logging.Logger,
    subject: str,
    *,
    loading_msg: Optional[str] = None,
    done_msg: Optional[str] = None,
    only_once: bool = False,
    rank_id: int = 0,
    worker_id: int = 0,
) -> Generator[None, None, None]:
    """Context manager that logs loading status with elapsed time.

    On enter, logs "Loading {subject}..." message.
    On exit, logs "Done loading {subject} ({elapsed_time})" message.
    Elapsed time is automatically formatted in a human-readable way.

    Args:
        logger: Logger instance for the status messages.
        subject: Description of what is being loaded (e.g., "query dataset for X").
        loading_msg: Custom loading message. Defaults to "Loading {subject}...".
        done_msg: Custom completion message. Defaults to "Done loading {subject}".
        only_once: If True, only show status on rank 0, worker 0.
        rank_id: Current process rank ID (for distributed training).
        worker_id: Current DataLoader worker ID.

    Yields:
        None: Control is yielded to the code block being wrapped.

    Example:
        >>> with loading_status(logger, "query dataset for BeIR/msmarco"):
        ...     dataset = load_dataset("BeIR/msmarco", "queries", split="train")
        # Logs: "[INFO] Loading query dataset for BeIR/msmarco..."
        # Then: "[INFO] Done loading query dataset for BeIR/msmarco (2.34s)"
    """
    # Build messages
    loading_text: str = loading_msg or f"Loading {subject}..."
    done_text: str = done_msg or f"Done loading {subject}"

    # Check if we should skip (only_once filtering)
    should_skip: bool = only_once and (rank_id != 0 or worker_id != 0)
    if should_skip:
        yield
        return

    # Log "Loading..." message, run the operation, then log "Done..." with elapsed time
    logger.info(loading_text)
    start_time: float = time.time()
    try:
        yield
    finally:
        elapsed: float = time.time() - start_time
        time_str: str = _format_elapsed_time(elapsed)
        logger.info(f"{done_text} ({time_str})")


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable way.

    Automatically chooses the most appropriate format based on duration:
    - Less than 1 minute: shows seconds (e.g., "2.34s")
    - Less than 1 hour: shows minutes and seconds (e.g., "5m 23s")
    - 1 hour or more: shows hours, minutes, and seconds (e.g., "1h 23m 45s")

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Human-readable time string.

    Examples:
        >>> _format_elapsed_time(2.345)
        '2.35s'
        >>> _format_elapsed_time(125.5)
        '2m 5s'
        >>> _format_elapsed_time(3725.0)
        '1h 2m 5s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes: int = int(seconds // 60)
        secs: int = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours: int = int(seconds // 3600)
        minutes: int = int((seconds % 3600) // 60)
        secs: int = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
