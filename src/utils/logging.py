import argparse
import logging
import os
import sys
from typing import Any, Optional

import torch


def get_logger(name: str, file_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
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
            _logger.debug("hydra.main.get_args_parser patched")

        _logger.debug("hydra._internal.utils.get_args_parser patched")

    except ImportError:
        _logger.debug("hydra._internal.utils not found, patch not applied")


def suppress_httpx_logging() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)


def suppress_pytorch_lightning_tips() -> None:
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


def suppress_dataloader_workers_warning() -> None:
    logging.getLogger("torch.utils.data").setLevel(logging.WARNING)


def suppress_accumulate_grad_stream_mismatch_warning() -> None:
    """Disable the AccumulateGrad stream mismatch warning in PyTorch."""
    graph_module: Any | None = getattr(torch.autograd, "graph", None)
    if graph_module is None:
        return
    set_warn_fn: Any | None = getattr(
        graph_module, "set_warn_on_accumulate_grad_stream_mismatch", None
    )
    if callable(set_warn_fn):
        # Avoid spamming warnings for benign DDP/compile stream mismatches.
        set_warn_fn(False)
