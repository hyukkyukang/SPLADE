import logging
import os
from typing import Optional


def get_logger(name: str, file_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def setup_tqdm_friendly_logging() -> None:
    logging.getLogger("tqdm").setLevel(logging.WARNING)


def patch_hydra_argparser_for_python314() -> None:
    return None


def suppress_httpx_logging() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)


def suppress_pytorch_lightning_tips() -> None:
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


def suppress_dataloader_workers_warning() -> None:
    logging.getLogger("torch.utils.data").setLevel(logging.WARNING)
