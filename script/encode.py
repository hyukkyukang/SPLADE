import logging
import os
from typing import Any

import hydra
import lightning as L
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.module.encode import EncodeDataModule
from src.model.module.encode import SPLADEEncodeModule
from src.utils import log_if_rank_zero, set_seed
from src.utils.logging import get_logger, setup_tqdm_friendly_logging
from src.utils.script_setup import configure_script_environment
from src.utils.trainer import (
    get_cpu_trainer_kwargs,
    get_gpu_trainer_kwargs,
    resolve_precision,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_script_environment(
    load_env=False,
    set_tokenizers_parallelism=True,
    set_matmul_precision=True,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="encode")
def main(cfg: DictConfig) -> None:
    setup_tqdm_friendly_logging()
    os.makedirs(cfg.log_dir, exist_ok=True)
    set_seed(cfg.seed)
    log_if_rank_zero(logger, f"Random seed set to: {cfg.seed}")

    encode_module: SPLADEEncodeModule = SPLADEEncodeModule(cfg=cfg)
    data_module: EncodeDataModule = EncodeDataModule(cfg=cfg)

    encoding_cfg: DictConfig = cfg.encoding
    trainer_kwargs: dict[str, Any] = (
        get_cpu_trainer_kwargs(encoding_cfg)
        if encoding_cfg.use_cpu
        else get_gpu_trainer_kwargs(encoding_cfg)
    )
    precision: str = resolve_precision(encoding_cfg)

    trainer: L.Trainer = L.Trainer(
        precision=precision,
        logger=False,
        default_root_dir=cfg.log_dir,
        **trainer_kwargs,
    )
    trainer.predict(model=encode_module, datamodule=data_module)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
