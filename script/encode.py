import logging
from typing import Any

import hydra
import lightning as L
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import EncodeDataModule
from src.model.pl_module import SPLADEEncodeModule
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.model_utils import apply_checkpoint_model_config
from src.utils.script_setup import (
    configure_default_entrypoint_environment,
    initialize_run,
    resolve_trainer_settings,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_default_entrypoint_environment(
    load_env=True,
    set_matmul_precision=True,
)


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="encode")
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)

    cfg = apply_checkpoint_model_config(
        cfg,
        checkpoint_path=cfg.encoding.checkpoint_path,
        logger=logger,
    )

    encode_module: SPLADEEncodeModule = SPLADEEncodeModule(cfg=cfg)
    data_module: EncodeDataModule = EncodeDataModule(cfg=cfg)

    encoding_cfg: DictConfig = cfg.encoding
    trainer_kwargs, precision = resolve_trainer_settings(encoding_cfg)

    trainer: L.Trainer = L.Trainer(
        precision=precision,
        logger=False,
        default_root_dir=cfg.log_dir,
        **trainer_kwargs,
    )
    trainer.predict(model=encode_module, datamodule=data_module)
    log_if_rank_zero(logger, "Encoding complete")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
