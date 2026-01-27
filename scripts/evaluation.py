import logging
import os
from typing import Any

import hydra
import lightning as L
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.module.eval import EvalDataModule
from src.model.module.eval import SPLADEEvaluationModule
from src.utils import log_if_rank_zero, set_seed
from src.utils.logging import (
    get_logger,
    setup_tqdm_friendly_logging,
)
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


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate")
def main(cfg: DictConfig) -> None:
    setup_tqdm_friendly_logging()
    os.makedirs(cfg.log_dir, exist_ok=True)
    set_seed(cfg.seed)
    log_if_rank_zero(logger, f"Random seed set to: {cfg.seed}")

    eval_module: SPLADEEvaluationModule = SPLADEEvaluationModule(cfg=cfg)
    eval_module.eval()
    data_module: EvalDataModule = EvalDataModule(cfg=cfg)

    testing_cfg: DictConfig = cfg.testing
    trainer_kwargs: dict[str, Any] = (
        get_cpu_trainer_kwargs(testing_cfg)
        if testing_cfg.use_cpu
        else get_gpu_trainer_kwargs(testing_cfg)
    )
    precision: str = resolve_precision(testing_cfg)

    trainer: L.Trainer = L.Trainer(
        precision=precision,
        default_root_dir=cfg.log_dir,
        logger=False,
        **trainer_kwargs,
    )

    results: list[dict[str, float]] = trainer.test(
        model=eval_module, datamodule=data_module
    )
    if results:
        metrics: dict[str, float] = results[0]
        for name, value in metrics.items():
            logger.info("%s %s: %.4f", str(cfg.dataset.name), name, value)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
