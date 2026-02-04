import logging
import os
from typing import Any

import hydra
import lightning as L
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import RerankingDataModule, RetrievalDataModule
from src.model.pl_module import RerankingLightningModule, RetrievalLightningModule
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

    eval_type: str = str(cfg.evaluation.type).lower()
    eval_module: L.LightningModule
    data_module: L.LightningDataModule
    if eval_type == "retrieval":
        eval_module = RetrievalLightningModule(cfg=cfg)
        data_module = RetrievalDataModule(cfg=cfg)
    elif eval_type == "reranking":
        eval_module = RerankingLightningModule(cfg=cfg)
        data_module = RerankingDataModule(cfg=cfg)
    else:
        raise ValueError(f"Unsupported evaluation.type: {eval_type}")
    eval_module.eval()

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
            log_if_rank_zero(
                logger,
                f"{str(cfg.dataset.name)} {name}: {value:.4f}",
            )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
