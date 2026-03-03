import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import RetrievalDataModule
from src.model.pl_module import RetrievalEvalLightningModule
from src.utils import log_if_rank_zero
from src.utils.evaluation_mode import enforce_retrieval_evaluation_isolation
from src.utils.logging import get_logger
from src.utils.metrics_io import to_jsonable
from src.utils.model_utils import apply_checkpoint_model_config, resolve_tagged_output_dir
from src.utils.script_setup import (
    configure_default_entrypoint_environment,
    initialize_run,
    resolve_model_source,
    resolve_trainer_settings,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_default_entrypoint_environment(
    load_env=True,
    set_matmul_precision=True,
)

def _resolve_index_path(cfg: DictConfig) -> Path:
    index_dir_value: str = str(cfg.encoding.index_dir)
    if not index_dir_value:
        raise ValueError("encoding.index_dir must be set for retrieval evaluation.")
    index_path: Path = resolve_tagged_output_dir(
        index_dir_value,
        model_name=str(cfg.model.name),
        tag=cfg.encoding.index_tag,
    )
    if not index_path.exists():
        raise FileNotFoundError(
            f"Retrieval index does not exist: {index_path}. "
            "Build the index first with script/index.py."
        )
    return index_path


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate")
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)
    cfg = enforce_retrieval_evaluation_isolation(cfg, logger=logger)

    cfg = resolve_model_source(cfg, logger=logger, set_nanobeir_flag=False)
    cfg = apply_checkpoint_model_config(
        cfg,
        checkpoint_path=cfg.testing.checkpoint_path,
        logger=logger,
    )

    index_path: Path = _resolve_index_path(cfg)
    log_if_rank_zero(logger, f"Using retrieval index: {index_path}")

    eval_module: RetrievalEvalLightningModule = RetrievalEvalLightningModule(cfg=cfg)
    data_module: RetrievalDataModule = RetrievalDataModule(cfg=cfg)
    eval_module.eval()

    trainer_kwargs, precision = resolve_trainer_settings(cfg.testing)
    trainer: L.Trainer = L.Trainer(
        precision=precision,
        deterministic=False,
        default_root_dir=cfg.log_dir,
        logger=False,
        enable_checkpointing=False,
        **trainer_kwargs,
    )

    test_results: list[dict[str, Any]] = trainer.test(
        model=eval_module, datamodule=data_module
    )
    if not test_results:
        log_if_rank_zero(
            logger,
            "trainer.test returned no metrics.",
            level="warning",
        )
        return

    log_if_rank_zero(logger, "Retrieval Evaluation Metrics:")
    dataloader_idx: int
    metrics: dict[str, Any]
    for dataloader_idx, metrics in enumerate(test_results):
        if not metrics:
            log_if_rank_zero(
                logger,
                f"dataloader_{dataloader_idx}: empty metrics",
                level="warning",
            )
            continue
        metric_name: str
        for metric_name in sorted(metrics):
            log_if_rank_zero(
                logger,
                f"dataloader_{dataloader_idx} {metric_name}: {metrics[metric_name]}",
            )

    output_path: str = os.path.join(cfg.log_dir, "evaluation_metrics.json")
    payload: dict[str, Any] = {
        "evaluation_mode": "retrieval_index_based",
        "checkpoint_path": cfg.testing.checkpoint_path,
        "hf_model_path": cfg.testing.hf_model_path,
        "index_path": str(index_path),
        "results": to_jsonable(test_results),
    }
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)
    log_if_rank_zero(logger, f"Saved evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
