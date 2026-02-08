import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import lightning as L
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import ScoringDataModule, ScoringHardNegativesDataModule
from src.model.pl_module import CrossEncoderScoringModule
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.script_setup import (
    configure_script_environment,
    initialize_run,
    normalize_optional_str,
    resolve_trainer_settings,
)

logger: logging.Logger = get_logger("script.score_cross_encoder_ensemble", __file__)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=True,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


@dataclass(frozen=True)
class ModelEntry:
    model_name: str
    model_backend: str
    model_checkpoint_path: str | None
    output_basename: str


def _append_rank_suffix(path: Path, rank: int) -> Path:
    suffix: str = path.suffix
    stem: str = path.stem
    if suffix:
        return path.with_name(f"{stem}.rank{rank}{suffix}")
    return path.with_name(f"{path.name}.rank{rank}")


def _resolve_output_path(scoring_cfg: DictConfig) -> Path:
    output_dir: str = str(scoring_cfg.output_dir)
    output_basename: str = str(scoring_cfg.output_basename)
    output_format: str = str(scoring_cfg.output_format).lower()
    if output_format != "jsonl":
        raise ValueError("Only jsonl output is supported for scoring.")
    return Path(output_dir) / f"{output_basename}.jsonl"


def _merge_rank_outputs(scoring_cfg: DictConfig, trainer: L.Trainer) -> None:
    merge_ranks: bool = bool(scoring_cfg.merge_ranks)
    if not merge_ranks:
        return
    world_size: int = int(trainer.world_size)
    if world_size <= 1:
        return
    if int(trainer.global_rank) != 0:
        return
    output_path: Path = _resolve_output_path(scoring_cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_handle:
        for rank in range(world_size):
            rank_path: Path = _append_rank_suffix(output_path, rank)
            if not rank_path.exists():
                log_if_rank_zero(
                    logger,
                    f"Missing rank output file: {rank_path}",
                    level="warning",
                )
                continue
            with rank_path.open("r", encoding="utf-8") as rank_handle:
                for line in rank_handle:
                    output_handle.write(line)
    log_if_rank_zero(logger, f"Merged rank outputs into {output_path}")


def _convert_jsonl_to_parquet(jsonl_path: Path) -> Path:
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    output_path: Path = jsonl_path.with_suffix(".parquet")
    tmp_path: Path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    dataset.to_parquet(str(tmp_path))
    tmp_path.replace(output_path)
    return output_path


def _postprocess_outputs(scoring_cfg: DictConfig, trainer: L.Trainer) -> None:
    post_cfg: DictConfig = scoring_cfg.postprocess
    if not bool(post_cfg.enabled):
        return
    if int(trainer.global_rank) != 0:
        return
    output_path: Path = _resolve_output_path(scoring_cfg)
    parquet_path: Path = _convert_jsonl_to_parquet(output_path)
    log_if_rank_zero(
        logger,
        f"Converted JSONL to Parquet: {output_path} -> {parquet_path}.",
    )
    if not bool(post_cfg.cleanup_jsonl):
        return
    output_path.unlink()
    log_if_rank_zero(logger, f"Removed JSONL output {output_path}")


def _resolve_models(cfg: DictConfig) -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    for entry_cfg in cfg.models:
        model_name: str = str(entry_cfg.model_name)
        model_backend: str = str(entry_cfg.model_backend)
        output_basename: str = str(entry_cfg.output_basename)
        checkpoint_path_value: str | None = normalize_optional_str(
            entry_cfg.model_checkpoint_path
        )
        entries.append(
            ModelEntry(
                model_name=model_name,
                model_backend=model_backend,
                model_checkpoint_path=checkpoint_path_value,
                output_basename=output_basename,
            )
        )
    if not entries:
        raise ValueError("models list is empty.")
    return entries


def _build_data_module(cfg: DictConfig) -> L.LightningDataModule:
    data_module_name: str = str(cfg.scoring.data_module)
    if data_module_name == "hard_negatives":
        return ScoringHardNegativesDataModule(cfg=cfg)
    if data_module_name == "standard":
        return ScoringDataModule(cfg=cfg)
    raise ValueError(f"Unsupported scoring.data_module={data_module_name}")


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name="score_cross_encoder_ensemble"
)
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)
    models = _resolve_models(cfg)

    scoring_cfg: DictConfig = cfg.scoring
    trainer_kwargs, precision = resolve_trainer_settings(scoring_cfg)

    for entry in models:
        scoring_cfg.model_name = entry.model_name
        scoring_cfg.model_backend = entry.model_backend
        scoring_cfg.model_checkpoint_path = entry.model_checkpoint_path
        scoring_cfg.output_basename = entry.output_basename

        log_if_rank_zero(
            logger,
            f"Scoring model={entry.model_name} backend={entry.model_backend} "
            f"output={entry.output_basename}",
        )

        scoring_module = CrossEncoderScoringModule(cfg=cfg)
        data_module = _build_data_module(cfg)
        trainer = L.Trainer(
            precision=precision,
            logger=False,
            default_root_dir=cfg.log_dir,
            **trainer_kwargs,
        )
        if bool(scoring_cfg.run_predict):
            trainer.predict(model=scoring_module, datamodule=data_module)
        else:
            log_if_rank_zero(
                logger,
                "Skipping prediction because scoring.run_predict is false.",
                level="warning",
            )
        _merge_rank_outputs(scoring_cfg, trainer)
        _postprocess_outputs(scoring_cfg, trainer)

    log_if_rank_zero(logger, "Cross-encoder ensemble scoring complete")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
