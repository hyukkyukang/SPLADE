import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Sequence, Value, load_dataset
import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import RetrievalDataModule
from src.model.pl_module import RetrievalSearchLightningModule
from src.utils import log_if_rank_zero
from src.utils.logging import get_logger
from src.utils.model_utils import apply_checkpoint_model_config
from src.utils.script_setup import (
    configure_script_environment,
    initialize_run,
    normalize_optional_str,
    resolve_model_source,
    resolve_trainer_settings,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=True,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


@dataclass(frozen=True)
class PostprocessSettings:
    cleanup_jsonl: bool
    require_count_match: bool
    add_rank_column: bool


def _slugify_component(value: Any, *, fallback: str) -> str:
    normalized: str = normalize_optional_str(value) or fallback
    slug: str = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower()).strip("_")
    return slug or fallback


def _resolve_dataset_name(cfg: DictConfig) -> str:
    dataset_cfgs: list[DictConfig] = []
    if "retrieval_dataset" in cfg:
        dataset_cfgs.append(cfg.retrieval_dataset)
    if "train_dataset" in cfg:
        dataset_cfgs.append(cfg.train_dataset)
    if "dataset" in cfg:
        dataset_cfgs.append(cfg.dataset)

    for dataset_cfg in dataset_cfgs:
        if dataset_cfg is None:
            continue
        for value in (
            dataset_cfg.name,
            dataset_cfg.beir_dataset,
            dataset_cfg.hf_name,
        ):
            if normalize_optional_str(value) is not None:
                return _slugify_component(value, fallback="dataset")
    return "dataset"


def _resolve_model_name(cfg: DictConfig) -> str:
    for value in (
        cfg.model.name,
        cfg.model.huggingface_name,
    ):
        if normalize_optional_str(value) is not None:
            return _slugify_component(value, fallback="model")
    return "model"


def _resolve_output_basename(cfg: DictConfig) -> str:
    dataset_name: str = _resolve_dataset_name(cfg)
    model_name: str = _resolve_model_name(cfg)
    return f"{dataset_name}_{model_name}_hardneg"


def _configure_mining_output(cfg: DictConfig) -> None:
    if "mining" not in cfg:
        return
    mining_cfg: DictConfig = cfg.mining
    output_dir_value: str | None = normalize_optional_str(mining_cfg.output_dir_base)

    if not output_dir_value:
        return

    dataset_name: str = _resolve_dataset_name(cfg)
    model_name: str = _resolve_model_name(cfg)
    output_dir = Path(output_dir_value) / dataset_name / model_name
    run_filename: str = (
        normalize_optional_str(mining_cfg.run_filename) or "search_results.jsonl"
    )
    testing_cfg: DictConfig = cfg.testing
    run_path_value: str | None = normalize_optional_str(
        testing_cfg.run_path
    )
    if run_path_value is None:
        run_path = output_dir / run_filename
        testing_cfg.run_path = str(run_path)
    if not bool(testing_cfg.save_run):
        testing_cfg.save_run = True
    log_if_rank_zero(
        logger,
        f"Mining output run_path set to {testing_cfg.run_path}.",
    )


def _resolve_postprocess_settings(cfg: DictConfig) -> PostprocessSettings | None:
    if "mining" not in cfg:
        return None
    mining_cfg: DictConfig = cfg.mining
    post_cfg: DictConfig = mining_cfg.postprocess
    if not bool(post_cfg.enabled):
        return None
    cleanup_jsonl: bool = bool(post_cfg.cleanup_jsonl)
    require_count_match: bool = bool(post_cfg.require_count_match)
    add_rank_column: bool = bool(post_cfg.add_rank_column)
    return PostprocessSettings(
        cleanup_jsonl=cleanup_jsonl,
        require_count_match=require_count_match,
        add_rank_column=add_rank_column,
    )


def _append_rank_suffix(path: Path, rank: int) -> Path:
    suffix: str = path.suffix
    stem: str = path.stem
    if suffix:
        return path.with_name(f"{stem}.rank{rank}{suffix}")
    return path.with_name(f"{path.name}.rank{rank}")


def _merge_rank_outputs(cfg: DictConfig, trainer: L.Trainer) -> None:
    search_cfg: DictConfig = cfg.search
    merge_ranks: bool = bool(search_cfg.merge_ranks)
    if not merge_ranks:
        return

    world_size: int = int(trainer.world_size)
    global_rank: int = int(trainer.global_rank)
    if world_size <= 1:
        return

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if global_rank != 0:
        return

    if not bool(cfg.testing.save_run):
        log_if_rank_zero(
            logger,
            "Skipping rank merge because testing.save_run is false.",
            level="warning",
        )
        return

    run_path_value: str | None = cfg.testing.run_path
    if not run_path_value:
        raise ValueError("testing.run_path must be set to merge rank outputs.")

    run_path = Path(str(run_path_value))
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("w", encoding="utf-8") as output_handle:
        for rank in range(world_size):
            rank_path: Path = _append_rank_suffix(run_path, rank)
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
    log_if_rank_zero(logger, f"Merged rank outputs into {run_path}")


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _build_search_result_features(include_query_text: bool) -> Features:
    feature_dict: dict[str, Any] = {
        "qid": Value("string"),
        "doc_ids": Sequence(Value("string")),
        "scores": Sequence(Value("float32")),
    }
    if include_query_text:
        feature_dict["query_text"] = Value("string")
    return Features(feature_dict)


def _resolve_jsonl_inputs(run_path: Path, world_size: int) -> list[tuple[int, Path]]:
    if world_size <= 1:
        return [(0, run_path)]
    return [
        (rank, _append_rank_suffix(run_path, rank)) for rank in range(world_size)
    ]


def _add_rank_column(dataset: Dataset, rank: int) -> Dataset:
    if len(dataset) == 0:
        return dataset.add_column("rank", [])

    def _fill_rank(batch: dict[str, list[Any]]) -> dict[str, list[int]]:
        batch_len = len(next(iter(batch.values())))
        return {"rank": [rank] * batch_len}

    return dataset.map(_fill_rank, batched=True)


def _convert_jsonl_to_parquet(
    jsonl_path: Path,
    *,
    add_rank_column: bool,
    rank: int | None,
    include_query_text: bool,
    count_jsonl_rows: bool,
) -> tuple[Path, int | None, int]:
    jsonl_rows: int | None = (
        _count_jsonl_rows(jsonl_path) if count_jsonl_rows else None
    )
    dataset: Dataset = load_dataset(
        "json",
        data_files=str(jsonl_path),
        split="train",
        features=_build_search_result_features(include_query_text),
    )
    if add_rank_column and rank is not None:
        dataset = _add_rank_column(dataset, rank)
    output_dir: Path = jsonl_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = output_dir / jsonl_path.with_suffix(".parquet").name
    tmp_path: Path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    dataset.to_parquet(str(tmp_path))
    tmp_path.replace(output_path)
    parquet_rows: int = len(dataset)
    return output_path, jsonl_rows, parquet_rows


def _postprocess_outputs(cfg: DictConfig, trainer: L.Trainer) -> None:
    settings: PostprocessSettings | None = _resolve_postprocess_settings(cfg)
    if settings is None:
        return

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if int(trainer.global_rank) != 0:
        return

    if not bool(cfg.testing.save_run):
        log_if_rank_zero(
            logger,
            "Skipping postprocess because testing.save_run is false.",
            level="warning",
        )
        return

    run_path_value: str | None = normalize_optional_str(cfg.testing.run_path)
    if run_path_value is None:
        log_if_rank_zero(
            logger,
            "Skipping postprocess because testing.run_path is not set.",
            level="warning",
        )
        return

    run_path = Path(run_path_value)
    search_cfg: DictConfig = cfg.search
    include_query_text: bool = bool(search_cfg.include_query_text)
    count_jsonl_rows: bool = bool(
        settings.cleanup_jsonl and settings.require_count_match
    )
    jsonl_inputs = _resolve_jsonl_inputs(run_path, int(trainer.world_size))
    if not jsonl_inputs:
        log_if_rank_zero(logger, "No JSONL inputs found for postprocess.")
        return

    converted = 0
    deleted = 0
    skipped = 0
    for rank, jsonl_path in jsonl_inputs:
        if not jsonl_path.exists():
            log_if_rank_zero(
                logger,
                f"Missing JSONL file for postprocess: {jsonl_path}",
                level="warning",
            )
            skipped += 1
            continue
        output_path, jsonl_rows, parquet_rows = _convert_jsonl_to_parquet(
            jsonl_path,
            add_rank_column=settings.add_rank_column,
            rank=rank,
            include_query_text=include_query_text,
            count_jsonl_rows=count_jsonl_rows,
        )
        converted += 1
        if jsonl_rows is None:
            log_if_rank_zero(
                logger,
                "Converted JSONL to Parquet: "
                f"{jsonl_path} -> {output_path} "
                f"({parquet_rows} rows).",
            )
        else:
            log_if_rank_zero(
                logger,
                "Converted JSONL to Parquet: "
                f"{jsonl_path} -> {output_path} "
                f"({jsonl_rows} rows).",
            )
        if not settings.cleanup_jsonl:
            continue
        if settings.require_count_match and jsonl_rows != parquet_rows:
            log_if_rank_zero(
                logger,
                "Skipping JSONL cleanup due to row count mismatch: "
                f"{jsonl_path} ({jsonl_rows}) vs {output_path} ({parquet_rows}).",
                level="warning",
            )
            continue
        jsonl_path.unlink()
        deleted += 1

    log_if_rank_zero(
        logger,
        "Postprocess complete: "
        f"converted={converted}, deleted={deleted}, skipped={skipped}.",
    )


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="mine_hard_negative")
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)
    cfg = resolve_model_source(cfg, logger=logger)
    cfg = apply_checkpoint_model_config(
        cfg,
        checkpoint_path=cfg.testing.checkpoint_path,
        logger=logger,
    )
    _configure_mining_output(cfg)

    search_module = RetrievalSearchLightningModule(cfg=cfg)
    data_module = RetrievalDataModule(cfg=cfg)
    search_module.eval()

    testing_cfg: DictConfig = cfg.testing
    trainer_kwargs, precision = resolve_trainer_settings(testing_cfg)

    trainer: L.Trainer = L.Trainer(
        precision=precision,
        default_root_dir=cfg.log_dir,
        logger=False,
        **trainer_kwargs,
    )
    trainer.test(model=search_module, datamodule=data_module)
    _merge_rank_outputs(cfg, trainer)
    _postprocess_outputs(cfg, trainer)
    log_if_rank_zero(logger, "Search complete")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
