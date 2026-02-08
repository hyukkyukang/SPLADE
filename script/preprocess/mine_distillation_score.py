import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import constants as hf_constants
import hydra
import lightning as L
import torch
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

logger: logging.Logger = get_logger("scripts.preprocess.score_cross_encoder", __file__)

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


def _normalize_namespace(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", value.strip().lower())


def _normalize_repo_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip().lower()).strip("-")


def _resolve_hf_dataset_name(
    hf_name: str, *, scoring_namespace: str | None
) -> str:
    hf_name = hf_name.strip()
    if "/" in hf_name:
        source_namespace_raw, repo_raw = hf_name.split("/", 1)
    else:
        source_namespace_raw, repo_raw = "", hf_name

    repo_name: str = _normalize_repo_name(repo_raw)
    if not repo_name:
        return "dataset"

    source_namespace: str = _normalize_namespace(source_namespace_raw)
    if not source_namespace:
        return repo_name

    scoring_namespace_norm: str = (
        _normalize_namespace(scoring_namespace) if scoring_namespace else ""
    )
    if scoring_namespace_norm and source_namespace == scoring_namespace_norm:
        return repo_name

    tokens = [token for token in repo_name.split("-") if token]
    if not tokens:
        return repo_name
    if len(tokens) == 1:
        return f"{tokens[0]}-{source_namespace}"
    return "-".join([tokens[0], source_namespace, *tokens[1:]])


def _resolve_dataset_name(cfg: DictConfig) -> str:
    if "score_dataset" not in cfg:
        return "dataset"
    dataset_cfg: DictConfig = cfg.score_dataset
    name_value: str | None = normalize_optional_str(dataset_cfg.name)
    if name_value is not None:
        return _slugify_component(name_value, fallback="dataset")
    beir_value: str | None = normalize_optional_str(dataset_cfg.beir_dataset)
    if beir_value is not None:
        return _slugify_component(beir_value, fallback="dataset")
    hf_name_value: str | None = normalize_optional_str(dataset_cfg.hf_name)
    if hf_name_value is not None:
        scoring_cfg: DictConfig = cfg.scoring
        namespace_value: str | None = normalize_optional_str(scoring_cfg.hf_namespace)
        return _resolve_hf_dataset_name(
            hf_name_value,
            scoring_namespace=namespace_value,
        )
    return "dataset"


def _resolve_hub_repo_id(cfg: DictConfig) -> str:
    scoring_cfg: DictConfig = cfg.scoring
    repo_id_override: str | None = normalize_optional_str(scoring_cfg.hf_repo_id)
    if repo_id_override is not None:
        return repo_id_override
    namespace_value: str | None = normalize_optional_str(scoring_cfg.hf_namespace)
    namespace: str = namespace_value or "Hyukkyu"
    dataset_name: str = _resolve_dataset_name(cfg)
    return f"{namespace}/{dataset_name}-scores"


def _apply_hf_hub_timeouts(scoring_cfg: DictConfig) -> None:
    etag_timeout_value = scoring_cfg.hf_hub_etag_timeout
    if etag_timeout_value is not None:
        etag_timeout = int(etag_timeout_value)
        os.environ["HF_HUB_ETAG_TIMEOUT"] = str(etag_timeout)
        hf_constants.HF_HUB_ETAG_TIMEOUT = etag_timeout
    download_timeout_value = scoring_cfg.hf_hub_download_timeout
    if download_timeout_value is not None:
        download_timeout = int(download_timeout_value)
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(download_timeout)
        hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = download_timeout


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
        raise ValueError("Only jsonl output is supported for rank merging.")
    return Path(output_dir) / f"{output_basename}.jsonl"


def _resolve_postprocess_settings(cfg: DictConfig) -> PostprocessSettings | None:
    scoring_cfg: DictConfig = cfg.scoring
    post_cfg: DictConfig = scoring_cfg.postprocess
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


def _resolve_parquet_outputs(cfg: DictConfig, trainer: L.Trainer) -> list[Path]:
    output_path: Path = _resolve_output_path(cfg.scoring)
    world_size: int = int(trainer.world_size)
    if world_size <= 1:
        parquet_path = output_path.with_suffix(".parquet")
        return [parquet_path] if parquet_path.exists() else []
    parquet_paths: list[Path] = []
    for rank in range(world_size):
        parquet_path = _append_rank_suffix(output_path, rank).with_suffix(".parquet")
        if parquet_path.exists():
            parquet_paths.append(parquet_path)
    return parquet_paths


def _upload_parquet_to_hub(cfg: DictConfig, trainer: L.Trainer) -> None:
    scoring_cfg: DictConfig = cfg.scoring
    if not bool(scoring_cfg.upload_to_hub):
        return
    if int(trainer.global_rank) != 0:
        return

    post_cfg: DictConfig = scoring_cfg.postprocess
    if not bool(post_cfg.enabled):
        log_if_rank_zero(
            logger,
            "Skipping hub upload because scoring.postprocess.enabled is false.",
            level="warning",
        )
        return

    parquet_paths = _resolve_parquet_outputs(cfg, trainer)
    if not parquet_paths:
        log_if_rank_zero(
            logger,
            "Skipping hub upload because no parquet outputs were found.",
            level="warning",
        )
        return

    repo_id: str = _resolve_hub_repo_id(cfg)
    token = normalize_optional_str(
        os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    if token is None:
        raise ValueError(
            "HF_TOKEN or HUGGINGFACE_HUB_TOKEN must be set to upload the dataset."
        )

    data_files = {"train": [str(path) for path in parquet_paths]}
    log_if_rank_zero(
        logger,
        f"Uploading parquet dataset to {repo_id} from {len(parquet_paths)} shard(s).",
    )
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    dataset.push_to_hub(repo_id, token=token)
    log_if_rank_zero(logger, f"Upload complete: {repo_id}")


def _merge_rank_outputs(cfg: DictConfig, trainer: L.Trainer) -> None:
    scoring_cfg: DictConfig = cfg.scoring
    merge_ranks: bool = bool(scoring_cfg.merge_ranks)
    if not merge_ranks:
        return

    world_size: int = int(trainer.world_size)
    if world_size <= 1:
        return

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

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


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _resolve_jsonl_inputs(output_path: Path, world_size: int) -> list[tuple[int, Path]]:
    if world_size <= 1:
        return [(0, output_path)]
    return [
        (rank, _append_rank_suffix(output_path, rank)) for rank in range(world_size)
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
    count_jsonl_rows: bool,
) -> tuple[Path, int | None, int]:
    jsonl_rows: int | None = _count_jsonl_rows(jsonl_path) if count_jsonl_rows else None
    dataset: Dataset = load_dataset(
        "json",
        data_files=str(jsonl_path),
        split="train",
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

    output_path: Path = _resolve_output_path(cfg.scoring)
    count_jsonl_rows: bool = bool(
        settings.cleanup_jsonl and settings.require_count_match
    )
    jsonl_inputs = _resolve_jsonl_inputs(output_path, int(trainer.world_size))
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


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name="score_cross_encoder"
)
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)

    scoring_cfg: DictConfig = cfg.scoring
    _apply_hf_hub_timeouts(scoring_cfg)

    scoring_module: CrossEncoderScoringModule = CrossEncoderScoringModule(cfg=cfg)
    data_module_name: str = str(cfg.scoring.data_module)
    if data_module_name == "hard_negatives":
        data_module: L.LightningDataModule = ScoringHardNegativesDataModule(cfg=cfg)
    elif data_module_name == "standard":
        data_module = ScoringDataModule(cfg=cfg)
    else:
        raise ValueError(f"Unsupported scoring.data_module={data_module_name}")

    trainer_kwargs, precision = resolve_trainer_settings(scoring_cfg)

    trainer: L.Trainer = L.Trainer(
        precision=precision,
        logger=False,
        default_root_dir=cfg.log_dir,
        **trainer_kwargs,
    )
    run_predict: bool = bool(scoring_cfg.run_predict)
    if run_predict:
        trainer.predict(model=scoring_module, datamodule=data_module)
    else:
        log_if_rank_zero(
            logger,
            "Skipping prediction because scoring.run_predict is false.",
            level="warning",
        )
    _merge_rank_outputs(cfg, trainer)
    _postprocess_outputs(cfg, trainer)
    _upload_parquet_to_hub(cfg, trainer)
    log_if_rank_zero(logger, "Cross-encoder scoring complete")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
