import logging
import math
from typing import Any

from datasets import Dataset, load_dataset
import hydra
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.data.dataset.utils import normalize_optional_str
from src.data.registry import build_dataset
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.script_setup import configure_script_environment

logger: logging.Logger = get_logger("script.check_triplet_scores", __file__)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=False,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


def _apply_sample_window(
    dataset: Dataset, *, skip_samples: int, max_samples: int | None
) -> Dataset:
    if skip_samples <= 0 and max_samples is None:
        return dataset
    dataset_length: int = int(len(dataset))
    start_index: int = min(skip_samples, dataset_length)
    end_index: int = dataset_length
    if max_samples is not None:
        end_index = min(start_index + int(max_samples), dataset_length)
    indices: range = range(start_index, end_index)
    return dataset.select(indices)


def _load_base_triplets(cfg: DictConfig) -> Dataset:
    hf_name: str = str(cfg.hf_name)
    hf_subset: str | None = normalize_optional_str(cfg.hf_subset)
    hf_split: str = str(cfg.split)
    hf_cache_dir: str | None = normalize_optional_str(cfg.hf_cache_dir)
    hf_data_files: Any | None = cfg.hf_data_files
    dataset: Dataset = load_dataset(
        hf_name,
        name=hf_subset,
        split=hf_split,
        cache_dir=hf_cache_dir,
        data_files=hf_data_files,
    )
    skip_samples: int = int(cfg.hf_skip_samples)
    max_samples: int | None = (
        None if cfg.hf_max_samples is None else int(cfg.hf_max_samples)
    )
    return _apply_sample_window(
        dataset, skip_samples=skip_samples, max_samples=max_samples
    )


def _validate_row(row: dict[str, Any]) -> None:
    doc_ids: list[Any] = list(row.get("doc_ids") or [])
    labels: list[Any] = list(row.get("labels") or [])
    scores: list[Any] = list(row.get("scores") or [])
    if not doc_ids:
        raise ValueError("Row is missing doc_ids.")
    if len(doc_ids) != len(labels) or len(doc_ids) != len(scores):
        raise ValueError(
            "Row lengths mismatch: "
            f"doc_ids={len(doc_ids)}, labels={len(labels)}, scores={len(scores)}."
        )
    if float(labels[0]) <= 0:
        raise ValueError("First label is not positive.")
    for score in scores:
        if not math.isfinite(float(score)):
            raise ValueError("Found non-finite score.")


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="check_triplet_scores")
def main(cfg: DictConfig) -> None:
    base_dataset: Dataset = _load_base_triplets(cfg.dataset)
    base_count: int = int(len(base_dataset))
    log_if_rank_zero(logger, f"Base triplet rows: {base_count}")

    joined_dataset: Dataset = build_dataset(cfg.dataset).meta_dataset
    joined_count: int = int(len(joined_dataset))
    log_if_rank_zero(logger, f"Joined rows: {joined_count}")

    strict: bool = bool(cfg.sanity.strict)
    if joined_count != base_count:
        message = (
            "Joined dataset size does not match base triplets: "
            f"{joined_count} vs {base_count}."
        )
        if strict:
            raise ValueError(message)
        log_if_rank_zero(logger, message, level="warning")

    max_rows: int | None = (
        None if cfg.sanity.max_rows is None else int(cfg.sanity.max_rows)
    )
    log_every: int = int(cfg.sanity.log_every)
    sample_rows: int = int(cfg.sanity.sample_rows)

    inspect_rows: int = joined_count
    if max_rows is not None:
        inspect_rows = min(inspect_rows, max_rows)

    for idx in range(inspect_rows):
        row: dict[str, Any] = dict(joined_dataset[int(idx)])
        _validate_row(row)
        if log_every > 0 and (idx + 1) % log_every == 0:
            log_if_rank_zero(logger, f"Validated {idx + 1} rows.")
        if sample_rows > 0 and idx < sample_rows:
            qid: str = str(row.get("query_id") or "")
            log_if_rank_zero(logger, f"Sample {idx + 1} qid={qid}")

    log_if_rank_zero(logger, "Sanity check complete.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
