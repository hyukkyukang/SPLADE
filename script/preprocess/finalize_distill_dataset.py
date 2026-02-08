import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from config.path import ABS_CONFIG_DIR
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.script_setup import (
    configure_script_environment,
    initialize_run,
    normalize_optional_str,
)

logger: logging.Logger = get_logger("script.finalize_distill_dataset", __file__)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=False,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / float(self.count)
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def finalize(self) -> tuple[float, float]:
        if self.count == 0:
            return 0.0, 0.0
        variance = self.m2 / float(self.count)
        return self.mean, math.sqrt(max(variance, 0.0))


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return "json"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(f"Unable to infer format from {path.as_posix()}")


def _load_dataset_from_path(path: Path, format_value: str) -> Dataset:
    if format_value == "json":
        return load_dataset("json", data_files=str(path), split="train")
    if format_value == "parquet":
        return load_dataset("parquet", data_files=str(path), split="train")
    raise ValueError(f"Unsupported input format: {format_value}")


def _load_input_dataset(cfg: DictConfig) -> Dataset:
    path_value: str | None = normalize_optional_str(cfg.path)
    format_value: str = str(cfg.format).lower()
    if path_value is not None:
        path = Path(path_value)
        resolved_format = format_value
        if resolved_format == "auto":
            resolved_format = _infer_format(path)
        return _load_dataset_from_path(path, resolved_format)
    hf_name_value: str | None = normalize_optional_str(cfg.hf_name)
    if hf_name_value is None:
        raise ValueError("inputs[].path or inputs[].hf_name must be set.")
    hf_subset_value: str | None = normalize_optional_str(cfg.hf_subset)
    hf_split: str = str(cfg.hf_split)
    hf_cache_dir: str | None = normalize_optional_str(cfg.hf_cache_dir)
    hf_data_files: Any | None = cfg.hf_data_files
    return load_dataset(
        hf_name_value,
        name=hf_subset_value,
        split=hf_split,
        cache_dir=hf_cache_dir,
        data_files=hf_data_files,
    )


def _resolve_row_count(datasets: list[Dataset], *, strict: bool) -> int:
    lengths = [int(len(dataset)) for dataset in datasets]
    if strict and len(set(lengths)) != 1:
        raise ValueError(f"Input dataset lengths mismatch: {lengths}")
    return min(lengths) if lengths else 0


def _resolve_row_value(row: dict[str, Any], key: str, *, required: bool) -> Any:
    if key in row:
        return row[key]
    if required:
        raise KeyError(f"Missing required key {key} in row.")
    return None


def _normalize_scores(scores: list[float], *, epsilon: float) -> list[float]:
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    if not math.isfinite(min_val) or not math.isfinite(max_val):
        raise ValueError("Found non-finite scores during normalization.")
    denom = max_val - min_val
    if denom <= epsilon:
        return [0.0] * len(scores)
    return [(score - min_val) / denom for score in scores]


def _extract_scores(
    row: dict[str, Any], *, score_key: str, epsilon: float, normalize: bool
) -> list[float]:
    raw_scores: list[Any] = list(_resolve_row_value(row, score_key, required=True))
    scores = [float(value) for value in raw_scores]
    if not normalize:
        return scores
    return _normalize_scores(scores, epsilon=epsilon)


def _validate_alignment(
    base_row: dict[str, Any],
    other_row: dict[str, Any],
    *,
    query_id_key: str,
    doc_ids_key: str,
) -> None:
    base_qid: str = str(_resolve_row_value(base_row, query_id_key, required=True))
    other_qid: str = str(_resolve_row_value(other_row, query_id_key, required=True))
    if base_qid != other_qid:
        raise ValueError(f"Query id mismatch: {base_qid} vs {other_qid}")
    base_doc_ids: list[str] = [
        str(doc_id)
        for doc_id in _resolve_row_value(base_row, doc_ids_key, required=True)
    ]
    other_doc_ids: list[str] = [
        str(doc_id)
        for doc_id in _resolve_row_value(other_row, doc_ids_key, required=True)
    ]
    if base_doc_ids != other_doc_ids:
        raise ValueError(f"Doc id mismatch for query_id={base_qid}")


def _build_ensemble_scores(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    normalize_scores: bool,
    epsilon: float,
) -> list[float]:
    normalized_scores: list[list[float]] = []
    for row in rows:
        normalized_scores.append(
            _extract_scores(
                row,
                score_key=score_key,
                epsilon=epsilon,
                normalize=normalize_scores,
            )
        )
    if not normalized_scores:
        return []
    score_count = len(normalized_scores[0])
    if any(len(scores) != score_count for scores in normalized_scores):
        raise ValueError("Score length mismatch across inputs.")
    model_count = len(normalized_scores)
    return [
        sum(scores[idx] for scores in normalized_scores) / float(model_count)
        for idx in range(score_count)
    ]


def _apply_affine_rescore(
    scores: list[float],
    *,
    source_mean: float,
    source_std: float,
    target_mean: float,
    target_std: float,
    epsilon: float,
) -> list[float]:
    denom = source_std if source_std > epsilon else epsilon
    return [((score - source_mean) / denom) * target_std + target_mean for score in scores]


def _compute_reference_stats(cfg: DictConfig, *, epsilon: float) -> tuple[float, float]:
    reference_cfg: DictConfig = cfg.rescore.reference
    reference_dataset: Dataset = _load_input_dataset(reference_cfg)
    score_key: str = str(reference_cfg.score_key)
    stats = RunningStats()
    for row in tqdm(reference_dataset, desc="reference stats", mininterval=30.0):
        scores_raw = _resolve_row_value(dict(row), score_key, required=True)
        for value in scores_raw:
            score = float(value)
            if not math.isfinite(score):
                raise ValueError("Non-finite score in reference dataset.")
            stats.update(score)
    mean, std = stats.finalize()
    if std <= epsilon:
        log_if_rank_zero(
            logger,
            "Reference scores have near-zero std; rescore may be unstable.",
            level="warning",
        )
    return mean, std


def _compute_ensemble_stats(
    datasets: list[Dataset],
    *,
    score_key: str,
    query_id_key: str,
    doc_ids_key: str,
    normalize_scores: bool,
    epsilon: float,
    strict: bool,
    max_rows: int | None,
) -> tuple[float, float]:
    row_count = _resolve_row_count(datasets, strict=strict)
    if max_rows is not None:
        row_count = min(row_count, int(max_rows))
    stats = RunningStats()
    for idx in tqdm(range(row_count), desc="ensemble stats", mininterval=30.0):
        rows = [dict(dataset[int(idx)]) for dataset in datasets]
        base_row = rows[0]
        if strict:
            for other_row in rows[1:]:
                _validate_alignment(
                    base_row,
                    other_row,
                    query_id_key=query_id_key,
                    doc_ids_key=doc_ids_key,
                )
        ensemble_scores = _build_ensemble_scores(
            rows,
            score_key=score_key,
            normalize_scores=normalize_scores,
            epsilon=epsilon,
        )
        for score in ensemble_scores:
            stats.update(float(score))
    return stats.finalize()


def _select_top_negatives(
    *,
    labels: list[float],
    scores: list[float],
    trim_negatives: int | None,
) -> list[int]:
    pos_indices = [idx for idx, label in enumerate(labels) if float(label) > 0.0]
    neg_indices = [idx for idx, label in enumerate(labels) if float(label) <= 0.0]
    if trim_negatives is None:
        return pos_indices + neg_indices
    sorted_negs = sorted(neg_indices, key=lambda idx: scores[idx], reverse=True)
    selected_negs = sorted_negs[:trim_negatives]
    return pos_indices + selected_negs


def _convert_jsonl_to_parquet(
    jsonl_path: Path, parquet_path: Path
) -> tuple[int, int]:
    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    tmp_path = parquet_path.with_name(f"{parquet_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    dataset.to_parquet(str(tmp_path))
    tmp_path.replace(parquet_path)
    return len(dataset), len(dataset)


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name="finalize_distill_dataset"
)
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)

    inputs_cfg = cfg.inputs
    datasets: list[Dataset] = []
    for input_cfg in inputs_cfg:
        datasets.append(_load_input_dataset(input_cfg))
    if not datasets:
        raise ValueError("No input datasets configured.")

    merge_cfg: DictConfig = cfg.merge
    normalize_cfg: DictConfig = cfg.normalization
    rescore_cfg: DictConfig = cfg.rescore
    trim_cfg: DictConfig = cfg.trim

    epsilon: float = float(normalize_cfg.epsilon)
    strict: bool = bool(merge_cfg.strict_alignment)
    max_rows: int | None = (
        None if merge_cfg.max_rows is None else int(merge_cfg.max_rows)
    )
    score_keys: list[str] = [str(input_cfg.score_key) for input_cfg in inputs_cfg]
    if len(set(score_keys)) != 1:
        raise ValueError(f"Input score_key values must match: {score_keys}")
    score_key: str = score_keys[0]

    query_id_key: str = str(merge_cfg.query_id_key)
    doc_ids_key: str = str(merge_cfg.doc_ids_key)
    labels_key: str = str(merge_cfg.labels_key)

    target_mean: float | None = None
    target_std: float | None = None
    source_mean: float | None = None
    source_std: float | None = None
    if bool(rescore_cfg.enabled):
        output_mean = rescore_cfg.output_mean
        output_std = rescore_cfg.output_std
        if output_mean is not None and output_std is not None:
            target_mean = float(output_mean)
            target_std = float(output_std)
        else:
            target_mean, target_std = _compute_reference_stats(cfg, epsilon=epsilon)
        source_mean, source_std = _compute_ensemble_stats(
            datasets,
            score_key=score_key,
            query_id_key=query_id_key,
            doc_ids_key=doc_ids_key,
            normalize_scores=bool(normalize_cfg.enabled),
            epsilon=epsilon,
            strict=strict,
            max_rows=max_rows,
        )
        log_if_rank_zero(
            logger,
            f"Rescore stats: source_mean={source_mean:.6f}, "
            f"source_std={source_std:.6f}, target_mean={target_mean:.6f}, "
            f"target_std={target_std:.6f}",
        )

    output_path = Path(str(merge_cfg.output_path))
    if output_path.exists() and not bool(merge_cfg.overwrite):
        raise FileExistsError(f"Output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flush_every: int = max(int(merge_cfg.flush_every), 1)

    row_count = _resolve_row_count(datasets, strict=strict)
    if max_rows is not None:
        row_count = min(row_count, max_rows)

    trim_negatives: int | None = (
        None if trim_cfg.negatives is None else int(trim_cfg.negatives)
    )
    require_positive: bool = bool(trim_cfg.require_positive)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in tqdm(range(row_count), desc="finalize dataset", mininterval=30.0):
            rows = [dict(dataset[int(idx)]) for dataset in datasets]
            base_row = rows[0]
            if strict:
                for other_row in rows[1:]:
                    _validate_alignment(
                        base_row,
                        other_row,
                        query_id_key=query_id_key,
                        doc_ids_key=doc_ids_key,
                    )
            ensemble_scores = _build_ensemble_scores(
                rows,
                score_key=score_key,
                normalize_scores=bool(normalize_cfg.enabled),
                epsilon=epsilon,
            )
            if (
                target_mean is not None
                and target_std is not None
                and source_mean is not None
                and source_std is not None
            ):
                ensemble_scores = _apply_affine_rescore(
                    ensemble_scores,
                    source_mean=source_mean,
                    source_std=source_std,
                    target_mean=target_mean,
                    target_std=target_std,
                    epsilon=epsilon,
                )

            labels = _resolve_row_value(base_row, labels_key, required=True)
            label_list: list[float] = [float(value) for value in labels]
            if require_positive and not any(label > 0.0 for label in label_list):
                raise ValueError("Missing positive label in row.")

            doc_ids: list[str] = [
                str(doc_id)
                for doc_id in _resolve_row_value(base_row, doc_ids_key, required=True)
            ]
            if len(doc_ids) != len(label_list) or len(doc_ids) != len(ensemble_scores):
                raise ValueError("doc_ids/labels/scores length mismatch.")

            selected_indices = _select_top_negatives(
                labels=label_list,
                scores=ensemble_scores,
                trim_negatives=trim_negatives,
            )
            selected_doc_ids = [doc_ids[idx] for idx in selected_indices]
            selected_labels = [label_list[idx] for idx in selected_indices]
            selected_scores = [ensemble_scores[idx] for idx in selected_indices]

            output_row = {
                "query_id": str(_resolve_row_value(base_row, query_id_key, required=True)),
                "doc_ids": selected_doc_ids,
                "labels": selected_labels,
                merge_cfg.output_score_key: selected_scores,
            }
            handle.write(json.dumps(output_row) + "\n")
            written += 1
            if written % flush_every == 0:
                handle.flush()

    log_if_rank_zero(logger, f"Wrote {written} rows to {output_path}")

    post_cfg: DictConfig = cfg.postprocess
    parquet_path_value: str | None = normalize_optional_str(post_cfg.parquet_path)
    parquet_path = (
        Path(parquet_path_value)
        if parquet_path_value is not None
        else output_path.with_suffix(".parquet")
    )
    if bool(post_cfg.enabled):
        json_rows, parquet_rows = _convert_jsonl_to_parquet(output_path, parquet_path)
        log_if_rank_zero(
            logger,
            f"Converted JSONL to Parquet: {output_path} -> {parquet_path} "
            f"({parquet_rows} rows).",
        )
        if bool(post_cfg.cleanup_jsonl):
            if bool(post_cfg.require_count_match) and json_rows != parquet_rows:
                log_if_rank_zero(
                    logger,
                    "Skipping JSONL cleanup due to row count mismatch.",
                    level="warning",
                )
            else:
                output_path.unlink()
                log_if_rank_zero(logger, f"Removed JSONL output {output_path}")

    if bool(cfg.upload_to_hub):
        repo_id_value: str = str(cfg.hf_repo_id)
        token_value: str | None = normalize_optional_str(cfg.hf_token)
        if token_value is None:
            token_value = normalize_optional_str(
                os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            )
        if token_value is None:
            raise ValueError(
                "HF_TOKEN or HUGGINGFACE_HUB_TOKEN must be set to upload."
            )
        data_files = {"train": [str(parquet_path)]}
        dataset = load_dataset("parquet", data_files=data_files, split="train")
        dataset.push_to_hub(repo_id_value, token=token_value)
        log_if_rank_zero(logger, f"Upload complete: {repo_id_value}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
