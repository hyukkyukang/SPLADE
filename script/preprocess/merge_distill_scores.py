import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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

logger: logging.Logger = get_logger("script.merge_distill_scores", __file__)

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


def _write_output(
    cfg: DictConfig,
    *,
    datasets: list[Dataset],
    input_score_key: str,
    target_mean: float | None,
    target_std: float | None,
    source_mean: float | None,
    source_std: float | None,
) -> None:
    merge_cfg: DictConfig = cfg.merge
    normalize_cfg: DictConfig = cfg.normalization
    output_path = Path(str(merge_cfg.output_path))
    if output_path.exists() and not bool(merge_cfg.overwrite):
        raise FileExistsError(f"Output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score_key: str = str(merge_cfg.output_score_key)
    query_id_key: str = str(merge_cfg.query_id_key)
    doc_ids_key: str = str(merge_cfg.doc_ids_key)
    labels_key: str = str(merge_cfg.labels_key)
    doc_sources_key: str = str(merge_cfg.doc_sources_key)
    strict: bool = bool(merge_cfg.strict_alignment)
    flush_every: int = max(int(merge_cfg.flush_every), 1)

    normalize_scores: bool = bool(normalize_cfg.enabled)
    epsilon: float = float(normalize_cfg.epsilon)

    row_count = _resolve_row_count(datasets, strict=strict)
    max_rows: int | None = (
        None if merge_cfg.max_rows is None else int(merge_cfg.max_rows)
    )
    if max_rows is not None:
        row_count = min(row_count, max_rows)

    output_rows = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in tqdm(range(row_count), desc="merge scores", mininterval=30.0):
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
                score_key=input_score_key,
                normalize_scores=normalize_scores,
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

            qid_value: str = str(_resolve_row_value(base_row, query_id_key, True))
            doc_ids_value: list[str] = [
                str(doc_id)
                for doc_id in _resolve_row_value(base_row, doc_ids_key, True)
            ]
            labels_value = _resolve_row_value(base_row, labels_key, required=False)
            doc_sources_value = _resolve_row_value(
                base_row, doc_sources_key, required=False
            )
            output_row: dict[str, Any] = {
                "query_id": qid_value,
                "doc_ids": doc_ids_value,
                score_key: ensemble_scores,
            }
            if labels_value is not None:
                output_row["labels"] = labels_value
            if doc_sources_value is not None:
                output_row["doc_sources"] = doc_sources_value
            handle.write(json.dumps(output_row) + "\n")
            output_rows += 1
            if output_rows % flush_every == 0:
                handle.flush()
    log_if_rank_zero(logger, f"Wrote {output_rows} rows to {output_path}")


def _postprocess_output(cfg: DictConfig) -> None:
    post_cfg: DictConfig = cfg.postprocess
    if not bool(post_cfg.enabled):
        return
    output_path = Path(str(cfg.merge.output_path))
    parquet_path_value: str | None = normalize_optional_str(post_cfg.parquet_path)
    if parquet_path_value is None:
        parquet_path = output_path.with_suffix(".parquet")
    else:
        parquet_path = Path(parquet_path_value)
    dataset = load_dataset("json", data_files=str(output_path), split="train")
    tmp_path = parquet_path.with_name(f"{parquet_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    dataset.to_parquet(str(tmp_path))
    tmp_path.replace(parquet_path)
    log_if_rank_zero(logger, f"Wrote parquet output to {parquet_path}")
    if not bool(post_cfg.cleanup_jsonl):
        return
    if bool(post_cfg.require_count_match):
        json_rows = len(dataset)
        parquet_rows = len(load_dataset("parquet", data_files=str(parquet_path), split="train"))
        if json_rows != parquet_rows:
            log_if_rank_zero(
                logger,
                "Skipping JSONL cleanup due to row count mismatch.",
                level="warning",
            )
            return
    output_path.unlink()
    log_if_rank_zero(logger, f"Removed JSONL output {output_path}")


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="merge_distill_scores")
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)

    inputs_cfg = cfg.inputs
    datasets: list[Dataset] = []
    for input_cfg in inputs_cfg:
        datasets.append(_load_input_dataset(input_cfg))
    if not datasets:
        raise ValueError("No input datasets configured.")

    normalize_cfg: DictConfig = cfg.normalization
    merge_cfg: DictConfig = cfg.merge
    epsilon: float = float(normalize_cfg.epsilon)
    strict: bool = bool(merge_cfg.strict_alignment)
    max_rows: int | None = (
        None if merge_cfg.max_rows is None else int(merge_cfg.max_rows)
    )
    query_id_key: str = str(merge_cfg.query_id_key)
    doc_ids_key: str = str(merge_cfg.doc_ids_key)
    score_keys: list[str] = [str(input_cfg.score_key) for input_cfg in inputs_cfg]
    if len(set(score_keys)) != 1:
        raise ValueError(f"Input score_key values must match: {score_keys}")
    score_key: str = score_keys[0]
    input_query_id_keys: list[str] = [
        str(input_cfg.query_id_key) for input_cfg in inputs_cfg
    ]
    if len(set(input_query_id_keys)) != 1 or input_query_id_keys[0] != query_id_key:
        raise ValueError(
            "merge.query_id_key must match inputs[].query_id_key for all inputs."
        )
    input_doc_ids_keys: list[str] = [
        str(input_cfg.doc_ids_key) for input_cfg in inputs_cfg
    ]
    if len(set(input_doc_ids_keys)) != 1 or input_doc_ids_keys[0] != doc_ids_key:
        raise ValueError(
            "merge.doc_ids_key must match inputs[].doc_ids_key for all inputs."
        )

    target_mean: float | None = None
    target_std: float | None = None
    source_mean: float | None = None
    source_std: float | None = None

    if bool(cfg.rescore.enabled):
        output_mean = cfg.rescore.output_mean
        output_std = cfg.rescore.output_std
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

    _write_output(
        cfg,
        datasets=datasets,
        input_score_key=score_key,
        target_mean=target_mean,
        target_std=target_std,
        source_mean=source_mean,
        source_std=source_std,
    )
    _postprocess_output(cfg)
    log_if_rank_zero(logger, "Ensemble merge complete.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
