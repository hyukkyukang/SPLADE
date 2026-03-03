import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
import hydra
from huggingface_hub import HfApi
from omegaconf import DictConfig
from tqdm import tqdm

from config.path import ABS_CONFIG_DIR
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.script_setup import (
    configure_script_environment,
    initialize_run,
    normalize_optional_str,
)

logger: logging.Logger = get_logger(
    "script.build_multi_teacher_scores_dataset", __file__
)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=False,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


@dataclass(frozen=True)
class InputSpec:
    name: str
    output_score_key: str
    score_key: str
    query_id_key: str
    doc_ids_key: str
    labels_key: str
    path: str | None
    format: str
    hf_name: str | None
    hf_subset: str | None
    hf_split: str
    hf_cache_dir: str | None
    hf_data_files: Any | None


def _slugify(name: str) -> str:
    lowered: str = str(name).strip().lower()
    slug: str = re.sub(r"[^a-z0-9]+", "_", lowered)
    slug = slug.strip("_")
    if not slug:
        raise ValueError(f"Unable to derive slug from input name: {name!r}")
    return slug


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


def _load_input_dataset(spec: InputSpec) -> Dataset:
    if spec.path is not None:
        path = Path(spec.path)
        resolved_format = spec.format
        if resolved_format == "auto":
            resolved_format = _infer_format(path)
        return _load_dataset_from_path(path, resolved_format)
    if spec.hf_name is None:
        raise ValueError("inputs[].path or inputs[].hf_name must be set.")
    return load_dataset(
        spec.hf_name,
        name=spec.hf_subset,
        split=spec.hf_split,
        cache_dir=spec.hf_cache_dir,
        data_files=spec.hf_data_files,
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


def _resolve_doc_ids(row: dict[str, Any], *, key: str) -> list[str]:
    values = _resolve_row_value(row, key, required=True)
    return [str(doc_id) for doc_id in values]


def _resolve_labels(
    row: dict[str, Any],
    *,
    key: str,
    required: bool,
) -> list[float] | None:
    values = _resolve_row_value(row, key, required=required)
    if values is None:
        return None
    return [float(value) for value in values]


def _validate_alignment(
    base_row: dict[str, Any],
    other_row: dict[str, Any],
    *,
    query_id_key: str,
    doc_ids_key: str,
    labels_key: str,
    check_labels: bool,
) -> None:
    base_qid: str = str(_resolve_row_value(base_row, query_id_key, required=True))
    other_qid: str = str(_resolve_row_value(other_row, query_id_key, required=True))
    if base_qid != other_qid:
        raise ValueError(f"Query id mismatch: {base_qid} vs {other_qid}")

    base_doc_ids: list[str] = _resolve_doc_ids(base_row, key=doc_ids_key)
    other_doc_ids: list[str] = _resolve_doc_ids(other_row, key=doc_ids_key)
    if base_doc_ids != other_doc_ids:
        raise ValueError(f"Doc id mismatch for query_id={base_qid}")

    if not check_labels:
        return
    base_labels: list[float] | None = _resolve_labels(
        base_row, key=labels_key, required=False
    )
    other_labels: list[float] | None = _resolve_labels(
        other_row, key=labels_key, required=False
    )
    if base_labels != other_labels:
        raise ValueError(f"Label mismatch for query_id={base_qid}")


def _extract_scores(
    row: dict[str, Any],
    *,
    score_key: str,
    expected_len: int,
    validate_finite: bool,
) -> list[float]:
    raw_scores = _resolve_row_value(row, score_key, required=True)
    scores: list[float] = [float(value) for value in raw_scores]
    if len(scores) != expected_len:
        raise ValueError(
            f"Score length mismatch for key={score_key}: "
            f"expected={expected_len}, got={len(scores)}"
        )
    if validate_finite and any(not math.isfinite(score) for score in scores):
        raise ValueError(f"Found non-finite score in column {score_key}.")
    return scores


def _normalize_scores(scores: list[float], *, epsilon: float) -> list[float]:
    if not scores:
        return []
    min_score: float = min(scores)
    max_score: float = max(scores)
    if not math.isfinite(min_score) or not math.isfinite(max_score):
        raise ValueError("Found non-finite score during normalization.")
    denom: float = max_score - min_score
    if denom <= epsilon:
        return [0.0] * len(scores)
    return [(score - min_score) / denom for score in scores]


def _build_teacher_scores(score_columns: list[list[float]]) -> list[float]:
    if not score_columns:
        return []
    score_count: int = len(score_columns[0])
    if any(len(scores) != score_count for scores in score_columns):
        raise ValueError("Score length mismatch across score columns.")
    model_count: int = len(score_columns)
    return [
        sum(scores[idx] for scores in score_columns) / float(model_count)
        for idx in range(score_count)
    ]


def _convert_jsonl_to_parquet(
    jsonl_path: Path,
    parquet_path: Path,
) -> tuple[int, int]:
    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    tmp_path: Path = parquet_path.with_name(f"{parquet_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    dataset.to_parquet(str(tmp_path))
    tmp_path.replace(parquet_path)
    parquet_dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
    return int(len(dataset)), int(len(parquet_dataset))


def _resolve_input_specs(cfg: DictConfig) -> list[InputSpec]:
    specs: list[InputSpec] = []
    for input_cfg in cfg.inputs:
        name: str = str(input_cfg.name)
        output_score_key: str | None = normalize_optional_str(input_cfg.output_score_key)
        if output_score_key is None:
            output_score_key = f"{_slugify(name)}_scores"
        specs.append(
            InputSpec(
                name=name,
                output_score_key=output_score_key,
                score_key=str(input_cfg.score_key),
                query_id_key=str(input_cfg.query_id_key),
                doc_ids_key=str(input_cfg.doc_ids_key),
                labels_key=str(input_cfg.labels_key),
                path=normalize_optional_str(input_cfg.path),
                format=str(input_cfg.format).lower(),
                hf_name=normalize_optional_str(input_cfg.hf_name),
                hf_subset=normalize_optional_str(input_cfg.hf_subset),
                hf_split=str(input_cfg.hf_split),
                hf_cache_dir=normalize_optional_str(input_cfg.hf_cache_dir),
                hf_data_files=input_cfg.hf_data_files,
            )
        )
    if not specs:
        raise ValueError("inputs list is empty.")
    score_keys: list[str] = [spec.output_score_key for spec in specs]
    if len(set(score_keys)) != len(score_keys):
        raise ValueError(f"Duplicate output score keys in inputs: {score_keys}")
    return specs


def _resolve_hf_token(upload_cfg: DictConfig) -> str:
    token_value: str | None = normalize_optional_str(upload_cfg.hf_token)
    if token_value is not None:
        return token_value
    env_token: str | None = normalize_optional_str(
        os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    if env_token is None:
        raise ValueError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN must be set to upload.")
    return env_token


def _upload_to_hub(cfg: DictConfig, *, parquet_path: Path) -> None:
    upload_cfg: DictConfig = cfg.upload
    if not bool(upload_cfg.enabled):
        return

    token: str = _resolve_hf_token(upload_cfg)
    repo_id: str = str(upload_cfg.repo_id)
    split_name: str = str(upload_cfg.split)
    private: bool = bool(upload_cfg.private)
    commit_message: str = str(upload_cfg.commit_message)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    data_files: dict[str, str] = {split_name: str(parquet_path)}
    dataset_dict: DatasetDict = load_dataset("parquet", data_files=data_files)
    dataset_dict.push_to_hub(
        repo_id,
        token=token,
        private=private,
        commit_message=commit_message,
    )
    log_if_rank_zero(logger, f"Uploaded dataset to Hugging Face Hub: {repo_id}")


@hydra.main(
    version_base=None,
    config_path=ABS_CONFIG_DIR,
    config_name="build_multi_teacher_scores_dataset",
)
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)

    input_specs: list[InputSpec] = _resolve_input_specs(cfg)
    datasets: list[Dataset] = []
    for spec in input_specs:
        log_if_rank_zero(logger, f"Loading input={spec.name}")
        datasets.append(_load_input_dataset(spec))

    merge_cfg: DictConfig = cfg.merge
    strict_alignment: bool = bool(merge_cfg.strict_alignment)
    strict_labels: bool = bool(merge_cfg.strict_labels)
    require_labels: bool = bool(merge_cfg.require_labels)
    validate_finite_scores: bool = bool(merge_cfg.validate_finite_scores)
    store_normalized_columns: bool = bool(merge_cfg.store_normalized_score_columns)
    normalized_suffix: str = str(merge_cfg.normalized_score_suffix)

    normalization_cfg: DictConfig | None = cfg.get("normalization")
    normalization_enabled: bool = (
        bool(normalization_cfg.enabled) if normalization_cfg is not None else False
    )
    normalization_epsilon: float = (
        float(normalization_cfg.epsilon) if normalization_cfg is not None else 1e-6
    )
    log_if_rank_zero(
        logger,
        "teacher_scores aggregation="
        f"{'mean(min-max normalized)' if normalization_enabled else 'mean(raw)'}",
    )

    row_count: int = _resolve_row_count(datasets, strict=strict_alignment)
    max_rows: int | None = (
        None if merge_cfg.max_rows is None else int(merge_cfg.max_rows)
    )
    if max_rows is not None:
        row_count = min(row_count, max_rows)

    output_path: Path = Path(str(merge_cfg.output_path))
    if output_path.exists() and not bool(merge_cfg.overwrite):
        raise FileExistsError(f"Output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flush_every: int = max(int(merge_cfg.flush_every), 1)
    teacher_score_key: str = str(merge_cfg.teacher_score_key)
    output_query_id_key: str = str(merge_cfg.output_query_id_key)
    output_doc_ids_key: str = str(merge_cfg.output_doc_ids_key)
    output_labels_key: str = str(merge_cfg.output_labels_key)

    output_rows: int = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in tqdm(range(row_count), desc="build multi-teacher dataset", mininterval=30.0):
            rows: list[dict[str, Any]] = [dict(dataset[int(idx)]) for dataset in datasets]
            base_row: dict[str, Any] = rows[0]
            base_spec: InputSpec = input_specs[0]
            if strict_alignment:
                for other_row, other_spec in zip(rows[1:], input_specs[1:]):
                    # Input schema must be consistent for strict row alignment.
                    if (
                        other_spec.query_id_key != base_spec.query_id_key
                        or other_spec.doc_ids_key != base_spec.doc_ids_key
                    ):
                        raise ValueError(
                            "All inputs must use matching query/doc id keys when "
                            "strict_alignment is enabled."
                        )
                    if strict_labels and other_spec.labels_key != base_spec.labels_key:
                        raise ValueError(
                            "All inputs must use matching label keys when "
                            "merge.strict_labels is enabled."
                        )
                    _validate_alignment(
                        base_row,
                        other_row,
                        query_id_key=base_spec.query_id_key,
                        doc_ids_key=base_spec.doc_ids_key,
                        labels_key=base_spec.labels_key,
                        check_labels=strict_labels,
                    )

            query_id: str = str(
                _resolve_row_value(base_row, base_spec.query_id_key, required=True)
            )
            doc_ids: list[str] = _resolve_doc_ids(base_row, key=base_spec.doc_ids_key)
            labels: list[float] | None = _resolve_labels(
                base_row,
                key=base_spec.labels_key,
                required=require_labels,
            )
            if labels is not None and len(labels) != len(doc_ids):
                raise ValueError(f"Label length mismatch for query_id={query_id}")

            output_row: dict[str, Any] = {
                output_query_id_key: query_id,
                output_doc_ids_key: doc_ids,
            }
            if labels is not None:
                output_row[output_labels_key] = labels

            score_columns: list[list[float]] = []
            for row, spec in zip(rows, input_specs):
                raw_scores: list[float] = _extract_scores(
                    row,
                    score_key=spec.score_key,
                    expected_len=len(doc_ids),
                    validate_finite=validate_finite_scores,
                )
                output_row[spec.output_score_key] = raw_scores

                teacher_scores_input: list[float] = raw_scores
                if normalization_enabled:
                    normalized_scores: list[float] = _normalize_scores(
                        raw_scores, epsilon=normalization_epsilon
                    )
                    teacher_scores_input = normalized_scores
                    if store_normalized_columns:
                        output_row[f"{spec.output_score_key}{normalized_suffix}"] = (
                            normalized_scores
                        )
                score_columns.append(teacher_scores_input)

            teacher_scores: list[float] = _build_teacher_scores(score_columns)
            output_row[teacher_score_key] = teacher_scores
            handle.write(json.dumps(output_row) + "\n")
            output_rows += 1
            if output_rows % flush_every == 0:
                handle.flush()

    log_if_rank_zero(logger, f"Wrote {output_rows} rows to {output_path}")

    post_cfg: DictConfig = cfg.postprocess
    parquet_path_value: str | None = normalize_optional_str(post_cfg.parquet_path)
    parquet_path: Path = (
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

    if bool(cfg.upload.enabled) and not bool(post_cfg.enabled):
        raise ValueError("postprocess.enabled must be true when upload.enabled=true.")
    if bool(cfg.upload.enabled) and not parquet_path.exists():
        raise FileNotFoundError(f"Expected parquet file for upload: {parquet_path}")
    _upload_to_hub(cfg, parquet_path=parquet_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
