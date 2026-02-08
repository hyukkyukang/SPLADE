import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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

logger: logging.Logger = get_logger("script.extract_hard_negatives", __file__)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=False,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


@dataclass
class SelectionStats:
    total_rows: int = 0
    skipped_rows: int = 0
    total_selected: int = 0


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _resolve_qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("query_id") or row.get("_id") or "")


def _resolve_pos_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("pos") or row.get("pos_doc_ids") or []
    if isinstance(values, (list, tuple)):
        return _dedupe_preserve_order([str(value) for value in values])
    return _dedupe_preserve_order([str(values)])


def _resolve_neg_dict(row: dict[str, Any]) -> dict[str, list[str]]:
    value = row.get("neg") or {}
    if isinstance(value, Mapping):
        neg_dict: dict[str, list[str]] = {}
        for key, items in value.items():
            if isinstance(items, (list, tuple)):
                neg_dict[str(key)] = _dedupe_preserve_order(
                    [str(item) for item in items]
                )
            else:
                neg_dict[str(key)] = _dedupe_preserve_order([str(items)])
        return neg_dict
    if isinstance(value, (list, tuple)):
        return {"neg": _dedupe_preserve_order([str(item) for item in value])}
    return {"neg": _dedupe_preserve_order([str(value)])}


def _round_robin_select(
    source_lists: dict[str, list[str]],
    *,
    used: set[str],
    total_target: int,
    per_source_target: int,
) -> dict[str, list[str]]:
    sources: list[str] = list(source_lists.keys())
    selected: dict[str, list[str]] = {source: [] for source in sources}
    positions: dict[str, int] = {source: 0 for source in sources}
    total_selected: int = 0

    def _advance_source(source: str, *, max_count: int) -> bool:
        nonlocal total_selected
        values = source_lists[source]
        pos = positions[source]
        while pos < len(values):
            doc_id = values[pos]
            pos += 1
            if doc_id in used:
                continue
            selected[source].append(doc_id)
            used.add(doc_id)
            total_selected += 1
            positions[source] = pos
            return True
        positions[source] = pos
        return False

    # Primary pass: equal per source.
    while total_selected < total_target:
        progressed = False
        for source in sources:
            if len(selected[source]) >= per_source_target:
                continue
            if _advance_source(source, max_count=per_source_target):
                progressed = True
            if total_selected >= total_target:
                break
        if not progressed:
            break

    # Backfill if some sources are short.
    while total_selected < total_target:
        progressed = False
        for source in sources:
            if _advance_source(source, max_count=total_target):
                progressed = True
            if total_selected >= total_target:
                break
        if not progressed:
            break

    return selected


def _balanced_random_select(
    source_pools: dict[str, list[str]],
    *,
    rng: random.Random,
    used: set[str],
    total_target: int,
    per_source_target: int,
) -> dict[str, list[str]]:
    sources: list[str] = list(source_pools.keys())
    selected: dict[str, list[str]] = {source: [] for source in sources}
    total_selected: int = 0

    for source in sources:
        pool = [doc_id for doc_id in source_pools[source] if doc_id not in used]
        if not pool:
            continue
        sample_size = min(per_source_target, len(pool))
        chosen = rng.sample(pool, sample_size)
        selected[source].extend(chosen)
        used.update(chosen)
        total_selected += sample_size

    if total_selected > total_target:
        # Trim down to total_target in round-robin order.
        while total_selected > total_target:
            for source in sources:
                if not selected[source]:
                    continue
                removed = selected[source].pop()
                used.remove(removed)
                total_selected -= 1
                if total_selected <= total_target:
                    break

    if total_selected < total_target:
        remaining: dict[str, list[str]] = {}
        for source in sources:
            pool = [
                doc_id for doc_id in source_pools[source] if doc_id not in used
            ]
            remaining[source] = pool
        while total_selected < total_target:
            progressed = False
            for source in sources:
                pool = remaining[source]
                if not pool:
                    continue
                choice = rng.choice(pool)
                pool.remove(choice)
                selected[source].append(choice)
                used.add(choice)
                total_selected += 1
                progressed = True
                if total_selected >= total_target:
                    break
            if not progressed:
                break

    return selected


def _flatten_negatives(neg_by_source: dict[str, list[str]]) -> list[str]:
    output: list[str] = []
    for source in neg_by_source:
        output.extend(neg_by_source[source])
    return _dedupe_preserve_order(output)


def _build_output_row(
    *,
    qid: str,
    pos_ids: list[str],
    neg_by_source: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "qid": qid,
        "pos": pos_ids,
        "neg": neg_by_source,
    }


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


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="extract_hard_negatives")
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)

    input_cfg: DictConfig = cfg.input
    hf_name_value: str | None = normalize_optional_str(input_cfg.hf_name)
    if hf_name_value is None:
        raise ValueError("input.hf_name must be set.")
    hf_subset_value: str | None = normalize_optional_str(input_cfg.hf_subset)
    hf_split: str = str(input_cfg.hf_split)
    hf_cache_dir: str | None = normalize_optional_str(input_cfg.hf_cache_dir)
    hf_data_files: Any | None = input_cfg.hf_data_files
    dataset: Dataset = load_dataset(
        hf_name_value,
        name=hf_subset_value,
        split=hf_split,
        cache_dir=hf_cache_dir,
        data_files=hf_data_files,
    )

    output_cfg: DictConfig = cfg.output
    output_dir = Path(str(output_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_basename: str = str(output_cfg.output_basename)
    jsonl_path = output_dir / f"{output_basename}.jsonl"
    if jsonl_path.exists() and not bool(output_cfg.overwrite):
        raise FileExistsError(f"Output already exists: {jsonl_path}")

    sampling_cfg: DictConfig = cfg.sampling
    top_total: int = int(sampling_cfg.top_total)
    random_total: int = int(sampling_cfg.random_total)
    top_pool: int = int(sampling_cfg.top_pool)
    require_positive: bool = bool(sampling_cfg.require_positive)
    require_negatives: bool = bool(sampling_cfg.require_negatives)
    max_rows: int | None = (
        None if output_cfg.max_rows is None else int(output_cfg.max_rows)
    )
    flush_every: int = max(int(output_cfg.flush_every), 1)

    rng = random.Random(int(cfg.seed))
    stats = SelectionStats()

    with jsonl_path.open("w", encoding="utf-8") as handle:
        iterator = dataset
        if max_rows is not None:
            iterator = dataset.select(range(min(int(len(dataset)), max_rows)))
        for row in tqdm(iterator, desc="extract hard negatives", mininterval=30.0):
            stats.total_rows += 1
            row_dict: dict[str, Any] = dict(row)
            qid = _resolve_qid(row_dict)
            if not qid:
                stats.skipped_rows += 1
                continue
            pos_ids = _resolve_pos_ids(row_dict)
            if require_positive and not pos_ids:
                stats.skipped_rows += 1
                continue
            neg_dict_raw = _resolve_neg_dict(row_dict)
            if not neg_dict_raw:
                stats.skipped_rows += 1
                continue

            # Filter positives out of negatives and truncate pool.
            pos_set: set[str] = set(pos_ids)
            source_lists: dict[str, list[str]] = {}
            for source in sorted(neg_dict_raw.keys()):
                filtered = [
                    doc_id
                    for doc_id in neg_dict_raw[source]
                    if doc_id and doc_id not in pos_set
                ]
                source_lists[source] = filtered

            sources = list(source_lists.keys())
            if require_negatives and not sources:
                stats.skipped_rows += 1
                continue

            used: set[str] = set(pos_ids)
            num_sources = max(len(sources), 1)
            top_per_source = int(math.ceil(top_total / float(num_sources)))
            random_per_source = int(math.ceil(random_total / float(num_sources)))

            top_selected = _round_robin_select(
                source_lists,
                used=used,
                total_target=top_total,
                per_source_target=top_per_source,
            )

            pool_by_source: dict[str, list[str]] = {}
            for source in sources:
                pool_by_source[source] = source_lists[source][:top_pool]

            random_selected = _balanced_random_select(
                pool_by_source,
                rng=rng,
                used=used,
                total_target=random_total,
                per_source_target=random_per_source,
            )

            neg_by_source: dict[str, list[str]] = {}
            for source in sources:
                combined = top_selected[source] + random_selected[source]
                if combined:
                    neg_by_source[source] = combined

            if require_negatives and not _flatten_negatives(neg_by_source):
                stats.skipped_rows += 1
                continue

            output_row = _build_output_row(
                qid=qid,
                pos_ids=pos_ids,
                neg_by_source=neg_by_source,
            )
            handle.write(json.dumps(output_row) + "\n")
            stats.total_selected += 1
            if stats.total_selected % flush_every == 0:
                handle.flush()

    log_if_rank_zero(
        logger,
        f"Extracted {stats.total_selected} rows (skipped={stats.skipped_rows}).",
    )

    post_cfg: DictConfig = cfg.postprocess
    if bool(post_cfg.enabled):
        parquet_path_value: str | None = normalize_optional_str(post_cfg.parquet_path)
        parquet_path = (
            Path(parquet_path_value)
            if parquet_path_value is not None
            else jsonl_path.with_suffix(".parquet")
        )
        json_rows, parquet_rows = _convert_jsonl_to_parquet(jsonl_path, parquet_path)
        log_if_rank_zero(
            logger,
            f"Converted JSONL to Parquet: {jsonl_path} -> {parquet_path} "
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
                jsonl_path.unlink()
                log_if_rank_zero(logger, f"Removed JSONL output {jsonl_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
