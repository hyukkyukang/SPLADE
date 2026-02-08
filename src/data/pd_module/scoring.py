import logging
import os
import re
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, field
import tqdm
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DownloadConfig, load_dataset
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset as TorchDataset

from src.data.utils import id_to_idx, resolve_dataset_column
from src.utils.logging import log_if_rank_zero
from src.utils.script_setup import normalize_optional_str

logger: logging.Logger = logging.getLogger("ScoringPDModule")


@dataclass(frozen=True)
class PositivesSettings:
    """Settings for loading positive document ids."""

    enabled: bool
    cache_path: str | None
    hf_name: str | None
    hf_subset: str | None
    hf_split: str
    hf_cache_dir: str | None
    hf_data_files: Any | None


@dataclass(frozen=True)
class MiningInputSettings:
    """Settings for resolving mined hard-negative inputs."""

    enabled: bool
    output_dir_base: str
    run_filename: str
    dataset_name: str | None
    model_name: str | None
    prefer_parquet: bool


@dataclass(frozen=True)
class ScoringItem:
    """Payload for cross-encoder scoring."""

    row: dict[str, Any]
    qid: str
    doc_ids: list[str]
    labels: list[float] | None
    doc_sources: list[str] | None
    query_text: str
    doc_texts: list[str]


def _parse_positives_settings(cfg: DictConfig | None) -> PositivesSettings | None:
    if cfg is None:
        return None
    enabled: bool = bool(cfg.enabled)
    cache_path: str | None = normalize_optional_str(cfg.cache_path)
    hf_name: str | None = normalize_optional_str(cfg.hf_name)
    hf_subset: str | None = normalize_optional_str(cfg.hf_subset)
    hf_split: str = str(cfg.hf_split)
    hf_cache_dir: str | None = normalize_optional_str(cfg.hf_cache_dir)
    hf_data_files: Any | None = cfg.hf_data_files
    return PositivesSettings(
        enabled=enabled,
        cache_path=cache_path,
        hf_name=hf_name,
        hf_subset=hf_subset,
        hf_split=hf_split,
        hf_cache_dir=hf_cache_dir,
        hf_data_files=hf_data_files,
    )


def _parse_mining_settings(mining_cfg: DictConfig) -> MiningInputSettings:
    return MiningInputSettings(
        enabled=bool(mining_cfg.enabled),
        output_dir_base=str(mining_cfg.output_dir_base),
        run_filename=str(mining_cfg.run_filename),
        dataset_name=normalize_optional_str(mining_cfg.dataset_name),
        model_name=normalize_optional_str(mining_cfg.model_name),
        prefer_parquet=bool(mining_cfg.prefer_parquet),
    )


def _slugify_component(value: Any, *, fallback: str) -> str:
    normalized: str = normalize_optional_str(value) or fallback
    slug: str = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower()).strip("_")
    return slug or fallback


def _resolve_dataset_name(dataset_cfg: DictConfig, override: str | None) -> str:
    if normalize_optional_str(override) is not None:
        return _slugify_component(override, fallback="dataset")
    for value in (
        dataset_cfg.name,
        dataset_cfg.beir_dataset,
        dataset_cfg.hf_name,
    ):
        if normalize_optional_str(value) is not None:
            return _slugify_component(value, fallback="dataset")
    return "dataset"


def _resolve_model_name(model_name: str | None) -> str:
    return _slugify_component(model_name, fallback="model")


def _resolve_rank_shards(run_path: Path, suffix: str) -> list[Path]:
    pattern = f"{run_path.stem}.rank*.{suffix}"
    return sorted(run_path.parent.glob(pattern))


def _labels_to_sources(labels: list[float], *, neg_source: str) -> list[str]:
    return ["pos" if label > 0 else neg_source for label in labels]


def _resolve_mined_input_files(
    settings: MiningInputSettings, dataset_cfg: DictConfig
) -> tuple[str | None, Any | None]:
    if not settings.enabled:
        return None, None
    dataset_name: str = _resolve_dataset_name(dataset_cfg, settings.dataset_name)
    model_name: str = _resolve_model_name(settings.model_name)
    output_dir = Path(settings.output_dir_base) / dataset_name / model_name
    run_path = output_dir / settings.run_filename

    jsonl_paths: list[Path] = []
    parquet_paths: list[Path] = []
    if run_path.exists():
        if run_path.suffix == ".parquet":
            parquet_paths = [run_path]
        else:
            jsonl_paths = [run_path]

    if not parquet_paths:
        parquet_path = run_path.with_suffix(".parquet")
        if parquet_path.exists():
            parquet_paths = [parquet_path]
        else:
            parquet_paths = _resolve_rank_shards(run_path, "parquet")

    if not jsonl_paths:
        jsonl_path = run_path.with_suffix(".jsonl")
        if jsonl_path.exists():
            jsonl_paths = [jsonl_path]
        else:
            jsonl_paths = _resolve_rank_shards(run_path, "jsonl")

    resolved_format: str | None = None
    resolved_paths: list[Path] = []
    if settings.prefer_parquet and parquet_paths:
        resolved_format = "parquet"
        resolved_paths = parquet_paths
    elif not settings.prefer_parquet and jsonl_paths:
        resolved_format = "json"
        resolved_paths = jsonl_paths
    elif jsonl_paths:
        resolved_format = "json"
        resolved_paths = jsonl_paths
    elif parquet_paths:
        resolved_format = "parquet"
        resolved_paths = parquet_paths

    if resolved_format is None or not resolved_paths:
        raise FileNotFoundError(
            "Unable to resolve mined inputs. "
            f"Checked {run_path} and rank shards under {output_dir}."
        )

    resolved_files: list[str] = [path.as_posix() for path in resolved_paths]
    files_block = "\n".join(f"  - {path}" for path in resolved_files)
    log_if_rank_zero(
        logger,
        "Resolved mined inputs:\n"
        f"  format: {resolved_format}\n"
        f"  count: {len(resolved_files)}\n"
        "  files:\n"
        f"{files_block}",
    )
    return resolved_format, resolved_files


def _normalize_data_files(data_files: Any | None) -> Any | None:
    if data_files is None:
        return None
    if isinstance(data_files, (str, list, tuple)):
        return data_files
    if isinstance(data_files, Mapping):
        return dict(data_files)
    raise TypeError("hf_data_files must be a path, list/tuple of paths, or a mapping.")


def _resolve_local_files_only(scoring_cfg: DictConfig, *, is_primary: bool) -> bool:
    if not is_primary and bool(scoring_cfg.hf_local_files_only_non_primary):
        return True
    return bool(scoring_cfg.hf_local_files_only)


def _resolve_download_config(
    scoring_cfg: DictConfig, *, local_files_only: bool
) -> DownloadConfig:
    max_retries: int = max(int(scoring_cfg.hf_max_retries), 1)
    return DownloadConfig(
        local_files_only=local_files_only,
        max_retries=max_retries,
    )


def _apply_hf_offline_mode(local_files_only: bool) -> None:
    value = "1" if local_files_only else "0"
    os.environ["HF_DATASETS_OFFLINE"] = value
    os.environ["HF_HUB_OFFLINE"] = value
    os.environ["TRANSFORMERS_OFFLINE"] = value


def _load_dataset_from_config(
    cfg: DictConfig,
    *,
    hf_name: str | None = None,
    data_files: Any | None = None,
    download_config: DownloadConfig | None = None,
) -> Dataset:
    """Load a dataset based on the dataset config block."""
    resolved_hf_name: str = str(hf_name if hf_name is not None else cfg.hf_name)
    hf_subset: str | None = normalize_optional_str(cfg.hf_subset)
    hf_split: str = str(cfg.hf_split)
    hf_cache_dir: str | None = cfg.hf_cache_dir
    resolved_data_files: Any | None = _normalize_data_files(
        data_files if data_files is not None else cfg.hf_data_files
    )
    dataset: Dataset = load_dataset(
        resolved_hf_name,
        name=hf_subset,
        split=hf_split,
        cache_dir=hf_cache_dir,
        streaming=False,
        data_files=resolved_data_files,
        download_config=download_config,
    )
    return dataset


def _resolve_column(column_names: Iterable[str], candidates: Sequence[str]) -> str:
    """Pick the first matching column name."""
    for name in candidates:
        if name in column_names:
            return name
    raise ValueError(f"Unable to resolve column from {list(column_names)}")


def _resolve_configured_column(
    cfg: DictConfig,
    field_name: str,
    column_names: Iterable[str],
    candidates: Sequence[str],
) -> str:
    configured: str | None = normalize_optional_str(cfg.get(field_name))
    if configured:
        if configured in column_names:
            return configured
        log_if_rank_zero(
            logger,
            f"Configured {field_name}={configured} not found; falling back.",
            level="warning",
        )
    return _resolve_column(column_names, candidates)


def _build_id_to_idx_map(dataset: Dataset, id_column: str) -> dict[str, int]:
    """Build id->index mapping for dataset columns."""
    column = resolve_dataset_column(dataset, id_column)
    mapping: dict[str, int] = id_to_idx(column, desc="id_to_idx", enable_tqdm=False)
    return mapping


def _column_text_value(column: pa.Array | pa.ChunkedArray, idx: int) -> str:
    value: Any = column[idx]
    if isinstance(value, pa.Scalar):
        value = value.as_py()
    return "" if value is None else str(value)


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _collect_qids(score_dataset: Dataset, max_rows: int | None) -> set[str]:
    qids: set[str] = set()
    row_count: int = 0
    for row in tqdm.tqdm(score_dataset):
        if max_rows is not None and row_count >= max_rows:
            break
        qid: str = str(row.get("query_id") or row.get("qid") or row.get("_id") or "")
        if not qid:
            continue
        qids.add(qid)
        row_count += 1
    return qids


@dataclass
class _DocTextLookup:
    corpus_id_to_idx: dict[str, int]
    corpus_text_column: pa.Array | pa.ChunkedArray
    cache_size: int
    _cache: OrderedDict[str, str] = field(default_factory=OrderedDict, init=False)

    def __call__(self, doc_id: str) -> str:
        if not doc_id:
            return ""
        if self.cache_size <= 0:
            return self._lookup_text(doc_id)
        cached = self._cache.get(doc_id)
        if cached is not None:
            self._cache.move_to_end(doc_id)
            return cached
        value = self._lookup_text(doc_id)
        self._cache[doc_id] = value
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    def lookup_many(self, doc_ids: Iterable[str]) -> list[str]:
        return [self(doc_id) for doc_id in doc_ids]

    def _lookup_text(self, doc_id: str) -> str:
        doc_idx: int = int(self.corpus_id_to_idx.get(doc_id, -1))
        if doc_idx < 0:
            return ""
        return _column_text_value(self.corpus_text_column, doc_idx)


def _build_doc_text_lookup(
    corpus_id_to_idx: dict[str, int],
    corpus_text_column: pa.Array | pa.ChunkedArray,
    *,
    cache_size: int,
) -> _DocTextLookup:
    return _DocTextLookup(
        corpus_id_to_idx=corpus_id_to_idx,
        corpus_text_column=corpus_text_column,
        cache_size=cache_size,
    )


def _build_doc_text_lookup_for_corpus(
    *,
    corpus_dataset: Dataset,
    corpus_id_column_name: str,
    corpus_text_column_name: str,
    scoring_cfg: DictConfig,
    score_dataset_cfg: DictConfig,
) -> _DocTextLookup | _SqliteDocTextLookup:
    backend = _resolve_doc_lookup_backend(scoring_cfg)
    cache_size: int = int(scoring_cfg.doc_text_cache_size)
    if backend == "dict":
        corpus_id_to_idx = _build_id_to_idx_map(corpus_dataset, corpus_id_column_name)
        corpus_text_column = resolve_dataset_column(
            corpus_dataset, corpus_text_column_name
        )
        return _build_doc_text_lookup(
            corpus_id_to_idx,
            corpus_text_column,
            cache_size=cache_size,
        )
    db_path = _resolve_doc_lookup_db_path(scoring_cfg, score_dataset_cfg)
    id_column = resolve_dataset_column(corpus_dataset, corpus_id_column_name)
    text_column = resolve_dataset_column(corpus_dataset, corpus_text_column_name)
    chunk_size: int = max(int(scoring_cfg.doc_lookup_chunk_size), 1)
    _ensure_sqlite_doc_lookup(
        db_path=db_path,
        id_column=id_column,
        text_column=text_column,
        chunk_size=chunk_size,
    )
    query_chunk_size = max(int(scoring_cfg.doc_lookup_query_chunk_size), 1)
    return _SqliteDocTextLookup(
        db_path=db_path,
        cache_size=cache_size,
        query_chunk_size=query_chunk_size,
    )


@dataclass
class _SqliteDocTextLookup:
    db_path: Path
    cache_size: int
    query_chunk_size: int
    _cache: OrderedDict[str, str] = field(default_factory=OrderedDict, init=False)
    _conn: sqlite3.Connection | None = field(default=None, init=False)

    def __call__(self, doc_id: str) -> str:
        if not doc_id:
            return ""
        if self.cache_size <= 0:
            return self._lookup_text(doc_id)
        cached = self._cache.get(doc_id)
        if cached is not None:
            self._cache.move_to_end(doc_id)
            return cached
        value = self._lookup_text(doc_id)
        self._cache[doc_id] = value
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_conn"] = None
        return state

    def lookup_many(self, doc_ids: Iterable[str]) -> list[str]:
        doc_id_list = list(doc_ids)
        if not doc_id_list:
            return []
        results: list[str] = [""] * len(doc_id_list)
        to_fetch: dict[str, list[int]] = {}
        for idx, doc_id in enumerate(doc_id_list):
            if not doc_id:
                continue
            cached = self._cache.get(doc_id)
            if cached is not None:
                self._cache.move_to_end(doc_id)
                results[idx] = cached
                continue
            to_fetch.setdefault(doc_id, []).append(idx)
        if not to_fetch:
            return results
        conn = self._get_conn()
        chunk_size = max(1, int(self.query_chunk_size))
        for chunk in _chunked(list(to_fetch.keys()), chunk_size):
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT doc_id, text FROM doc_text WHERE doc_id IN ({placeholders})",
                chunk,
            ).fetchall()
            row_map = {str(doc_id): text for doc_id, text in rows}
            for doc_id in chunk:
                text_value = row_map.get(doc_id)
                text = "" if text_value is None else str(text_value)
                for idx in to_fetch[doc_id]:
                    results[idx] = text
                if self.cache_size > 0:
                    self._cache[doc_id] = text
                    self._cache.move_to_end(doc_id)
                    if len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        return results

    def _lookup_text(self, doc_id: str) -> str:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT text FROM doc_text WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return ""
        value = row[0]
        return "" if value is None else str(value)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False
            )
        return self._conn


def _resolve_doc_lookup_backend(scoring_cfg: DictConfig) -> str:
    backend: str = str(scoring_cfg.doc_lookup_backend).lower()
    if backend not in {"dict", "sqlite"}:
        raise ValueError(f"Invalid doc_lookup_backend={backend}")
    return backend


def _resolve_doc_lookup_cache_dir(scoring_cfg: DictConfig) -> Path:
    cache_dir_value: str | None = normalize_optional_str(
        scoring_cfg.doc_lookup_cache_dir
    )
    cache_dir: str = (
        cache_dir_value if cache_dir_value is not None else str(scoring_cfg.output_dir)
    )
    return Path(cache_dir)


def _resolve_corpus_cache_tag(score_dataset_cfg: DictConfig) -> str:
    corpus_name_value: str | None = normalize_optional_str(
        score_dataset_cfg.query_corpus_hf_name
    )
    if corpus_name_value is None:
        corpus_name_value = normalize_optional_str(score_dataset_cfg.hf_name)
    corpus_name: str = corpus_name_value or "corpus"
    return _slugify_component(corpus_name, fallback="corpus")


def _resolve_doc_lookup_db_path(
    scoring_cfg: DictConfig, score_dataset_cfg: DictConfig
) -> Path:
    base_dir = _resolve_doc_lookup_cache_dir(scoring_cfg)
    corpus_tag = _resolve_corpus_cache_tag(score_dataset_cfg)
    return base_dir / "doc_lookup" / f"{corpus_tag}.sqlite"


def _acquire_build_lock(lock_path: Path) -> bool:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _wait_for_file(path: Path, *, timeout_s: int, poll_s: float = 2.0) -> None:
    start = time.time()
    while not path.exists():
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(poll_s)


def _chunked(values: list[str], chunk_size: int) -> Iterable[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _ensure_sqlite_doc_lookup(
    *,
    db_path: Path,
    id_column: pa.Array | pa.ChunkedArray,
    text_column: pa.Array | pa.ChunkedArray,
    chunk_size: int,
) -> None:
    if db_path.exists():
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = db_path.with_suffix(".lock")
    if not _acquire_build_lock(lock_path):
        _wait_for_file(db_path, timeout_s=3600)
        return
    tmp_path = db_path.with_suffix(".tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        conn = sqlite3.connect(str(tmp_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("CREATE TABLE doc_text (doc_id TEXT PRIMARY KEY, text TEXT)")
        table = pa.Table.from_arrays([id_column, text_column], names=["doc_id", "text"])
        for batch in table.to_batches(max_chunksize=chunk_size):
            ids = batch.column(0).to_pylist()
            texts = batch.column(1).to_pylist()
            rows: list[tuple[str, str]] = []
            for doc_id, text in zip(ids, texts):
                if doc_id is None:
                    continue
                doc_text = "" if text is None else str(text)
                rows.append((str(doc_id), doc_text))
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO doc_text (doc_id, text) VALUES (?, ?)",
                    rows,
                )
        conn.commit()
        conn.close()
        tmp_path.replace(db_path)
    finally:
        if lock_path.exists():
            lock_path.unlink()


def _load_positive_cache(
    cache_path: Path, allowed_qids: set[str]
) -> dict[str, list[str]]:
    table: pa.Table = pq.read_table(cache_path, columns=["qid", "doc_ids"])
    qids: list[str] = [str(qid) for qid in table.column("qid").to_pylist()]
    doc_ids_list: list[list[str] | None] = table.column("doc_ids").to_pylist()
    positives: dict[str, list[str]] = {}
    for qid, doc_ids in zip(qids, doc_ids_list):
        if not qid or not doc_ids:
            continue
        if allowed_qids and qid not in allowed_qids:
            continue
        pos_ids: list[str] = _dedupe_preserve_order(str(doc_id) for doc_id in doc_ids)
        if not pos_ids:
            continue
        positives[qid] = pos_ids
    return positives


def _load_positive_doc_ids(
    settings: PositivesSettings | None, allowed_qids: set[str]
) -> dict[str, list[str]]:
    if settings is None or not settings.enabled:
        return {}
    if not allowed_qids:
        return {}

    cache_path: Path | None = Path(settings.cache_path) if settings.cache_path else None
    if cache_path is not None:
        if cache_path.exists():
            log_if_rank_zero(
                logger, f"Loading positives cache from {cache_path.as_posix()}."
            )
            positives_from_cache = _load_positive_cache(cache_path, allowed_qids)
            log_if_rank_zero(
                logger,
                f"Loaded positives for {len(positives_from_cache)} queries from cache.",
            )
            return positives_from_cache
        log_if_rank_zero(
            logger,
            f"Positives cache missing at {cache_path.as_posix()}, scanning triplets.",
            level="warning",
        )

    if settings.hf_name is None:
        raise ValueError("positives.hf_name must be set when no cache is available.")

    positives_dataset: Dataset = load_dataset(
        settings.hf_name,
        name=settings.hf_subset,
        split=settings.hf_split,
        cache_dir=settings.hf_cache_dir,
        streaming=False,
        data_files=_normalize_data_files(settings.hf_data_files),
    )
    qid_column: str = _resolve_column(
        positives_dataset.column_names, ("query_id", "qid", "_id")
    )
    pos_column: str = _resolve_column(
        positives_dataset.column_names, ("positive_id", "pos_id", "doc_pos_id")
    )
    positives: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for row in positives_dataset:
        qid: str = str(row.get(qid_column) or "")
        if not qid or (allowed_qids and qid not in allowed_qids):
            continue
        pos_id: str = str(row.get(pos_column) or "")
        if not pos_id:
            continue
        qid_seen: set[str] = seen.setdefault(qid, set())
        if pos_id in qid_seen:
            continue
        qid_seen.add(pos_id)
        positives.setdefault(qid, []).append(pos_id)
    log_if_rank_zero(
        logger, f"Loaded positives for {len(positives)} queries from triplets."
    )
    return positives


class ScoringPDModule(TorchDataset):
    """PyTorch dataset for cross-encoder scoring."""

    def __init__(
        self,
        *,
        score_dataset_cfg: DictConfig,
        scoring_cfg: DictConfig,
        mining_cfg: DictConfig,
        positives_cfg: DictConfig | None,
    ) -> None:
        self._score_dataset_cfg: DictConfig = score_dataset_cfg
        self._scoring_cfg: DictConfig = scoring_cfg
        self._mining_cfg: DictConfig = mining_cfg
        self._positives_cfg: DictConfig | None = positives_cfg

        self._max_rows: int | None = (
            None if scoring_cfg.max_rows is None else int(scoring_cfg.max_rows)
        )
        self._doc_text_cache_size: int = int(scoring_cfg.doc_text_cache_size)
        self._doc_source: str | None = normalize_optional_str(scoring_cfg.doc_source)

        self._mining_settings: MiningInputSettings = _parse_mining_settings(
            self._mining_cfg
        )
        self._positives_settings: PositivesSettings | None = _parse_positives_settings(
            self._positives_cfg
        )

        self._score_dataset: Dataset | None = None
        self._query_dataset: Dataset | None = None
        self._corpus_dataset: Dataset | None = None
        self._query_id_to_idx: dict[str, int] | None = None
        self._corpus_id_to_idx: dict[str, int] | None = None
        self._query_text_column: pa.Array | pa.ChunkedArray | None = None
        self._corpus_text_column: pa.Array | pa.ChunkedArray | None = None
        self._doc_text_lookup: _DocTextLookup | _SqliteDocTextLookup | None = None
        self._positives_by_qid: dict[str, list[str]] | None = None

    def prepare_data(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if int(torch.distributed.get_rank()) != 0:
                return
        local_files_only = _resolve_local_files_only(
            self._scoring_cfg, is_primary=True
        )
        _apply_hf_offline_mode(local_files_only)
        _ = self._load_score_dataset(local_files_only=local_files_only)
        _ = self._load_query_dataset(local_files_only=local_files_only)
        _ = self._load_corpus_dataset(local_files_only=local_files_only)
        if self._positives_settings is not None and self._positives_settings.enabled:
            cache_path_value: str | None = self._positives_settings.cache_path
            if cache_path_value is None:
                download_config = _resolve_download_config(
                    self._scoring_cfg, local_files_only=local_files_only
                )
                _ = load_dataset(
                    self._positives_settings.hf_name,
                    name=self._positives_settings.hf_subset,
                    split=self._positives_settings.hf_split,
                    cache_dir=self._positives_settings.hf_cache_dir,
                    streaming=False,
                    data_files=_normalize_data_files(
                        self._positives_settings.hf_data_files
                    ),
                    download_config=download_config,
                )

    def setup(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_primary = int(torch.distributed.get_rank()) == 0
            local_files_only = _resolve_local_files_only(
                self._scoring_cfg, is_primary=is_primary
            )
            if is_primary:
                self._load_all_datasets(local_files_only=local_files_only)
            torch.distributed.barrier()
            if not is_primary:
                local_files_only = _resolve_local_files_only(
                    self._scoring_cfg, is_primary=False
                )
                self._load_all_datasets(local_files_only=local_files_only)
        else:
            self._load_all_datasets(
                local_files_only=_resolve_local_files_only(
                    self._scoring_cfg, is_primary=True
                )
            )

        query_id_column: str = _resolve_configured_column(
            self._score_dataset_cfg,
            "query_id_column",
            self._query_dataset.column_names,
            ("query_id", "qid", "_id", "id"),
        )
        query_text_column_name: str = _resolve_configured_column(
            self._score_dataset_cfg,
            "query_text_column",
            self._query_dataset.column_names,
            ("text", "query"),
        )
        corpus_id_column: str = _resolve_configured_column(
            self._score_dataset_cfg,
            "corpus_id_column",
            self._corpus_dataset.column_names,
            ("doc_id", "corpus_id", "passage_id", "_id", "id"),
        )
        corpus_text_column_name: str = _resolve_configured_column(
            self._score_dataset_cfg,
            "corpus_text_column",
            self._corpus_dataset.column_names,
            ("text", "passage", "contents"),
        )

        self._query_id_to_idx = _build_id_to_idx_map(
            self._query_dataset, query_id_column
        )
        self._query_text_column = resolve_dataset_column(
            self._query_dataset, query_text_column_name
        )
        self._doc_text_lookup = _build_doc_text_lookup_for_corpus(
            corpus_dataset=self._corpus_dataset,
            corpus_id_column_name=corpus_id_column,
            corpus_text_column_name=corpus_text_column_name,
            scoring_cfg=self._scoring_cfg,
            score_dataset_cfg=self._score_dataset_cfg,
        )
        self._corpus_id_to_idx = None
        self._corpus_text_column = None

        if self._positives_by_qid is None:
            positives_enabled: bool = (
                self._positives_settings is not None
                and self._positives_settings.enabled
            )
            if positives_enabled:
                log_if_rank_zero(
                    logger,
                    f"Collecting qids...",
                    level="info",
                )
                allowed_qids: set[str] = _collect_qids(
                    self._score_dataset, self._max_rows
                )
                log_if_rank_zero(
                    logger,
                    f"Loading positives for {len(allowed_qids):,} qids...",
                    level="info",
                )
                self._positives_by_qid = _load_positive_doc_ids(
                    self._positives_settings, allowed_qids
                )
            else:
                self._positives_by_qid = {}

    def __len__(self) -> int:
        self._ensure_ready()
        if self._score_dataset is None:
            return 0
        row_count: int = int(len(self._score_dataset))
        if self._max_rows is None:
            return row_count
        return min(row_count, self._max_rows)

    def __getitem__(self, idx: int) -> ScoringItem | None:
        self._ensure_ready()
        if self._score_dataset is None:
            return None
        row: dict[str, Any] = dict(self._score_dataset[int(idx)])
        qid: str = str(row.get("query_id") or row.get("qid") or row.get("_id") or "")
        if not qid:
            return None

        inline_query_text: str | None = normalize_optional_str(row.get("query_text"))
        if inline_query_text is not None:
            query_text = inline_query_text
        else:
            query_idx: int = int(self._query_id_to_idx.get(qid, -1))
            if query_idx < 0:
                return None
            query_text = _column_text_value(self._query_text_column, query_idx)
            if not query_text:
                return None

        doc_ids: list[str]
        labels: list[float] | None = None
        doc_sources: list[str] | None = None
        neg_source: str = self._doc_source or "neg"
        if "pos_doc_ids" in row or "neg_doc_ids" in row:
            pos_ids: list[str] = _dedupe_preserve_order(
                str(doc_id) for doc_id in row.get("pos_doc_ids") or []
            )
            neg_ids: list[str] = _dedupe_preserve_order(
                str(doc_id) for doc_id in row.get("neg_doc_ids") or []
            )
            pos_id_set: set[str] = set(pos_ids)
            if pos_id_set:
                neg_ids = [doc_id for doc_id in neg_ids if doc_id not in pos_id_set]
            if not pos_ids:
                log_if_rank_zero(
                    logger,
                    f"Skipping {qid}: missing positives for hard negatives.",
                    level="warning",
                )
                return None
            if not neg_ids:
                log_if_rank_zero(
                    logger,
                    f"Skipping {qid}: missing hard negatives after merge.",
                    level="warning",
                )
                return None
            doc_ids = pos_ids + neg_ids
            labels = [1.0] * len(pos_ids) + [0.0] * len(neg_ids)
            doc_sources = ["pos"] * len(pos_ids) + [neg_source] * len(neg_ids)
        elif "doc_ids" in row:
            raw_doc_ids: list[str] = [
                str(doc_id) for doc_id in row.get("doc_ids") or []
            ]
            label_values: Any | None = row.get("labels")
            if label_values is not None:
                labels = [float(value) for value in label_values]
                if len(labels) != len(raw_doc_ids):
                    log_if_rank_zero(
                        logger,
                        f"Skipping {qid}: label count does not match doc_ids.",
                        level="warning",
                    )
                    return None
            if labels is None:
                pos_ids: list[str] = _dedupe_preserve_order(
                    self._positives_by_qid.get(qid, [])
                )
                if not pos_ids:
                    log_if_rank_zero(
                        logger,
                        f"Skipping {qid}: missing positives for hard negatives.",
                        level="warning",
                    )
                    return None
                pos_id_set: set[str] = set(pos_ids)
                neg_ids: list[str] = [
                    doc_id
                    for doc_id in raw_doc_ids
                    if doc_id and doc_id not in pos_id_set
                ]
                if not neg_ids:
                    log_if_rank_zero(
                        logger,
                        f"Skipping {qid}: missing hard negatives after merge.",
                        level="warning",
                    )
                    return None
                doc_ids = pos_ids + neg_ids
                labels = [1.0] * len(pos_ids) + [0.0] * len(neg_ids)
            else:
                doc_ids = raw_doc_ids
            if labels is not None:
                doc_sources = _labels_to_sources(labels, neg_source=neg_source)
            elif self._doc_source is not None:
                doc_sources = [self._doc_source] * len(doc_ids)
        else:
            pos_id: str = str(
                row.get("positive_id")
                or row.get("pos_id")
                or row.get("doc_pos_id")
                or ""
            )
            neg_id: str = str(
                row.get("negative_id")
                or row.get("neg_id")
                or row.get("doc_neg_id")
                or ""
            )
            doc_ids = []
            doc_sources = []
            if pos_id:
                doc_ids.append(pos_id)
                doc_sources.append("pos")
            if neg_id:
                doc_ids.append(neg_id)
                doc_sources.append(neg_source)
        if not doc_ids:
            return None

        doc_texts: list[str] = self._doc_text_lookup.lookup_many(doc_ids)
        return ScoringItem(
            row=row,
            qid=qid,
            doc_ids=doc_ids,
            labels=labels,
            doc_sources=doc_sources,
            query_text=query_text,
            doc_texts=doc_texts,
        )

    def _ensure_ready(self) -> None:
        if self._score_dataset is None or self._doc_text_lookup is None:
            self.setup()

    def _resolve_score_dataset_inputs(self) -> tuple[str, Any | None]:
        if self._mining_settings.enabled:
            if normalize_optional_str(self._mining_settings.model_name) is None:
                raise ValueError(
                    "mining.model_name must be set when mining is enabled for scoring."
                )
            resolved_format, resolved_files = _resolve_mined_input_files(
                self._mining_settings, self._score_dataset_cfg
            )
            if resolved_format is not None:
                return resolved_format, resolved_files
            raise ValueError("Unable to resolve mined inputs for scoring.")
        hf_name: str | None = normalize_optional_str(self._score_dataset_cfg.hf_name)
        if hf_name is None and self._score_dataset_cfg.hf_data_files is None:
            raise ValueError(
                "score_dataset.hf_name or score_dataset.hf_data_files must be set "
                "when mining is disabled."
            )
        return (str(self._score_dataset_cfg.hf_name), self._score_dataset_cfg.hf_data_files)

    def _load_all_datasets(self, *, local_files_only: bool) -> None:
        _apply_hf_offline_mode(local_files_only)
        if self._score_dataset is None:
            self._score_dataset = self._load_score_dataset(
                local_files_only=local_files_only
            )
        if self._query_dataset is None:
            self._query_dataset = self._load_query_dataset(
                local_files_only=local_files_only
            )
        if self._corpus_dataset is None:
            self._corpus_dataset = self._load_corpus_dataset(
                local_files_only=local_files_only
            )

    def _load_score_dataset(self, *, local_files_only: bool) -> Dataset:
        resolved_hf_name, resolved_data_files = self._resolve_score_dataset_inputs()
        download_config = _resolve_download_config(
            self._scoring_cfg, local_files_only=local_files_only
        )
        return _load_dataset_from_config(
            self._score_dataset_cfg,
            hf_name=resolved_hf_name,
            data_files=resolved_data_files,
            download_config=download_config,
        )

    def _load_query_dataset(self, *, local_files_only: bool) -> Dataset:
        text_name_value: str | None = normalize_optional_str(
            self._score_dataset_cfg.query_corpus_hf_name
        )
        text_name: str = (
            str(text_name_value)
            if text_name_value is not None
            else str(self._score_dataset_cfg.hf_name)
        )
        text_cache_dir_value: str | None = normalize_optional_str(
            self._score_dataset_cfg.query_corpus_hf_cache_dir
        )
        text_cache_dir: str | None = (
            text_cache_dir_value
            if text_cache_dir_value is not None
            else self._score_dataset_cfg.hf_cache_dir
        )
        query_subset_name: str = (
            normalize_optional_str(self._score_dataset_cfg.query_subset_name)
            or "queries"
        )
        query_split_name: str = (
            normalize_optional_str(self._score_dataset_cfg.query_split_name) or "train"
        )
        query_corpus_data_files: Any | None = _normalize_data_files(
            self._score_dataset_cfg.query_corpus_hf_data_files
        )
        download_config = _resolve_download_config(
            self._scoring_cfg, local_files_only=local_files_only
        )
        return load_dataset(
            text_name,
            name=query_subset_name,
            split=query_split_name,
            cache_dir=text_cache_dir,
            data_files=query_corpus_data_files,
            download_config=download_config,
        )

    def _load_corpus_dataset(self, *, local_files_only: bool) -> Dataset:
        text_name_value: str | None = normalize_optional_str(
            self._score_dataset_cfg.query_corpus_hf_name
        )
        text_name: str = (
            str(text_name_value)
            if text_name_value is not None
            else str(self._score_dataset_cfg.hf_name)
        )
        text_cache_dir_value: str | None = normalize_optional_str(
            self._score_dataset_cfg.query_corpus_hf_cache_dir
        )
        text_cache_dir: str | None = (
            text_cache_dir_value
            if text_cache_dir_value is not None
            else self._score_dataset_cfg.hf_cache_dir
        )
        corpus_subset_name: str = (
            normalize_optional_str(self._score_dataset_cfg.corpus_subset_name)
            or "corpus"
        )
        corpus_split_name: str = (
            normalize_optional_str(self._score_dataset_cfg.corpus_split_name) or "train"
        )
        query_corpus_data_files: Any | None = _normalize_data_files(
            self._score_dataset_cfg.query_corpus_hf_data_files
        )
        download_config = _resolve_download_config(
            self._scoring_cfg, local_files_only=local_files_only
        )
        return load_dataset(
            text_name,
            name=corpus_subset_name,
            split=corpus_split_name,
            cache_dir=text_cache_dir,
            data_files=query_corpus_data_files,
            download_config=download_config,
        )
