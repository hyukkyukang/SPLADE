import json
import logging
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.path import ABS_CONFIG_DIR
from src.data.utils import id_to_idx, resolve_dataset_column
from src.utils import set_seed
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.script_setup import configure_script_environment, normalize_optional_str

logger: logging.Logger = get_logger("scripts.preprocess.score_cross_encoder", __file__)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=True,
    suppress_lightning_tips=False,
    suppress_httpx=False,
    suppress_dataloader_workers=True,
)


@dataclass(frozen=True)
class ScoringSettings:
    """Typed container for scoring settings."""

    model_name: str
    output_dir: str
    output_basename: str
    output_format: str
    batch_size: int
    max_length: int
    use_cpu: bool
    max_rows: int | None
    overwrite: bool
    score_key: str


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
class _RowPayload:
    row: dict[str, Any]
    qid: str
    query_text: str
    doc_ids: list[str]
    labels: list[float] | None
    doc_texts: list[str]


@dataclass
class _RowState:
    payload: _RowPayload
    scores: list[float | None]
    remaining: int


def _parse_scoring_settings(cfg: DictConfig) -> ScoringSettings:
    """Parse Hydra config into a typed ScoringSettings instance."""
    scoring_cfg: DictConfig = cfg.scoring
    model_name: str = str(scoring_cfg.model_name)
    output_dir: str = str(scoring_cfg.output_dir)
    output_basename: str = str(scoring_cfg.output_basename)
    output_format: str = str(scoring_cfg.output_format).lower()
    batch_size: int = int(scoring_cfg.batch_size)
    max_length: int = int(scoring_cfg.max_length)
    use_cpu: bool = bool(scoring_cfg.use_cpu)
    max_rows: int | None = (
        None if scoring_cfg.max_rows is None else int(scoring_cfg.max_rows)
    )
    overwrite: bool = bool(scoring_cfg.overwrite)
    score_key: str = str(scoring_cfg.score_key)
    return ScoringSettings(
        model_name=model_name,
        output_dir=output_dir,
        output_basename=output_basename,
        output_format=output_format,
        batch_size=batch_size,
        max_length=max_length,
        use_cpu=use_cpu,
        max_rows=max_rows,
        overwrite=overwrite,
        score_key=score_key,
    )


def _parse_positives_settings(cfg: DictConfig | None) -> PositivesSettings | None:
    if cfg is None:
        return None
    enabled: bool = bool(cfg.get("enabled", True))
    cache_path: str | None = normalize_optional_str(cfg.get("cache_path"))
    hf_name: str | None = normalize_optional_str(cfg.get("hf_name"))
    hf_subset: str | None = normalize_optional_str(cfg.get("hf_subset"))
    hf_split: str = str(cfg.get("hf_split") or "train")
    hf_cache_dir: str | None = normalize_optional_str(cfg.get("hf_cache_dir"))
    hf_data_files: Any | None = cfg.get("hf_data_files")
    return PositivesSettings(
        enabled=enabled,
        cache_path=cache_path,
        hf_name=hf_name,
        hf_subset=hf_subset,
        hf_split=hf_split,
        hf_cache_dir=hf_cache_dir,
        hf_data_files=hf_data_files,
    )


def _normalize_data_files(data_files: Any | None) -> Any | None:
    if data_files is None:
        return None
    if isinstance(data_files, (str, list, tuple)):
        return data_files
    if isinstance(data_files, Mapping):
        return dict(data_files)
    raise TypeError("hf_data_files must be a path, list/tuple of paths, or a mapping.")


def _load_dataset_from_config(cfg: DictConfig) -> Dataset:
    """Load a dataset based on the dataset config block."""
    hf_name: str = str(cfg.hf_name)
    hf_subset: str | None = normalize_optional_str(cfg.hf_subset)
    hf_split: str = str(cfg.hf_split)
    hf_cache_dir: str | None = cfg.hf_cache_dir
    data_files: Any | None = _normalize_data_files(cfg.hf_data_files)
    dataset: Dataset = load_dataset(
        hf_name,
        name=hf_subset,
        split=hf_split,
        cache_dir=hf_cache_dir,
        streaming=False,
        data_files=data_files,
    )
    return dataset


def _resolve_column(column_names: Iterable[str], candidates: Sequence[str]) -> str:
    """Pick the first matching column name."""
    for name in candidates:
        if name in column_names:
            return name
    raise ValueError(f"Unable to resolve column from {list(column_names)}")


def _build_id_to_idx_map(dataset: Dataset, id_column: str) -> dict[str, int]:
    """Build id->index mapping for dataset columns."""
    column = resolve_dataset_column(dataset, id_column)
    mapping: dict[str, int] = id_to_idx(column, desc="id_to_idx", enable_tqdm=False)
    return mapping


def _extract_scores(logits: torch.Tensor) -> list[float]:
    """Convert model logits to a flat list of scores."""
    if logits.ndim == 2 and logits.shape[1] > 1:
        scores_tensor: torch.Tensor = logits[:, 0]
    else:
        scores_tensor = logits.squeeze(-1)
    scores: list[float] = [
        float(value) for value in scores_tensor.detach().cpu().tolist()
    ]
    return scores


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
    for row in score_dataset:
        if max_rows is not None and row_count >= max_rows:
            break
        qid: str = str(row.get("query_id") or row.get("qid") or row.get("_id") or "")
        if not qid:
            continue
        qids.add(qid)
        row_count += 1
    return qids


def _build_doc_text_lookup(
    corpus_id_to_idx: dict[str, int],
    corpus_text_column: pa.Array | pa.ChunkedArray,
    *,
    cache_size: int,
) -> Callable[[str], str]:
    if cache_size <= 0:

        def _lookup(doc_id: str) -> str:
            doc_idx: int = int(corpus_id_to_idx.get(doc_id, -1))
            if doc_idx < 0:
                return ""
            return _column_text_value(corpus_text_column, doc_idx)

        return _lookup

    @lru_cache(maxsize=cache_size)
    def _lookup(doc_id: str) -> str:
        doc_idx: int = int(corpus_id_to_idx.get(doc_id, -1))
        if doc_idx < 0:
            return ""
        return _column_text_value(corpus_text_column, doc_idx)

    return _lookup


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


def _iter_scoring_rows(
    score_dataset: Dataset,
    *,
    query_id_to_idx: dict[str, int],
    query_text_column: pa.Array | pa.ChunkedArray,
    doc_text_lookup: Callable[[str], str],
    settings: ScoringSettings,
    positives_by_qid: dict[str, list[str]],
) -> Iterable[_RowPayload]:
    row_count: int = 0
    for raw_row in score_dataset:
        if settings.max_rows is not None and row_count >= settings.max_rows:
            break
        row: dict[str, Any] = dict(raw_row)
        qid: str = str(row.get("query_id") or row.get("qid") or row.get("_id") or "")
        if not qid:
            continue
        query_idx: int = int(query_id_to_idx.get(qid, -1))
        if query_idx < 0:
            continue
        query_text: str = _column_text_value(query_text_column, query_idx)
        if not query_text:
            continue

        doc_ids: list[str]
        labels: list[float] | None = None
        if "doc_ids" in row:
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
                    continue
            if labels is None:
                pos_ids: list[str] = _dedupe_preserve_order(
                    positives_by_qid.get(qid, [])
                )
                if not pos_ids:
                    log_if_rank_zero(
                        logger,
                        f"Skipping {qid}: missing positives for hard negatives.",
                        level="warning",
                    )
                    continue
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
                    continue
                doc_ids = pos_ids + neg_ids
                labels = [1.0] * len(pos_ids) + [0.0] * len(neg_ids)
            else:
                doc_ids = raw_doc_ids
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
            doc_ids = [doc_id for doc_id in (pos_id, neg_id) if doc_id]
        if not doc_ids:
            continue

        doc_texts: list[str] = [doc_text_lookup(doc_id) for doc_id in doc_ids]
        yield _RowPayload(
            row=row,
            qid=qid,
            query_text=query_text,
            doc_ids=doc_ids,
            labels=labels,
            doc_texts=doc_texts,
        )
        row_count += 1


def _score_payloads(
    payloads: Iterable[_RowPayload],
    *,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    score_key: str,
) -> Iterable[dict[str, Any]]:
    pending_pairs: list[tuple[int, int, str, str]] = []
    row_queue: deque[int] = deque()
    row_states: dict[int, _RowState] = {}
    next_row_id: int = 0

    def _run_batch(pairs: list[tuple[int, int, str, str]]) -> None:
        if not pairs:
            return
        batch_queries: list[str] = [pair[2] for pair in pairs]
        batch_docs: list[str] = [pair[3] for pair in pairs]
        tokens: dict[str, torch.Tensor] = tokenizer(
            batch_queries,
            batch_docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {
            key: value.to(device, non_blocking=True) for key, value in tokens.items()
        }
        outputs = model(**tokens)
        batch_scores: list[float] = _extract_scores(outputs.logits)
        for (row_id, doc_idx, _, _), score in zip(pairs, batch_scores):
            state: _RowState = row_states[row_id]
            state.scores[doc_idx] = score
            state.remaining -= 1

    def _flush_ready() -> Iterable[dict[str, Any]]:
        while row_queue and row_states[row_queue[0]].remaining == 0:
            row_id = row_queue.popleft()
            state = row_states.pop(row_id)
            payload = state.payload
            output_row: dict[str, Any] = dict(payload.row)
            output_row["query_id"] = payload.qid
            output_row["doc_ids"] = payload.doc_ids
            if payload.labels is not None:
                output_row["labels"] = payload.labels
            output_row[score_key] = [float(score) for score in state.scores]
            yield output_row

    with torch.inference_mode():
        for payload in payloads:
            if not payload.doc_texts:
                continue
            row_id: int = next_row_id
            next_row_id += 1
            row_states[row_id] = _RowState(
                payload=payload,
                scores=[None] * len(payload.doc_texts),
                remaining=len(payload.doc_texts),
            )
            row_queue.append(row_id)
            for doc_idx, doc_text in enumerate(payload.doc_texts):
                pending_pairs.append((row_id, doc_idx, payload.query_text, doc_text))
                if len(pending_pairs) >= batch_size:
                    batch = pending_pairs[:batch_size]
                    del pending_pairs[:batch_size]
                    _run_batch(batch)
                    yield from _flush_ready()

        if pending_pairs:
            _run_batch(pending_pairs)
            pending_pairs = []
            yield from _flush_ready()
        yield from _flush_ready()


def _score_pairs(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    query_text: str,
    doc_texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[float]:
    """Score (query, doc) pairs with the cross-encoder."""
    scores: list[float] = []
    for start in range(0, len(doc_texts), batch_size):
        batch_docs: list[str] = doc_texts[start : start + batch_size]
        batch_queries: list[str] = [query_text for _ in batch_docs]
        tokens: dict[str, torch.Tensor] = tokenizer(
            batch_queries,
            batch_docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = tokens["input_ids"].to(device)
        attention_mask: torch.Tensor = tokens["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_scores: list[float] = _extract_scores(outputs.logits)
        scores.extend(batch_scores)
    return scores


def _write_jsonl(
    output_path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    flush_every: int = 1000,
) -> None:
    """Write JSONL rows to disk."""
    flush_every = max(int(flush_every), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        buffer: list[str] = []
        for row in rows:
            buffer.append(json.dumps(row))
            if len(buffer) >= flush_every:
                handle.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            handle.write("\n".join(buffer) + "\n")


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name="score_cross_encoder"
)
def main(cfg: DictConfig) -> None:
    settings: ScoringSettings = _parse_scoring_settings(cfg)
    set_seed(int(cfg.seed))
    if settings.output_format != "jsonl":
        raise ValueError("Only jsonl output is supported for scoring.")

    device_type: str = "cpu" if settings.use_cpu else "cuda"
    device: torch.device = torch.device(device_type)
    model: AutoModelForSequenceClassification = (
        AutoModelForSequenceClassification.from_pretrained(settings.model_name)
    ).to(device)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    model.eval()

    score_dataset_cfg: DictConfig = cfg.score_dataset
    score_dataset: Dataset = _load_dataset_from_config(score_dataset_cfg)
    text_name: str = str(
        score_dataset_cfg.query_corpus_hf_name
        if score_dataset_cfg.query_corpus_hf_name is not None
        else score_dataset_cfg.hf_name
    )
    text_cache_dir: str | None = (
        score_dataset_cfg.query_corpus_hf_cache_dir
        if score_dataset_cfg.query_corpus_hf_cache_dir is not None
        else score_dataset_cfg.hf_cache_dir
    )
    query_dataset: Dataset = load_dataset(
        text_name, "queries", split="train", cache_dir=text_cache_dir
    )
    corpus_dataset: Dataset = load_dataset(
        text_name, "corpus", split="train", cache_dir=text_cache_dir
    )

    query_id_column: str = _resolve_column(
        query_dataset.column_names, ("query_id", "qid", "_id", "id")
    )
    query_text_column: str = _resolve_column(
        query_dataset.column_names, ("text", "query")
    )
    corpus_id_column: str = _resolve_column(
        corpus_dataset.column_names, ("doc_id", "corpus_id", "passage_id", "_id", "id")
    )
    corpus_text_column: str = _resolve_column(
        corpus_dataset.column_names, ("text", "passage", "contents")
    )

    query_id_to_idx: dict[str, int] = _build_id_to_idx_map(
        query_dataset, query_id_column
    )
    corpus_id_to_idx: dict[str, int] = _build_id_to_idx_map(
        corpus_dataset, corpus_id_column
    )

    positives_cfg: DictConfig | None = (
        cfg.get("positives") if "positives" in cfg else None
    )
    positives_settings: PositivesSettings | None = _parse_positives_settings(
        positives_cfg
    )
    allowed_qids: set[str] = set()
    if positives_settings is not None and positives_settings.enabled:
        allowed_qids = _collect_qids(score_dataset, settings.max_rows)
    positives_by_qid: dict[str, list[str]] = _load_positive_doc_ids(
        positives_settings, allowed_qids
    )

    query_text_column_data: pa.Array | pa.ChunkedArray = resolve_dataset_column(
        query_dataset, query_text_column
    )
    corpus_text_column_data: pa.Array | pa.ChunkedArray = resolve_dataset_column(
        corpus_dataset, corpus_text_column
    )
    doc_text_cache_size: int = int(cfg.scoring.get("doc_text_cache_size", 200000))
    doc_text_lookup: Callable[[str], str] = _build_doc_text_lookup(
        corpus_id_to_idx,
        corpus_text_column_data,
        cache_size=doc_text_cache_size,
    )
    flush_every: int = int(cfg.scoring.get("flush_every", 1000))

    output_path: Path = Path(settings.output_dir) / f"{settings.output_basename}.jsonl"
    if output_path.exists() and not settings.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    payloads: Iterable[_RowPayload] = _iter_scoring_rows(
        score_dataset,
        query_id_to_idx=query_id_to_idx,
        query_text_column=query_text_column_data,
        doc_text_lookup=doc_text_lookup,
        settings=settings,
        positives_by_qid=positives_by_qid,
    )
    scored_rows: Iterable[dict[str, Any]] = _score_payloads(
        payloads,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=settings.batch_size,
        max_length=settings.max_length,
        score_key=settings.score_key,
    )

    _write_jsonl(output_path, scored_rows, flush_every=flush_every)
    log_if_rank_zero(logger, f"Saved scored dataset to {output_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
