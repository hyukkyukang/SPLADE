import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.path import ABS_CONFIG_DIR
from src.data.utils import id_to_idx, resolve_dataset_column
from src.utils import set_seed
from src.utils.logging import get_logger
from src.utils.script_setup import configure_script_environment

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


def _normalize_optional_str(value: Any) -> str | None:
    """Normalize optional string values from configs."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized: str = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
    return str(value)


def _load_dataset_from_config(cfg: DictConfig) -> Dataset:
    """Load a dataset based on the dataset config block."""
    hf_name: str = str(cfg.hf_name)
    hf_subset: str | None = _normalize_optional_str(getattr(cfg, "hf_subset", None))
    hf_split: str = str(cfg.hf_split)
    hf_cache_dir: str | None = getattr(cfg, "hf_cache_dir", None)
    data_files: Mapping[str, Any] | None = getattr(cfg, "hf_data_files", None)
    dataset: Dataset = load_dataset(
        hf_name,
        name=hf_subset,
        split=hf_split,
        cache_dir=hf_cache_dir,
        streaming=False,
        data_files=dict(data_files) if data_files else None,
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
) -> None:
    """Write JSONL rows to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


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
    text_name: str = str(score_dataset_cfg.hf_text_name or score_dataset_cfg.hf_name)
    text_cache_dir: str | None = getattr(score_dataset_cfg, "hf_cache_dir", None)
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

    output_path: Path = Path(settings.output_dir) / f"{settings.output_basename}.jsonl"
    if output_path.exists() and not settings.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    def _scored_rows() -> Iterable[dict[str, Any]]:
        row_count: int = 0
        for row in score_dataset:
            if settings.max_rows is not None and row_count >= settings.max_rows:
                break
            qid: str = str(
                row.get("query_id") or row.get("qid") or row.get("_id") or ""
            )
            if not qid:
                continue
            query_idx: int = int(query_id_to_idx.get(qid, -1))
            if query_idx < 0:
                continue
            query_row: dict[str, Any] = query_dataset[query_idx]
            query_text: str = str(query_row.get(query_text_column) or "")
            if not query_text:
                continue

            doc_ids: list[str]
            if "doc_ids" in row:
                doc_ids = [str(doc_id) for doc_id in row.get("doc_ids") or []]
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

            doc_texts: list[str] = []
            for doc_id in doc_ids:
                doc_idx: int = int(corpus_id_to_idx.get(doc_id, -1))
                if doc_idx < 0:
                    doc_texts.append("")
                    continue
                corpus_row: dict[str, Any] = corpus_dataset[doc_idx]
                doc_text: str = str(corpus_row.get(corpus_text_column) or "")
                doc_texts.append(doc_text)

            scores: list[float] = _score_pairs(
                model=model,
                tokenizer=tokenizer,
                query_text=query_text,
                doc_texts=doc_texts,
                device=device,
                batch_size=settings.batch_size,
                max_length=settings.max_length,
            )
            output_row: dict[str, Any] = dict(row)
            output_row["query_id"] = qid
            output_row[settings.score_key] = scores
            yield output_row
            row_count += 1

    _write_jsonl(output_path, _scored_rows())
    logger.info("Saved scored dataset to %s", output_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
