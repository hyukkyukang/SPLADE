import heapq
import math
from typing import Any, Iterable, Sequence

from datasets import Dataset, load_dataset
from src.data.dataset.base import BaseDataset
from src.data.dataset.utils import normalize_optional_str
from src.utils.logging import get_logger, is_rank_zero, log_if_rank_zero
from tqdm import tqdm

logger = get_logger("MSMARCOTripletScoresDataset")


def _resolve_column(
    configured: str,
    column_names: Iterable[str],
    candidates: Sequence[str],
    *,
    field_name: str,
) -> str:
    configured_value: str | None = normalize_optional_str(configured)
    if configured_value is not None:
        if configured_value in column_names:
            return configured_value
        log_if_rank_zero(
            logger,
            f"Configured {field_name}={configured_value} not found; falling back.",
            level="warning",
        )
    for name in candidates:
        if name in column_names:
            return name
    raise ValueError(f"Unable to resolve {field_name} from {list(column_names)}")


def _load_hf_dataset(
    *,
    hf_name: str,
    hf_subset: str | None,
    hf_split: str,
    hf_cache_dir: str | None,
    hf_data_files: Any | None,
) -> Dataset:
    return load_dataset(
        hf_name,
        name=hf_subset,
        split=hf_split,
        cache_dir=hf_cache_dir,
        data_files=hf_data_files,
    )


def _resolve_qid(row: dict[str, Any]) -> str:
    return str(
        row.get("query_id") or row.get("qid") or row.get("_id") or row.get("id") or ""
    )


def _resolve_pos_id(row: dict[str, Any]) -> str:
    return str(
        row.get("positive_id") or row.get("pos_id") or row.get("doc_pos_id") or ""
    )


class MSMARCOTripletScoresDataset(BaseDataset):
    """Join triplet meta rows with scored hard negatives by query id."""

    def _resolve_meta_dataset(self) -> Dataset:
        if self.hf_name is None:
            raise ValueError("hf_name must be set for triplet meta dataset.")
        base_dataset: Dataset = self._load_hf_dataset(
            hf_name=self.hf_name,
            hf_subset=self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
            data_files=self.hf_data_files,
        )
        base_dataset = self._apply_hf_sample_window(base_dataset)
        log_every: int = int(self.cfg.score_log_every)
        progress_bar: bool = bool(self.cfg.score_progress_bar)
        progress_refresh: float = float(self.cfg.score_progress_refresh)
        show_progress: bool = progress_bar and is_rank_zero()
        base_count: int = int(len(base_dataset))
        log_if_rank_zero(
            logger, f"Triplet base rows available for join: {base_count:,}"
        )

        # Expected score schema: query_id + doc_ids + labels + scores.
        score_hf_name: str | None = normalize_optional_str(self.cfg.score_hf_name)
        if score_hf_name is None:
            raise ValueError("score_hf_name must be set to join scores.")
        score_dataset: Dataset = _load_hf_dataset(
            hf_name=score_hf_name,
            hf_subset=normalize_optional_str(self.cfg.score_hf_subset),
            hf_split=str(self.cfg.score_hf_split),
            hf_cache_dir=normalize_optional_str(self.cfg.score_hf_cache_dir),
            hf_data_files=self.cfg.score_hf_data_files,
        )
        score_count: int = int(len(score_dataset))
        log_if_rank_zero(
            logger, f"Scored rows available for join: {score_count:,}"
        )

        score_qid_column: str = _resolve_column(
            str(self.cfg.score_query_id_column),
            score_dataset.column_names,
            ("query_id", "qid", "_id", "id"),
            field_name="score_query_id_column",
        )
        score_doc_ids_column: str = _resolve_column(
            str(self.cfg.score_doc_ids_column),
            score_dataset.column_names,
            ("doc_ids", "docid", "doc_id", "passage_id"),
            field_name="score_doc_ids_column",
        )
        score_labels_column: str = _resolve_column(
            str(self.cfg.score_labels_column),
            score_dataset.column_names,
            ("labels", "label"),
            field_name="score_labels_column",
        )
        score_scores_column: str = _resolve_column(
            str(self.cfg.score_scores_column),
            score_dataset.column_names,
            ("scores", "score"),
            field_name="score_scores_column",
        )

        missing_policy: str = str(self.cfg.score_missing_policy).lower()
        if missing_policy not in {"error", "drop"}:
            raise ValueError(
                "score_missing_policy must be one of: error, drop. "
                f"Got: {self.cfg.score_missing_policy}"
            )
        duplicate_policy: str = str(self.cfg.score_duplicate_policy).lower()
        if duplicate_policy not in {"error", "first", "last"}:
            raise ValueError(
                "score_duplicate_policy must be one of: error, first, last. "
                f"Got: {self.cfg.score_duplicate_policy}"
            )
        negatives_per_query: int = int(self.cfg.score_negatives_per_query)
        if negatives_per_query <= 0:
            raise ValueError("score_negatives_per_query must be a positive integer.")
        if negatives_per_query > int(self.cfg.num_negatives):
            log_if_rank_zero(
                logger,
                "score_negatives_per_query exceeds num_negatives; "
                "training will sample fewer negatives than the join provides.",
                level="warning",
            )

        score_index: dict[str, dict[str, Any]] = {}
        score_iter = score_dataset
        if show_progress:
            score_iter = tqdm(
                score_dataset,
                total=score_count,
                desc="index scored rows",
                mininterval=progress_refresh,
            )
        for idx, row in enumerate(score_iter):
            qid_value: Any = row.get(score_qid_column)
            if qid_value is None:
                continue
            qid: str = str(qid_value)
            if not qid:
                continue
            if qid in score_index:
                if duplicate_policy == "first":
                    continue
                if duplicate_policy == "error":
                    raise ValueError(f"Duplicate scored row for query_id={qid}.")
            score_index[qid] = {
                "doc_ids": row.get(score_doc_ids_column),
                "labels": row.get(score_labels_column),
                "scores": row.get(score_scores_column),
            }
            if log_every > 0 and (idx + 1) % log_every == 0:
                log_if_rank_zero(
                    logger,
                    "Scanned "
                    f"{idx + 1:,} scored rows (matched={len(score_index):,}).",
                )

        joined_rows: list[dict[str, Any]] = []
        base_iter = base_dataset
        if show_progress:
            base_iter = tqdm(
                base_dataset,
                total=base_count,
                desc="join base rows",
                mininterval=progress_refresh,
            )
        for idx, row in enumerate(base_iter):
            qid = _resolve_qid(row)
            pos_id = _resolve_pos_id(row)
            if not qid or not pos_id:
                if missing_policy == "drop":
                    continue
                raise ValueError(
                    "Triplet row is missing query_id or positive_id; "
                    f"qid={qid!r}, pos_id={pos_id!r}."
                )

            score_row = score_index.get(qid)
            if score_row is None:
                if missing_policy == "drop":
                    continue
                raise ValueError(f"Missing scored row for query_id={qid}.")

            doc_ids: list[str] = [str(doc_id) for doc_id in score_row["doc_ids"] or []]
            labels: list[float] = [
                float(label) for label in score_row["labels"] or []
            ]
            scores: list[float] = [
                float(score) for score in score_row["scores"] or []
            ]

            if not doc_ids or not labels or not scores:
                if missing_policy == "drop":
                    continue
                raise ValueError(f"Scored row missing fields for query_id={qid}.")
            if not (
                len(doc_ids) == len(labels) == len(scores)
            ):
                if missing_policy == "drop":
                    continue
                raise ValueError(
                    "Scored row lengths do not match for query_id="
                    f"{qid}: doc_ids={len(doc_ids)}, labels={len(labels)}, "
                    f"scores={len(scores)}."
                )

            if pos_id not in doc_ids:
                if missing_policy == "drop":
                    continue
                raise ValueError(
                    f"Positive id {pos_id} not found in scored doc_ids for {qid}."
                )

            doc_id_to_idx: dict[str, int] = {
                doc_id: idx for idx, doc_id in enumerate(doc_ids)
            }
            pos_index: int = doc_id_to_idx[pos_id]
            pos_score: float = scores[pos_index]
            if not math.isfinite(pos_score):
                if missing_policy == "drop":
                    continue
                raise ValueError(
                    f"Positive score is not finite for query_id={qid}."
                )

            neg_candidates: list[tuple[str, float]] = []
            for doc_id, label, score in zip(doc_ids, labels, scores):
                if doc_id == pos_id:
                    continue
                if label > 0:
                    continue
                if not math.isfinite(score):
                    continue
                neg_candidates.append((doc_id, score))

            if not neg_candidates:
                if missing_policy == "drop":
                    continue
                raise ValueError(f"No negative candidates found for query_id={qid}.")

            selected = heapq.nlargest(
                negatives_per_query, neg_candidates, key=lambda item: item[1]
            )
            neg_ids: list[str] = [doc_id for doc_id, _ in selected]
            neg_scores: list[float] = [score for _, score in selected]

            output_doc_ids: list[str] = [pos_id] + neg_ids
            output_labels: list[float] = [1.0] + [0.0] * len(neg_ids)
            output_scores: list[float] = [pos_score] + neg_scores
            joined_rows.append(
                {
                    "query_id": qid,
                    "doc_ids": output_doc_ids,
                    "labels": output_labels,
                    "scores": output_scores,
                }
            )
            if log_every > 0 and (idx + 1) % log_every == 0:
                log_if_rank_zero(
                    logger,
                    "Joined "
                    f"{idx + 1:,} base rows (output={len(joined_rows):,}).",
                )

        return Dataset.from_list(joined_rows)
