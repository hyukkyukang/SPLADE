from typing import Any

from datasets import Dataset

from src.data.dataclass import MetaItem
from src.data.dataset.base import BaseDataset


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values: list[Any] = list(value)
    else:
        values = [value]
    normalized: list[str] = []
    item: Any
    for item in values:
        text: str = _normalize_text(item)
        if text:
            normalized.append(text)
    return normalized


def _normalize_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values: list[Any] = list(value)
    else:
        values = [value]
    normalized: list[int] = []
    item: Any
    for item in values:
        try:
            normalized.append(int(item))
        except (TypeError, ValueError):
            continue
    return normalized


class Patent10KHardNegativesDataset(BaseDataset):
    """Patent chunk retrieval dataset with inline positives and hard negatives."""

    provides_query_texts_inline: bool = True
    provides_doc_texts_inline: bool = True

    def _resolve_meta_dataset(self) -> Dataset:
        if self.hf_name is None:
            raise ValueError("hf_name must be set for HuggingFace datasets.")
        meta_dataset: Dataset = self._load_hf_dataset(
            hf_name=self.hf_name,
            hf_subset=self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
            data_files=self.hf_data_files,
        )
        return self._apply_hf_sample_window(meta_dataset)

    def _resolve_qid(self, row: dict[str, Any], index: int) -> str:
        qid: str = _normalize_text(row.get("query_id"))
        if qid:
            return qid
        return str(index)

    def _extract_negative_entries(
        self, row: dict[str, Any]
    ) -> list[tuple[str, str, int]]:
        negative_ids: list[str] = _normalize_str_list(row.get("hard_negative_node_ids"))
        negative_texts: list[str] = _normalize_str_list(row.get("hard_negative_texts"))
        negative_ranks: list[int] = _normalize_int_list(row.get("hard_negative_ranks"))
        if not (
            len(negative_ids) == len(negative_texts) == len(negative_ranks)
        ):
            raise ValueError(
                "Patent-10k row has mismatched hard-negative ids/texts/ranks lengths."
            )

        positive_id: str = _normalize_text(row.get("positive_node_id"))
        seen_negative_ids: set[str] = set()
        entries: list[tuple[str, str, int]] = []
        negative_id: str
        negative_text: str
        negative_rank: int
        for negative_id, negative_text, negative_rank in zip(
            negative_ids, negative_texts, negative_ranks
        ):
            if (
                not negative_id
                or not negative_text
                or negative_id == positive_id
                or negative_id in seen_negative_ids
            ):
                continue
            seen_negative_ids.add(negative_id)
            entries.append((negative_id, negative_text, int(negative_rank)))
        return entries

    def _row_to_meta_item(
        self,
        row: dict[str, Any],
        index: int,
        *,
        num_positives: int,
        num_negatives: int,
        rng: Any,
    ) -> MetaItem:
        qid: str = self._resolve_qid(row, index)
        query_text: str = _normalize_text(row.get("query_text"))
        positive_id: str = _normalize_text(row.get("positive_node_id"))
        positive_text: str = _normalize_text(row.get("positive_text"))
        if not query_text:
            raise ValueError(f"Patent-10k row is missing query_text for qid={qid!r}.")
        if not positive_id:
            raise ValueError(
                f"Patent-10k row is missing positive_node_id for qid={qid!r}."
            )
        if not positive_text:
            raise ValueError(
                f"Patent-10k row is missing positive_text for qid={qid!r}."
            )

        negative_entries: list[tuple[str, str, int]] = self._extract_negative_entries(row)
        resolved_num_negatives: int = max(int(num_negatives), 0)
        if resolved_num_negatives > 0 and not negative_entries:
            raise ValueError(
                f"Patent-10k row is missing usable hard negatives for qid={qid!r}."
            )

        negative_score_pairs: list[tuple[str, float | None]] = [
            (negative_id, -float(negative_rank))
            for negative_id, _negative_text, negative_rank in negative_entries
        ]
        selected_negative_ids: list[str] = self._select_negative_ids(
            negative_score_pairs,
            num_negatives=resolved_num_negatives,
            rng=rng,
        )
        negative_text_by_id: dict[str, str] = {
            negative_id: negative_text
            for negative_id, negative_text, _negative_rank in negative_entries
        }
        selected_negative_texts: list[str] = [
            negative_text_by_id[negative_id] for negative_id in selected_negative_ids
        ]

        resolved_num_positives: int = max(int(num_positives), 0)
        pos_ids: list[str] = [positive_id] if resolved_num_positives > 0 else []
        pos_texts: list[str] = [positive_text] if resolved_num_positives > 0 else []
        return MetaItem(
            qid=qid,
            pos_ids=pos_ids,
            neg_ids=selected_negative_ids,
            pos_scores=None,
            neg_scores=None,
            query_text=query_text,
            pos_texts=pos_texts,
            neg_texts=selected_negative_texts,
        )
