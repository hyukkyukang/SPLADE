from typing import Any

from datasets import Dataset

from src.data.dataclass import MetaItem
from src.data.dataset.base import BaseDataset


def _to_doc_ids(value: Any) -> list[str]:
    if value is None:
        return []
    values: list[Any]
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    doc_ids: list[str] = []
    raw_id: Any
    for raw_id in values:
        doc_id: str = str(raw_id).strip()
        if doc_id:
            doc_ids.append(doc_id)
    return doc_ids


class MSMARCODevSmallNegativesDataset(BaseDataset):
    """MS MARCO dev small hard negatives dataset for reranking-style validation."""

    # --- Protected methods ---
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

    def _row_to_meta_item(
        self,
        row: dict[str, Any],
        index: int,
        *,
        num_positives: int,
        num_negatives: int,
        rng: Any,
    ) -> MetaItem:
        _ = rng
        qid_value: Any | None = row.get("qid")
        if qid_value is None:
            qid_value = row.get("query_id")
        qid: str = str(index) if qid_value is None else str(qid_value).strip()
        if not qid:
            qid = str(index)

        pos_ids_all: list[str] = _to_doc_ids(row.get("pos"))
        if not pos_ids_all:
            raise ValueError(
                "MSMARCO dev negatives row is missing positive ids "
                f"for qid={qid!r}."
            )
        neg_ids_all: list[str] = _to_doc_ids(row.get("neg"))
        if num_negatives > 0 and not neg_ids_all:
            raise ValueError(
                "MSMARCO dev negatives row is missing negative ids "
                f"for qid={qid!r}."
            )

        resolved_num_positives: int = max(int(num_positives), 0)
        resolved_num_negatives: int = max(int(num_negatives), 0)
        # Keep deterministic candidate order from the dataset.
        pos_ids: list[str] = (
            pos_ids_all[:resolved_num_positives] if resolved_num_positives > 0 else []
        )
        neg_ids: list[str] = (
            neg_ids_all[:resolved_num_negatives] if resolved_num_negatives > 0 else []
        )

        return MetaItem(
            qid=qid,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            pos_scores=None,
            neg_scores=None,
            query_text=None,
            pos_texts=None,
            neg_texts=None,
        )
