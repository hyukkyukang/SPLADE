from collections.abc import Mapping
import random
from typing import Any

from datasets import Dataset

from src.data.dataclass import MetaItem
from src.data.dataset.base import BaseDataset
from src.data.dataset.utils import sample_items


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_patent_title_abstract_claims(row: Mapping[str, Any]) -> str:
    title: str = _normalize_text(row.get("title"))
    abstract: str = _normalize_text(row.get("abstract"))
    claims: str = _normalize_text(row.get("claims"))
    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}.")
    if abstract:
        parts.append(f"Abstract: {abstract}.")
    if claims:
        parts.append(f"Claims: {claims}")
    return " ".join(parts).strip()


class PatentUsInBatchDataset(BaseDataset):
    """Patent stage-1 dataset with inline queries and corpus-resolved positives."""

    provides_query_texts_inline: bool = True
    provides_doc_texts_inline: bool = False

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
        rng: random.Random,
    ) -> MetaItem:
        qid: str = _normalize_text(row.get("query_id")) or str(index)
        query_text: str = _normalize_text(row.get("query_text"))
        if not query_text:
            raise ValueError(f"Patent in-batch row is missing query_text for qid={qid!r}.")

        pos_doc_ids_all: list[str] = [
            _normalize_text(doc_id) for doc_id in row.get("pos_doc_ids") or [] if doc_id
        ]
        if not pos_doc_ids_all:
            raise ValueError(f"Patent in-batch row is missing pos_doc_ids for qid={qid!r}.")
        pos_ids: list[str] = sample_items(pos_doc_ids_all, int(num_positives), rng)

        neg_doc_ids_all: list[str] = [
            _normalize_text(doc_id) for doc_id in row.get("neg_doc_ids") or [] if doc_id
        ]
        neg_ids: list[str] = sample_items(neg_doc_ids_all, int(num_negatives), rng)

        return MetaItem(
            qid=qid,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            pos_scores=None,
            neg_scores=None,
            query_text=query_text,
            pos_texts=None,
            neg_texts=None,
        )

    def _corpus_text_from_row(self, row: Mapping[str, Any]) -> str:
        return format_patent_title_abstract_claims(row)
