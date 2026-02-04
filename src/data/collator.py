from typing import Any, Sequence, Type, TypeVar

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.dataclass import (
    EncodingDataItem,
    RerankingDataItem,
    RetrievalDataItem,
    TrainingDataItem,
)

DataItem = EncodingDataItem | RerankingDataItem | RetrievalDataItem | TrainingDataItem
TDataItem = TypeVar("TDataItem", bound=DataItem)


class UniversalCollator:
    """Collate dataclass items into batch tensors/lists."""

    # --- Special methods ---
    def __init__(
        self,
        pad_token_id: int,
        require_teacher_scores: bool = False,
        *,
        max_padding: bool = False,
        max_query_length: int | None = None,
        max_doc_length: int | None = None,
        max_docs: int | None = None,
    ) -> None:
        self.pad_token_id: int = int(pad_token_id)
        self.require_teacher_scores: bool = bool(require_teacher_scores)
        self.max_padding: bool = bool(max_padding)
        self.max_query_length: int | None = (
            None if max_query_length is None else int(max_query_length)
        )
        self.max_doc_length: int | None = (
            None if max_doc_length is None else int(max_doc_length)
        )
        self.max_docs: int | None = None if max_docs is None else int(max_docs)

    # --- Public methods ---
    def __call__(self, batch: Sequence[DataItem]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch provided to collator.")

        first_item: DataItem = batch[0]
        if isinstance(first_item, TrainingDataItem):
            self._ensure_batch_type(batch, TrainingDataItem)
            return self._collate_training(batch)
        if isinstance(first_item, RerankingDataItem):
            self._ensure_batch_type(batch, RerankingDataItem)
            return self._collate_reranking(batch)
        if isinstance(first_item, RetrievalDataItem):
            self._ensure_batch_type(batch, RetrievalDataItem)
            return self._collate_retrieval(batch)
        if isinstance(first_item, EncodingDataItem):
            self._ensure_batch_type(batch, EncodingDataItem)
            return self._collate_encoding(batch)
        raise TypeError(f"Unsupported batch item type: {type(first_item)}")

    # --- Protected methods ---
    def _ensure_batch_type(
        self, batch: Sequence[DataItem], expected_type: Type[TDataItem]
    ) -> None:
        if not all(isinstance(item, expected_type) for item in batch):
            raise TypeError("Mixed batch types are not supported.")

    def _require_fixed_query_length(self) -> int:
        max_query_length: int | None = self.max_query_length
        if max_query_length is None or max_query_length <= 0:
            raise ValueError("max_padding requires a positive max_query_length.")
        return int(max_query_length)

    def _require_fixed_doc_sizes(self) -> tuple[int, int]:
        max_doc_length: int | None = self.max_doc_length
        max_docs: int | None = self.max_docs
        if max_doc_length is None or max_docs is None:
            raise ValueError(
                "max_padding requires max_doc_length and max_docs for doc batches."
            )
        if max_doc_length <= 0 or max_docs <= 0:
            raise ValueError("Fixed padding sizes must be positive integers.")
        return int(max_doc_length), int(max_docs)

    def _pad_query(
        self, batch: Sequence[RerankingDataItem | RetrievalDataItem]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_padding:
            max_query_length: int = self._require_fixed_query_length()
            batch_size: int = len(batch)
            query_input_ids: torch.Tensor = torch.full(
                (batch_size, max_query_length),
                self.pad_token_id,
                dtype=torch.long,
            )
            query_attention_mask: torch.Tensor = torch.zeros(
                (batch_size, max_query_length), dtype=torch.long
            )
            for batch_idx, item in enumerate(batch):
                query_len: int = min(
                    int(item.query_input_ids.shape[0]), max_query_length
                )
                if query_len == 0:
                    continue
                query_input_ids[batch_idx, :query_len] = item.query_input_ids[
                    :query_len
                ]
                query_attention_mask[batch_idx, :query_len] = item.query_attention_mask[
                    :query_len
                ]
            return query_input_ids, query_attention_mask

        query_input_ids = pad_sequence(
            [item.query_input_ids for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        query_attention_mask = pad_sequence(
            [item.query_attention_mask for item in batch],
            batch_first=True,
            padding_value=0,
        )
        return query_input_ids, query_attention_mask

    def _pad_docs(
        self, batch: Sequence[RerankingDataItem]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.max_padding:
            max_doc_length, max_docs = self._require_fixed_doc_sizes()
            batch_size: int = len(batch)
            doc_input_ids: torch.Tensor = torch.full(
                (batch_size, max_docs, max_doc_length),
                self.pad_token_id,
                dtype=torch.long,
            )
            doc_attention_mask: torch.Tensor = torch.zeros(
                (batch_size, max_docs, max_doc_length), dtype=torch.long
            )
            doc_mask: torch.Tensor = torch.zeros(
                (batch_size, max_docs), dtype=torch.bool
            )
            pos_mask: torch.Tensor = torch.zeros(
                (batch_size, max_docs), dtype=torch.bool
            )
            teacher_scores: torch.Tensor = torch.full(
                (batch_size, max_docs), float("nan"), dtype=torch.float
            )

            for batch_idx, item in enumerate(batch):
                doc_count: int = min(int(item.doc_input_ids.shape[0]), max_docs)
                if doc_count == 0:
                    continue
                doc_len: int = min(int(item.doc_input_ids.shape[1]), max_doc_length)
                doc_input_ids[batch_idx, :doc_count, :doc_len] = item.doc_input_ids[
                    :doc_count, :doc_len
                ]
                doc_attention_mask[batch_idx, :doc_count, :doc_len] = (
                    item.doc_attention_mask[:doc_count, :doc_len]
                )
                doc_mask[batch_idx, :doc_count] = item.doc_mask[:doc_count]
                pos_mask[batch_idx, :doc_count] = item.pos_mask[:doc_count]
                teacher_scores[batch_idx, :doc_count] = item.teacher_scores[:doc_count]
            return doc_input_ids, doc_attention_mask, doc_mask, pos_mask, teacher_scores

        max_docs: int = max(int(item.doc_input_ids.shape[0]) for item in batch)
        max_doc_len: int = max(int(item.doc_input_ids.shape[1]) for item in batch)
        doc_input_ids = torch.full(
            (len(batch), max_docs, max_doc_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        doc_attention_mask = torch.zeros(
            (len(batch), max_docs, max_doc_len), dtype=torch.long
        )
        doc_mask = torch.zeros((len(batch), max_docs), dtype=torch.bool)
        pos_mask = torch.zeros((len(batch), max_docs), dtype=torch.bool)
        teacher_scores = torch.full(
            (len(batch), max_docs), float("nan"), dtype=torch.float
        )

        for batch_idx, item in enumerate(batch):
            doc_count: int = int(item.doc_input_ids.shape[0])
            if doc_count == 0:
                continue
            doc_len: int = int(item.doc_input_ids.shape[1])
            doc_input_ids[batch_idx, :doc_count, :doc_len] = item.doc_input_ids
            doc_attention_mask[batch_idx, :doc_count, :doc_len] = (
                item.doc_attention_mask
            )
            doc_mask[batch_idx, :doc_count] = item.doc_mask
            pos_mask[batch_idx, :doc_count] = item.pos_mask
            teacher_scores[batch_idx, :doc_count] = item.teacher_scores

        return doc_input_ids, doc_attention_mask, doc_mask, pos_mask, teacher_scores

    def _collate_reranking(
        self, batch: Sequence[RerankingDataItem]
    ) -> dict[str, torch.Tensor]:
        query_input_ids: torch.Tensor
        query_attention_mask: torch.Tensor
        query_input_ids, query_attention_mask = self._pad_query(batch)
        (
            doc_input_ids,
            doc_attention_mask,
            doc_mask,
            pos_mask,
            teacher_scores,
        ) = self._pad_docs(batch)

        if self.require_teacher_scores:
            missing_scores_mask: torch.Tensor = torch.isnan(teacher_scores) & doc_mask
            if missing_scores_mask.any():
                raise ValueError("Missing teacher scores in training batch.")

        return {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "doc_input_ids": doc_input_ids,
            "doc_attention_mask": doc_attention_mask,
            "doc_mask": doc_mask,
            "pos_mask": pos_mask,
            "teacher_scores": teacher_scores,
        }

    def _collate_retrieval(self, batch: Sequence[RetrievalDataItem]) -> dict[str, Any]:
        query_input_ids: torch.Tensor
        query_attention_mask: torch.Tensor
        query_input_ids, query_attention_mask = self._pad_query(batch)
        qids: list[str] = [item.qid for item in batch]
        query_texts: list[str] = [item.query_text for item in batch]
        relevance_judgments: list[dict[str, float]] = [
            item.relevance_judgments for item in batch
        ]
        return {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "qid": qids,
            "query_text": query_texts,
            "relevance_judgments": relevance_judgments,
        }

    def _collate_training(
        self, batch: Sequence[TrainingDataItem]
    ) -> dict[str, torch.Tensor | None]:
        batch_dict: dict[str, torch.Tensor | None] = self._collate_reranking(batch)
        labels: torch.Tensor = pad_sequence(
            [item.labels for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
        batch_dict["labels"] = labels

        has_pos_scores: bool = any(item.pos_scores is not None for item in batch)
        has_neg_scores: bool = any(item.neg_scores is not None for item in batch)
        if has_pos_scores or has_neg_scores:
            if any(
                item.pos_scores is None or item.neg_scores is None for item in batch
            ):
                raise ValueError("Mixed presence of pos_scores/neg_scores in batch.")
            pos_scores_items: list[torch.Tensor] = []
            neg_scores_items: list[torch.Tensor] = []
            for item in batch:
                pos_count: int = int(item.pos_mask.sum().item())
                doc_count: int = int(item.doc_mask.sum().item())
                neg_count: int = max(doc_count - pos_count, 0)
                if item.pos_scores.shape[0] != pos_count:
                    raise ValueError("pos_scores length does not match pos_mask.")
                if item.neg_scores.shape[0] != neg_count:
                    raise ValueError("neg_scores length does not match neg_mask.")
                pos_scores_items.append(item.pos_scores)
                neg_scores_items.append(item.neg_scores)
            batch_dict["pos_scores"] = pad_sequence(
                pos_scores_items, batch_first=True, padding_value=float("nan")
            )
            batch_dict["neg_scores"] = pad_sequence(
                neg_scores_items, batch_first=True, padding_value=float("nan")
            )

        return batch_dict

    def _collate_encoding(self, batch: Sequence[EncodingDataItem]) -> dict[str, Any]:
        doc_ids: list[str] = [item.doc_id for item in batch]
        if self.max_padding:
            max_doc_length: int | None = self.max_doc_length
            if max_doc_length is None or max_doc_length <= 0:
                raise ValueError("max_padding requires a positive max_doc_length.")
            batch_size: int = len(batch)
            doc_input_ids: torch.Tensor = torch.full(
                (batch_size, max_doc_length),
                self.pad_token_id,
                dtype=torch.long,
            )
            doc_attention_mask: torch.Tensor = torch.zeros(
                (batch_size, max_doc_length), dtype=torch.long
            )
            for batch_idx, item in enumerate(batch):
                doc_len: int = min(
                    int(item.doc_input_ids.shape[0]), max_doc_length
                )
                if doc_len == 0:
                    continue
                doc_input_ids[batch_idx, :doc_len] = item.doc_input_ids[:doc_len]
                doc_attention_mask[batch_idx, :doc_len] = item.doc_attention_mask[
                    :doc_len
                ]
        else:
            doc_input_ids = pad_sequence(
                [item.doc_input_ids for item in batch],
                batch_first=True,
                padding_value=self.pad_token_id,
            )
            doc_attention_mask = pad_sequence(
                [item.doc_attention_mask for item in batch],
                batch_first=True,
                padding_value=0,
            )
        return {
            "doc_ids": doc_ids,
            "doc_input_ids": doc_input_ids,
            "doc_attention_mask": doc_attention_mask,
        }
