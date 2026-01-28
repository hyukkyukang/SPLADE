from __future__ import annotations

from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.dataclass import RerankingDataItem, RetrievalDataItem


class RerankingCollator:
    def __init__(
        self,
        pad_token_id: int,
        require_teacher_scores: bool,
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

    def __call__(self, batch: Sequence[RerankingDataItem]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch provided to collator.")

        if self.max_padding:
            max_query_length_opt: int | None = self.max_query_length
            max_doc_length_opt: int | None = self.max_doc_length
            max_docs_opt: int | None = self.max_docs
            if (
                max_query_length_opt is None
                or max_doc_length_opt is None
                or max_docs_opt is None
            ):
                raise ValueError(
                    "max_padding requires max_query_length, max_doc_length, and max_docs."
                )
            if (
                max_query_length_opt <= 0
                or max_doc_length_opt <= 0
                or max_docs_opt <= 0
            ):
                raise ValueError("Fixed padding sizes must be positive integers.")
            max_query_length: int = int(max_query_length_opt)
            max_doc_length: int = int(max_doc_length_opt)
            max_docs: int = int(max_docs_opt)

            batch_size: int = len(batch)
            # Allocate fixed-shape tensors for consistent step input sizes.
            query_input_ids: torch.Tensor = torch.full(
                (batch_size, max_query_length),
                self.pad_token_id,
                dtype=torch.long,
            )
            query_attention_mask: torch.Tensor = torch.zeros(
                (batch_size, max_query_length), dtype=torch.long
            )
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

            for i, item in enumerate(batch):
                query_len: int = min(
                    int(item.query_input_ids.shape[0]), max_query_length
                )
                if query_len > 0:
                    query_input_ids[i, :query_len] = item.query_input_ids[:query_len]
                    query_attention_mask[i, :query_len] = item.query_attention_mask[
                        :query_len
                    ]

                doc_count: int = min(int(item.doc_input_ids.shape[0]), max_docs)
                if doc_count == 0:
                    continue
                doc_len: int = min(int(item.doc_input_ids.shape[1]), max_doc_length)
                doc_input_ids[i, :doc_count, :doc_len] = item.doc_input_ids[
                    :doc_count, :doc_len
                ]
                doc_attention_mask[i, :doc_count, :doc_len] = item.doc_attention_mask[
                    :doc_count, :doc_len
                ]
                doc_mask[i, :doc_count] = item.doc_mask[:doc_count]
                pos_mask[i, :doc_count] = item.pos_mask[:doc_count]
                teacher_scores[i, :doc_count] = item.teacher_scores[:doc_count]
        else:
            query_input_ids: torch.Tensor = pad_sequence(
                [item.query_input_ids for item in batch],
                batch_first=True,
                padding_value=self.pad_token_id,
            )
            query_attention_mask: torch.Tensor = pad_sequence(
                [item.query_attention_mask for item in batch],
                batch_first=True,
                padding_value=0,
            )

            max_docs: int = max(int(item.doc_input_ids.shape[0]) for item in batch)
            max_doc_len: int = max(int(item.doc_input_ids.shape[1]) for item in batch)

            doc_input_ids: torch.Tensor = torch.full(
                (len(batch), max_docs, max_doc_len),
                self.pad_token_id,
                dtype=torch.long,
            )
            doc_attention_mask: torch.Tensor = torch.zeros(
                (len(batch), max_docs, max_doc_len), dtype=torch.long
            )
            doc_mask: torch.Tensor = torch.zeros(
                (len(batch), max_docs), dtype=torch.bool
            )
            pos_mask: torch.Tensor = torch.zeros(
                (len(batch), max_docs), dtype=torch.bool
            )
            teacher_scores: torch.Tensor = torch.full(
                (len(batch), max_docs), float("nan"), dtype=torch.float
            )

            for i, item in enumerate(batch):
                doc_count: int = int(item.doc_input_ids.shape[0])
                if doc_count == 0:
                    continue
                doc_len: int = int(item.doc_input_ids.shape[1])
                doc_input_ids[i, :doc_count, :doc_len] = item.doc_input_ids
                doc_attention_mask[i, :doc_count, :doc_len] = item.doc_attention_mask
                doc_mask[i, :doc_count] = item.doc_mask
                pos_mask[i, :doc_count] = item.pos_mask
                teacher_scores[i, :doc_count] = item.teacher_scores

        if self.require_teacher_scores:
            missing_scores_mask: torch.Tensor = torch.isnan(teacher_scores) & doc_mask
            if missing_scores_mask.any():
                raise ValueError("Missing teacher scores in training batch")

        return {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "doc_input_ids": doc_input_ids,
            "doc_attention_mask": doc_attention_mask,
            "doc_mask": doc_mask,
            "pos_mask": pos_mask,
            "teacher_scores": teacher_scores,
        }


class RetrievalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[RetrievalDataItem]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch provided to collator.")

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

        return {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "qids": [item.qid for item in batch],
            "query_texts": [item.query_text for item in batch],
        }
