from __future__ import annotations

from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.dataclass import RerankingDataItem, RetrievalDataItem


class RerankingCollator:
    def __init__(self, pad_token_id: int, require_teacher_scores: bool) -> None:
        self.pad_token_id = pad_token_id
        self.require_teacher_scores = require_teacher_scores

    def __call__(self, batch: Sequence[RerankingDataItem]) -> dict[str, torch.Tensor]:
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

        max_docs = max(item.doc_input_ids.shape[0] for item in batch)
        max_doc_len = max(item.doc_input_ids.shape[1] for item in batch)

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
        teacher_scores = torch.full((len(batch), max_docs), float("nan"))

        for i, item in enumerate(batch):
            doc_count = item.doc_input_ids.shape[0]
            if doc_count == 0:
                continue
            doc_len = item.doc_input_ids.shape[1]
            doc_input_ids[i, :doc_count, :doc_len] = item.doc_input_ids
            doc_attention_mask[i, :doc_count, :doc_len] = item.doc_attention_mask
            doc_mask[i, :doc_count] = item.doc_mask
            pos_mask[i, :doc_count] = item.pos_mask
            teacher_scores[i, :doc_count] = item.teacher_scores

        if self.require_teacher_scores and torch.isnan(teacher_scores).any():
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
