from __future__ import annotations

from typing import Sequence

import torch

from src.dataset.datasets.train import TrainSample


class TrainCollator:
    def __init__(
        self,
        tokenizer,
        max_query_length: int,
        max_doc_length: int,
        require_teacher_scores: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.require_teacher_scores = require_teacher_scores

    def __call__(self, batch: Sequence[TrainSample]) -> dict[str, torch.Tensor]:
        queries = [sample.query for sample in batch]
        docs_per_query = [sample.docs for sample in batch]
        pos_counts = [sample.pos_count for sample in batch]
        teacher_scores = [sample.teacher_scores for sample in batch]

        tokenized_queries = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        )

        doc_texts = [doc for docs in docs_per_query for doc in docs]
        doc_counts = [len(docs) for docs in docs_per_query]

        if doc_texts:
            tokenized_docs = self.tokenizer(
                doc_texts,
                padding=True,
                truncation=True,
                max_length=self.max_doc_length,
                return_tensors="pt",
            )
        else:
            tokenized_docs = {"input_ids": torch.empty(0, 0, dtype=torch.long)}

        max_docs = max(doc_counts) if doc_counts else 0
        doc_seq_len = (
            tokenized_docs["input_ids"].shape[1] if doc_texts else self.max_doc_length
        )

        doc_input_ids = torch.zeros(
            (len(batch), max_docs, doc_seq_len), dtype=torch.long
        )
        doc_attention_mask = torch.zeros(
            (len(batch), max_docs, doc_seq_len), dtype=torch.long
        )
        doc_mask = torch.zeros((len(batch), max_docs), dtype=torch.bool)
        pos_mask = torch.zeros((len(batch), max_docs), dtype=torch.bool)
        teacher_score_tensor = torch.full(
            (len(batch), max_docs), float("nan"), dtype=torch.float
        )

        offset = 0
        for i, count in enumerate(doc_counts):
            if count == 0:
                continue
            doc_input_ids[i, :count] = tokenized_docs["input_ids"][
                offset : offset + count
            ]
            doc_attention_mask[i, :count] = tokenized_docs["attention_mask"][
                offset : offset + count
            ]
            doc_mask[i, :count] = True
            pos_mask[i, : pos_counts[i]] = True
            teacher_score_tensor[i, :count] = torch.tensor(
                teacher_scores[i], dtype=torch.float
            )
            offset += count

        if self.require_teacher_scores and torch.isnan(teacher_score_tensor).any():
            raise ValueError("Missing teacher scores in training batch")

        return {
            "query_input_ids": tokenized_queries["input_ids"],
            "query_attention_mask": tokenized_queries["attention_mask"],
            "doc_input_ids": doc_input_ids,
            "doc_attention_mask": doc_attention_mask,
            "doc_mask": doc_mask,
            "pos_mask": pos_mask,
            "teacher_scores": teacher_score_tensor,
        }
