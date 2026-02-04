from typing import Any

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.collator import UniversalCollator
from src.data.dataclass import MetaItem, TrainingDataItem
from src.data.pd_module import PDModule


class TrainingPDModule(PDModule):
    """Training PyTorch datasets module."""

    # --- Special methods ---
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        *,
        seed: int,
        load_teacher_scores: bool | None = None,
        require_teacher_scores: bool | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            tokenizer=tokenizer,
            seed=seed,
            load_teacher_scores=load_teacher_scores,
            require_teacher_scores=require_teacher_scores,
        )
        self._collator: UniversalCollator | None = None

    def __getitem__(self, idx: int) -> TrainingDataItem:
        meta_item: MetaItem = self._build_meta_item(int(idx))
        query_text: str = self.dataset.resolve_query_text(meta_item)
        pos_ids: list[str] = meta_item.pos_ids
        neg_ids: list[str] = meta_item.neg_ids
        pos_texts: list[str] = self.dataset.resolve_doc_texts(
            pos_ids, meta_item.pos_texts
        )
        neg_texts: list[str] = self.dataset.resolve_doc_texts(
            neg_ids, meta_item.neg_texts
        )

        doc_texts: list[str] = pos_texts + neg_texts
        doc_ids: list[str] = pos_ids + neg_ids

        query_input_ids: torch.Tensor
        query_attention_mask: torch.Tensor
        query_input_ids, query_attention_mask = self._tokenize_text(
            query_text, max_length=self.max_query_length
        )
        doc_input_ids: torch.Tensor
        doc_attention_mask: torch.Tensor
        doc_input_ids, doc_attention_mask = self._tokenize_docs(
            doc_texts, max_length=self.max_doc_length
        )

        doc_mask: torch.Tensor = torch.zeros(len(doc_texts), dtype=torch.bool)
        if doc_texts:
            doc_mask[:] = True
        pos_mask: torch.Tensor = torch.zeros(len(doc_texts), dtype=torch.bool)
        if pos_texts:
            pos_mask[: len(pos_texts)] = True

        pos_scores_list: list[float] = (
            meta_item.pos_scores
            if meta_item.pos_scores is not None
            else [float("nan")] * len(pos_ids)
        )
        neg_scores_list: list[float] = (
            meta_item.neg_scores
            if meta_item.neg_scores is not None
            else [float("nan")] * len(neg_ids)
        )
        teacher_scores_list: list[float] = pos_scores_list + neg_scores_list
        teacher_scores: torch.Tensor = torch.tensor(
            teacher_scores_list, dtype=torch.float
        )

        labels: list[float] = [1.0] * len(pos_texts) + [0.0] * len(neg_texts)
        label_tensor: torch.Tensor = torch.tensor(labels, dtype=torch.float)
        pos_scores_tensor: torch.Tensor | None = (
            None
            if meta_item.pos_scores is None
            else torch.tensor(meta_item.pos_scores, dtype=torch.float)
        )
        neg_scores_tensor: torch.Tensor | None = (
            None
            if meta_item.neg_scores is None
            else torch.tensor(meta_item.neg_scores, dtype=torch.float)
        )

        return TrainingDataItem(
            data_idx=int(idx),
            qid=meta_item.qid,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            query_text=query_text,
            doc_texts=doc_texts,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            doc_mask=doc_mask,
            pos_mask=pos_mask,
            teacher_scores=teacher_scores,
            labels=label_tensor,
            pos_scores=pos_scores_tensor,
            neg_scores=neg_scores_tensor,
        )

    # --- Property methods ---
    @property
    def collator(self) -> UniversalCollator:
        if self._collator is None:
            max_docs: int = int(self.num_positives + self.num_negatives)
            self._collator = UniversalCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                require_teacher_scores=self.require_teacher_scores,
                max_padding=self.max_padding,
                max_query_length=self.max_query_length,
                max_doc_length=self.max_doc_length,
                max_docs=max_docs,
            )
        return self._collator
