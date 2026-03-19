from dataclasses import dataclass
from typing import Iterable

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.lens_formatting import (
    build_doc_pooling_mask,
    build_query_pooling_mask,
    format_query_text,
)
from src.data.dataclass import MetaItem
from src.data.dataset import BaseDataset


@dataclass(frozen=True)
class RerankInputs:
    """Shared tensors and metadata for train/rerank batches."""

    qid: str
    pos_ids: list[str]
    neg_ids: list[str]
    query_text: str
    doc_texts: list[str]
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    query_pooling_mask: torch.Tensor
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    doc_pooling_mask: torch.Tensor
    doc_mask: torch.Tensor
    pos_mask: torch.Tensor
    teacher_scores: torch.Tensor
    num_pos: int
    num_neg: int


def _resolve_padding(
    *,
    max_padding: bool | None = None,
    padding: str | bool | None = None,
) -> str | bool:
    if padding is not None:
        return padding
    if max_padding:
        return "max_length"
    return True


def tokenize_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_length: int,
    max_padding: bool | None = None,
    padding: str | bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_padding: str | bool = _resolve_padding(
        max_padding=max_padding, padding=padding
    )
    tokens: dict[str, torch.Tensor] = tokenizer(
        text,
        padding=resolved_padding,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    input_ids: torch.Tensor = tokens["input_ids"].squeeze(0)
    attention_mask: torch.Tensor = tokens["attention_mask"].squeeze(0)
    return input_ids, attention_mask


def tokenize_text_windows(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_length: int,
    max_padding: bool | None = None,
    overlap_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_padding: str | bool = _resolve_padding(max_padding=max_padding)
    overlap: int = max(0, int(overlap_tokens))
    tokens: dict[str, torch.Tensor] = tokenizer(
        text,
        padding=resolved_padding,
        truncation=True,
        max_length=int(max_length),
        stride=overlap,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    input_ids: torch.Tensor = tokens["input_ids"]
    attention_mask: torch.Tensor = tokens["attention_mask"]
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask


def tokenize_docs(
    tokenizer: PreTrainedTokenizerBase,
    docs: Iterable[str],
    *,
    max_length: int,
    max_padding: bool | None = None,
    padding: str | bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_padding: str | bool = _resolve_padding(
        max_padding=max_padding, padding=padding
    )
    docs_list: list[str] = list(docs)
    if not docs_list:
        empty_ids: torch.Tensor = torch.empty((0, max_length), dtype=torch.long)
        empty_mask: torch.Tensor = torch.empty((0, max_length), dtype=torch.long)
        return empty_ids, empty_mask
    tokens: dict[str, torch.Tensor] = tokenizer(
        docs_list,
        padding=resolved_padding,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    input_ids: torch.Tensor = tokens["input_ids"]
    attention_mask: torch.Tensor = tokens["attention_mask"]
    return input_ids, attention_mask


def build_doc_masks(num_docs: int, num_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    doc_mask: torch.Tensor = torch.zeros(num_docs, dtype=torch.bool)
    if num_docs:
        doc_mask[:] = True
    pos_mask: torch.Tensor = torch.zeros(num_docs, dtype=torch.bool)
    if num_pos:
        pos_mask[:num_pos] = True
    return doc_mask, pos_mask


def build_teacher_scores(
    pos_scores: list[float] | None,
    neg_scores: list[float] | None,
    *,
    num_pos: int,
    num_neg: int,
) -> torch.Tensor:
    pos_scores_list: list[float] = (
        list(pos_scores) if pos_scores is not None else [float("nan")] * num_pos
    )
    neg_scores_list: list[float] = (
        list(neg_scores) if neg_scores is not None else [float("nan")] * num_neg
    )
    teacher_scores_list: list[float] = pos_scores_list + neg_scores_list
    return torch.tensor(teacher_scores_list, dtype=torch.float)


def build_rerank_inputs(
    dataset: BaseDataset,
    tokenizer: PreTrainedTokenizerBase,
    meta_item: MetaItem,
    *,
    model_cfg: DictConfig | None,
    max_query_length: int,
    max_doc_length: int,
    max_padding: bool,
) -> RerankInputs:
    raw_query_text: str = dataset.resolve_query_text(meta_item)
    query_text: str = format_query_text(raw_query_text, model_cfg)
    pos_ids: list[str] = meta_item.pos_ids
    neg_ids: list[str] = meta_item.neg_ids
    pos_texts: list[str] = dataset.resolve_doc_texts(pos_ids, meta_item.pos_texts)
    neg_texts: list[str] = dataset.resolve_doc_texts(neg_ids, meta_item.neg_texts)

    doc_texts: list[str] = pos_texts + neg_texts
    num_pos: int = len(pos_texts)
    num_neg: int = len(neg_texts)

    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    query_input_ids, query_attention_mask = tokenize_text(
        tokenizer,
        query_text,
        max_length=max_query_length,
        max_padding=max_padding,
    )
    query_pooling_mask: torch.Tensor = build_query_pooling_mask(
        query_input_ids,
        query_attention_mask,
        tokenizer,
        model_cfg,
    )
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    doc_input_ids, doc_attention_mask = tokenize_docs(
        tokenizer,
        doc_texts,
        max_length=max_doc_length,
        max_padding=max_padding,
    )
    doc_pooling_mask: torch.Tensor = build_doc_pooling_mask(
        doc_attention_mask,
        model_cfg,
    )

    doc_mask: torch.Tensor
    pos_mask: torch.Tensor
    doc_mask, pos_mask = build_doc_masks(len(doc_texts), num_pos)
    teacher_scores: torch.Tensor = build_teacher_scores(
        meta_item.pos_scores, meta_item.neg_scores, num_pos=num_pos, num_neg=num_neg
    )

    return RerankInputs(
        qid=meta_item.qid,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        query_text=query_text,
        doc_texts=doc_texts,
        query_input_ids=query_input_ids,
        query_attention_mask=query_attention_mask,
        query_pooling_mask=query_pooling_mask,
        doc_input_ids=doc_input_ids,
        doc_attention_mask=doc_attention_mask,
        doc_pooling_mask=doc_pooling_mask,
        doc_mask=doc_mask,
        pos_mask=pos_mask,
        teacher_scores=teacher_scores,
        num_pos=num_pos,
        num_neg=num_neg,
    )
