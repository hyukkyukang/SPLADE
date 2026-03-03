from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from src.data.dataclass import MetaItem
from src.data.dataset import BaseDataset
from src.data.pd_module.pretokenize import make_id_key, make_text_key


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
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
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
    max_query_length: int,
    max_doc_length: int,
    max_padding: bool,
) -> RerankInputs:
    query_text: str = dataset.resolve_query_text(meta_item)
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
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    doc_input_ids, doc_attention_mask = tokenize_docs(
        tokenizer,
        doc_texts,
        max_length=max_doc_length,
        max_padding=max_padding,
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
        doc_input_ids=doc_input_ids,
        doc_attention_mask=doc_attention_mask,
        doc_mask=doc_mask,
        pos_mask=pos_mask,
        teacher_scores=teacher_scores,
        num_pos=num_pos,
        num_neg=num_neg,
    )


def _resolve_cached_tokens(
    *,
    cache: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    id_value: str,
    text_value: str | None,
    allow_text_key_lookup: bool,
    cache_name: str,
    allow_runtime_tokenize_fallback: bool,
    tokenizer: PreTrainedTokenizerBase | None,
    max_length: int,
    max_padding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    id_key: str = make_id_key(id_value)
    tokens: tuple[torch.Tensor, torch.Tensor] | None = cache.get(id_key)
    if tokens is None and allow_text_key_lookup and text_value is not None:
        text_key: str = make_text_key(text_value)
        tokens = cache.get(text_key)
    if tokens is not None:
        return tokens
    if (
        not allow_runtime_tokenize_fallback
        or tokenizer is None
        or text_value is None
    ):
        raise KeyError(
            f"Missing pretokenized {cache_name} cache entry for id={id_value!r}."
        )
    return tokenize_text(
        tokenizer,
        text_value,
        max_length=max_length,
        max_padding=max_padding,
    )


def _can_skip_text_resolution(
    *,
    meta_item: MetaItem,
    doc_ids: list[str],
    allow_runtime_tokenize_fallback: bool,
) -> bool:
    if allow_runtime_tokenize_fallback:
        return False
    if (
        meta_item.query_text is not None
        or meta_item.pos_texts is not None
        or meta_item.neg_texts is not None
    ):
        return False
    if not str(meta_item.qid).strip():
        return False
    doc_id: str
    for doc_id in doc_ids:
        if not str(doc_id).strip():
            return False
    return True


def build_rerank_inputs_from_cache(
    dataset: BaseDataset,
    meta_item: MetaItem,
    *,
    query_cache: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    doc_cache: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    max_query_length: int,
    max_doc_length: int,
    max_padding: bool,
    pad_token_id: int,
    allow_runtime_tokenize_fallback: bool = False,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> RerankInputs:
    pos_ids: list[str] = meta_item.pos_ids
    neg_ids: list[str] = meta_item.neg_ids
    doc_ids: list[str] = pos_ids + neg_ids
    num_pos: int = len(pos_ids)
    num_neg: int = len(neg_ids)

    skip_text_resolution: bool = _can_skip_text_resolution(
        meta_item=meta_item,
        doc_ids=doc_ids,
        allow_runtime_tokenize_fallback=allow_runtime_tokenize_fallback,
    )
    if skip_text_resolution:
        query_text: str = ""
        doc_texts: list[str] = [""] * len(doc_ids)
    else:
        query_text = dataset.resolve_query_text(meta_item)
        pos_texts: list[str] = dataset.resolve_doc_texts(pos_ids, meta_item.pos_texts)
        neg_texts: list[str] = dataset.resolve_doc_texts(neg_ids, meta_item.neg_texts)
        doc_texts = pos_texts + neg_texts
        if len(doc_texts) < len(doc_ids):
            doc_texts.extend([""] * (len(doc_ids) - len(doc_texts)))
        elif len(doc_texts) > len(doc_ids):
            doc_texts = doc_texts[: len(doc_ids)]

    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    query_id_value: str = str(meta_item.qid).strip()
    query_lookup_id: str = query_id_value if query_id_value else query_text
    query_lookup_text: str | None = query_text if query_text else None
    query_input_ids, query_attention_mask = _resolve_cached_tokens(
        cache=query_cache,
        id_value=query_lookup_id,
        text_value=query_lookup_text,
        allow_text_key_lookup=bool(query_lookup_text and not query_id_value),
        cache_name="query",
        allow_runtime_tokenize_fallback=allow_runtime_tokenize_fallback,
        tokenizer=tokenizer,
        max_length=max_query_length,
        max_padding=max_padding,
    )

    doc_input_rows: list[torch.Tensor] = []
    doc_mask_rows: list[torch.Tensor] = []
    doc_idx: int
    for doc_idx, doc_id in enumerate(doc_ids):
        doc_text: str = doc_texts[doc_idx] if doc_idx < len(doc_texts) else ""
        doc_id_value: str = str(doc_id).strip()
        resolved_id: str = doc_id_value if doc_id_value else doc_text
        doc_lookup_text: str | None = doc_text if doc_text else None
        doc_input_ids_row: torch.Tensor
        doc_attention_mask_row: torch.Tensor
        doc_input_ids_row, doc_attention_mask_row = _resolve_cached_tokens(
            cache=doc_cache,
            id_value=resolved_id,
            text_value=doc_lookup_text,
            allow_text_key_lookup=bool(doc_lookup_text and not doc_id_value),
            cache_name="document",
            allow_runtime_tokenize_fallback=allow_runtime_tokenize_fallback,
            tokenizer=tokenizer,
            max_length=max_doc_length,
            max_padding=max_padding,
        )
        doc_input_rows.append(doc_input_ids_row)
        doc_mask_rows.append(doc_attention_mask_row)

    if not doc_input_rows:
        doc_input_ids = torch.empty((0, max_doc_length), dtype=torch.long)
        doc_attention_mask = torch.empty((0, max_doc_length), dtype=torch.long)
    elif max_padding:
        doc_input_ids = torch.stack(doc_input_rows, dim=0)
        doc_attention_mask = torch.stack(doc_mask_rows, dim=0)
    else:
        doc_input_ids = pad_sequence(
            doc_input_rows,
            batch_first=True,
            padding_value=int(pad_token_id),
        )
        doc_attention_mask = pad_sequence(
            doc_mask_rows,
            batch_first=True,
            padding_value=0,
        )

    doc_mask: torch.Tensor
    pos_mask: torch.Tensor
    doc_mask, pos_mask = build_doc_masks(len(doc_ids), num_pos)
    teacher_scores: torch.Tensor = build_teacher_scores(
        meta_item.pos_scores,
        meta_item.neg_scores,
        num_pos=num_pos,
        num_neg=num_neg,
    )

    return RerankInputs(
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
        num_pos=num_pos,
        num_neg=num_neg,
    )
