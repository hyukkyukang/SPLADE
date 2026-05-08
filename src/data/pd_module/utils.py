from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

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
from src.data.text_prefix import TextPrefix, slice_text_prefix


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
    query_slot_target_ids: torch.Tensor | None = None
    doc_slot_target_ids: torch.Tensor | None = None


_ORDERED_MASK_SLOT_FAMILIES: frozenset[str] = frozenset(
    {
        "ordered_mask_slot_splade",
        "pretrained_diffusion_ordered_mask_slot_splade",
    }
)


def uses_ordered_mask_slot_pooling(model_cfg: DictConfig | None) -> bool:
    if model_cfg is None:
        return False
    family: str = str(model_cfg.get("family", "splade")).strip().lower()
    return family in _ORDERED_MASK_SLOT_FAMILIES


def resolve_num_mask_slots(model_cfg: DictConfig | None) -> int:
    if model_cfg is None:
        return 0
    return max(int(model_cfg.get("num_mask_slots", 0)), 0)


def resolve_mask_slot_ignore_index(training_cfg: DictConfig | None) -> int:
    if training_cfg is None:
        return -100
    ordered_cfg: DictConfig | None = training_cfg.get("ordered_mask_slots")
    if ordered_cfg is None:
        return -100
    return int(ordered_cfg.get("ignore_index", -100))


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
    fast_truncate_chars_per_token: int | None = None,
    fast_truncate_min_chars: int = 4096,
    prefix_builder: Callable[[int], TextPrefix] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_padding: str | bool = _resolve_padding(
        max_padding=max_padding, padding=padding
    )
    if (
        fast_truncate_chars_per_token is None
        or int(fast_truncate_chars_per_token) <= 0
        or int(max_length) <= 0
    ):
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

    char_budget: int = max(
        int(fast_truncate_min_chars),
        int(max_length) * int(fast_truncate_chars_per_token),
    )
    full_text: str = str(text)
    while True:
        prefix: TextPrefix
        if prefix_builder is None:
            prefix = slice_text_prefix(full_text, char_budget=char_budget)
        else:
            prefix = prefix_builder(int(char_budget))
        tokens = tokenizer(
            prefix.text,
            padding=resolved_padding,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        if not prefix.truncated or int(input_ids.shape[0]) >= int(max_length):
            return input_ids, attention_mask
        next_budget: int = max(int(char_budget) * 2, int(char_budget) + 1)
        if next_budget == char_budget:
            return input_ids, attention_mask
        char_budget = next_budget


def _resolve_mask_token_id(
    tokenizer: PreTrainedTokenizerBase,
    model_cfg: DictConfig | None,
) -> int:
    configured_value: Any = None if model_cfg is None else model_cfg.get("mask_token_id")
    if configured_value is not None:
        token_id: int = int(configured_value)
        if token_id >= 0:
            return token_id
    mask_token_id: int | None = tokenizer.mask_token_id
    if mask_token_id is None or int(mask_token_id) < 0:
        raise ValueError(
            "Ordered mask-slot models require a valid mask token id from the tokenizer "
            "or model.mask_token_id."
        )
    return int(mask_token_id)


def _tokenize_texts_with_mask_slots(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    max_length: int,
    num_mask_slots: int,
    max_padding: bool | None,
    model_cfg: DictConfig | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_mask_slots <= 0:
        raise ValueError("num_mask_slots must be positive for ordered mask-slot inputs.")
    if max_length <= num_mask_slots:
        raise ValueError(
            "max_length must be greater than num_mask_slots for ordered mask-slot inputs."
        )

    text_budget: int = int(max_length) - int(num_mask_slots)
    mask_token_id: int = _resolve_mask_token_id(tokenizer, model_cfg)
    encoded: dict[str, list[list[int]]] = tokenizer(
        list(texts),
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=text_budget,
        return_attention_mask=False,
    )
    input_id_rows: list[list[int]] = encoded["input_ids"]
    resolved_max_length: int
    if max_padding:
        resolved_max_length = int(max_length)
    else:
        resolved_max_length = 0
        row_ids: list[int]
        for row_ids in input_id_rows:
            resolved_max_length = max(
                resolved_max_length,
                min(len(row_ids) + int(num_mask_slots), int(max_length)),
            )

    padded_input_ids: list[torch.Tensor] = []
    padded_attention_masks: list[torch.Tensor] = []
    padded_pooling_masks: list[torch.Tensor] = []
    row_input_ids: list[int]
    for row_input_ids in input_id_rows:
        active_ids: list[int] = list(row_input_ids) + ([mask_token_id] * int(num_mask_slots))
        if len(active_ids) > resolved_max_length:
            active_ids = active_ids[:resolved_max_length]
        active_length: int = len(active_ids)
        if active_length < int(num_mask_slots):
            raise ValueError(
                "Mask-slot tokenization produced fewer active slots than requested."
            )
        pad_length: int = max(resolved_max_length - active_length, 0)
        pad_token_id: int = int(tokenizer.pad_token_id)
        input_ids = torch.tensor(
            active_ids + ([pad_token_id] * pad_length),
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            ([1] * active_length) + ([0] * pad_length),
            dtype=torch.long,
        )
        pooling_mask = torch.zeros(resolved_max_length, dtype=torch.long)
        slot_start: int = active_length - int(num_mask_slots)
        pooling_mask[slot_start:active_length] = 1
        padded_input_ids.append(input_ids)
        padded_attention_masks.append(attention_mask)
        padded_pooling_masks.append(pooling_mask)

    return (
        torch.stack(padded_input_ids, dim=0),
        torch.stack(padded_attention_masks, dim=0),
        torch.stack(padded_pooling_masks, dim=0),
    )


def tokenize_text_with_mask_slots(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_length: int,
    num_mask_slots: int,
    max_padding: bool | None,
    model_cfg: DictConfig | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, pooling_mask = _tokenize_texts_with_mask_slots(
        tokenizer,
        [text],
        max_length=max_length,
        num_mask_slots=num_mask_slots,
        max_padding=max_padding,
        model_cfg=model_cfg,
    )
    return input_ids.squeeze(0), attention_mask.squeeze(0), pooling_mask.squeeze(0)


def tokenize_docs_with_mask_slots(
    tokenizer: PreTrainedTokenizerBase,
    docs: Iterable[str],
    *,
    max_length: int,
    num_mask_slots: int,
    max_padding: bool | None,
    model_cfg: DictConfig | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    docs_list: list[str] = list(docs)
    if not docs_list:
        empty_shape: tuple[int, int] = (0, int(max_length))
        empty_ids: torch.Tensor = torch.empty(empty_shape, dtype=torch.long)
        empty_mask: torch.Tensor = torch.empty(empty_shape, dtype=torch.long)
        empty_pooling_mask: torch.Tensor = torch.empty(empty_shape, dtype=torch.long)
        return empty_ids, empty_mask, empty_pooling_mask
    return _tokenize_texts_with_mask_slots(
        tokenizer,
        docs_list,
        max_length=max_length,
        num_mask_slots=num_mask_slots,
        max_padding=max_padding,
        model_cfg=model_cfg,
    )


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
    term_supervision: Any | None = None,
    term_supervision_ignore_index: int = -100,
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
    use_mask_slots: bool = uses_ordered_mask_slot_pooling(model_cfg)
    num_mask_slots: int = resolve_num_mask_slots(model_cfg)

    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    query_pooling_mask: torch.Tensor
    if use_mask_slots:
        query_input_ids, query_attention_mask, query_pooling_mask = (
            tokenize_text_with_mask_slots(
                tokenizer,
                query_text,
                max_length=max_query_length,
                num_mask_slots=num_mask_slots,
                max_padding=max_padding,
                model_cfg=model_cfg,
            )
        )
    else:
        query_input_ids, query_attention_mask = tokenize_text(
            tokenizer,
            query_text,
            max_length=max_query_length,
            max_padding=max_padding,
        )
        query_pooling_mask = build_query_pooling_mask(
            query_input_ids,
            query_attention_mask,
            tokenizer,
            model_cfg,
        )
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    doc_pooling_mask: torch.Tensor
    if use_mask_slots:
        doc_input_ids, doc_attention_mask, doc_pooling_mask = (
            tokenize_docs_with_mask_slots(
                tokenizer,
                doc_texts,
                max_length=max_doc_length,
                num_mask_slots=num_mask_slots,
                max_padding=max_padding,
                model_cfg=model_cfg,
            )
        )
    else:
        doc_input_ids, doc_attention_mask = tokenize_docs(
            tokenizer,
            doc_texts,
            max_length=max_doc_length,
            max_padding=max_padding,
        )
        doc_pooling_mask = build_doc_pooling_mask(
            doc_attention_mask,
            model_cfg,
        )

    doc_mask: torch.Tensor
    pos_mask: torch.Tensor
    doc_mask, pos_mask = build_doc_masks(len(doc_texts), num_pos)
    teacher_scores: torch.Tensor = build_teacher_scores(
        meta_item.pos_scores, meta_item.neg_scores, num_pos=num_pos, num_neg=num_neg
    )
    query_slot_target_ids: torch.Tensor | None = None
    doc_slot_target_ids: torch.Tensor | None = None
    if use_mask_slots and term_supervision is not None:
        query_slot_target_ids = term_supervision.top_k_doc_target_ids(
            pos_texts[0] if num_pos > 0 else "",
            k=num_mask_slots,
            ignore_index=term_supervision_ignore_index,
        )
        query_term_target_ids: torch.Tensor = term_supervision.top_k_query_target_ids(
            raw_query_text,
            k=num_mask_slots,
            ignore_index=term_supervision_ignore_index,
        )
        total_docs: int = len(doc_texts)
        doc_slot_target_ids = torch.full(
            (total_docs, num_mask_slots),
            int(term_supervision_ignore_index),
            dtype=torch.long,
        )
        if num_pos > 0:
            doc_slot_target_ids[:num_pos] = query_term_target_ids.unsqueeze(0).expand(
                num_pos, -1
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
        query_slot_target_ids=query_slot_target_ids,
        doc_slot_target_ids=doc_slot_target_ids,
    )
