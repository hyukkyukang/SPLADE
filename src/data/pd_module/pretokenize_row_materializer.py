from typing import Any

import torch

from src.data.dataclass import MetaItem, TrainingDataItem
from src.data.pd_module.utils import (
    RerankInputs,
    build_rerank_inputs,
    build_rerank_inputs_from_cache,
)


class TrainingRowMaterializer:
    """Build a TrainingDataItem from a dataset row index."""

    def __init__(self, *, owner: Any) -> None:
        self._owner: Any = owner

    def materialize(self, idx: int) -> TrainingDataItem:
        owner: Any = self._owner
        data_idx: int = int(idx)
        meta_item: MetaItem = owner._build_meta_item(data_idx)
        pretokenize_enabled: bool = bool(owner._pretokenize_enabled)
        use_streaming_cache: bool = bool(owner._use_streaming_cache)
        if pretokenize_enabled:
            if not owner._cache_ready:
                raise RuntimeError(
                    "Pretokenized cache is not ready. Call setup() before iterating."
                )
            if use_streaming_cache:
                owner._ensure_streaming_cache_for_worker()
            inputs: RerankInputs | None = None
            if use_streaming_cache:
                inputs = owner._build_rerank_inputs_from_meta_row_pointers(
                    data_idx=data_idx, meta_item=meta_item
                )
            if inputs is None:
                inputs = build_rerank_inputs_from_cache(
                    dataset=owner.dataset,
                    meta_item=meta_item,
                    query_cache=owner._query_token_cache,
                    doc_cache=owner._doc_token_cache,
                    max_query_length=owner.max_query_length,
                    max_doc_length=owner.max_doc_length,
                    max_padding=owner.max_padding,
                    pad_token_id=int(owner.tokenizer.pad_token_id),
                    allow_runtime_tokenize_fallback=owner._allow_runtime_tokenize_fallback,
                    tokenizer=owner.tokenizer,
                )
        else:
            inputs = build_rerank_inputs(
                dataset=owner.dataset,
                tokenizer=owner.tokenizer,
                meta_item=meta_item,
                max_query_length=owner.max_query_length,
                max_doc_length=owner.max_doc_length,
                max_padding=owner.max_padding,
            )

        total_docs: int = int(inputs.num_pos + inputs.num_neg)
        label_tensor: torch.Tensor = torch.zeros(total_docs, dtype=torch.float)
        if inputs.num_pos:
            label_tensor[: inputs.num_pos] = 1.0
        pos_scores_tensor: torch.Tensor | None = (
            None
            if meta_item.pos_scores is None
            else torch.as_tensor(meta_item.pos_scores, dtype=torch.float)
        )
        neg_scores_tensor: torch.Tensor | None = (
            None
            if meta_item.neg_scores is None
            else torch.as_tensor(meta_item.neg_scores, dtype=torch.float)
        )

        return TrainingDataItem(
            data_idx=data_idx,
            qid=inputs.qid,
            pos_ids=inputs.pos_ids,
            neg_ids=inputs.neg_ids,
            query_text=inputs.query_text,
            doc_texts=inputs.doc_texts,
            query_input_ids=inputs.query_input_ids,
            query_attention_mask=inputs.query_attention_mask,
            doc_input_ids=inputs.doc_input_ids,
            doc_attention_mask=inputs.doc_attention_mask,
            doc_mask=inputs.doc_mask,
            pos_mask=inputs.pos_mask,
            teacher_scores=inputs.teacher_scores,
            labels=label_tensor,
            pos_scores=pos_scores_tensor,
            neg_scores=neg_scores_tensor,
        )
