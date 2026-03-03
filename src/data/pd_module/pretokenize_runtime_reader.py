import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.dataclass import MetaItem
from src.data.pd_module.token_store import StreamingTokenStore
from src.data.pd_module.utils import RerankInputs, build_doc_masks, build_teacher_scores


class PretokenizeRuntimeCacheReader:
    """Handle runtime loading/access for pretokenized streaming caches."""

    def __init__(self, *, owner: Any) -> None:
        self._owner: Any = owner

    def close_token_stores(self) -> None:
        owner: Any = self._owner
        cache_obj: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
        for cache_obj in (owner._query_token_cache, owner._doc_token_cache):
            if isinstance(cache_obj, StreamingTokenStore):
                cache_obj.close()
        owner._query_token_cache = {}
        owner._doc_token_cache = {}
        owner._cache_owner_pid = None

    def ensure_streaming_cache_for_worker(self) -> None:
        owner: Any = self._owner
        current_pid: int = int(os.getpid())
        if (
            owner._cache_owner_pid == current_pid
            and isinstance(owner._query_token_cache, StreamingTokenStore)
            and isinstance(owner._doc_token_cache, StreamingTokenStore)
        ):
            return
        self.close_token_stores()
        use_dataset_row_index: bool = bool(owner._streaming_use_dataset_row_index)
        query_id_to_idx: Mapping[str, int] | None = (
            owner.dataset.query_dataset_id_to_idx if use_dataset_row_index else None
        )
        doc_id_to_idx: Mapping[str, int] | None = (
            owner.dataset.corpus_dataset_id_to_idx if use_dataset_row_index else None
        )
        query_row_index_path: Path | None = (
            owner._query_row_index_path if use_dataset_row_index else None
        )
        doc_row_index_path: Path | None = (
            owner._doc_row_index_path if use_dataset_row_index else None
        )
        owner._query_token_cache = StreamingTokenStore(
            cache_dir=owner._cache_dir,
            prefix="queries",
            max_cached_shards=owner._streaming_max_cached_shards,
            max_cached_rows=owner._streaming_row_cache_size,
            max_cached_index_rows=owner._streaming_index_cache_size,
            sqlite_cache_size_kib=owner._streaming_sqlite_cache_size_kib,
            sqlite_mmap_size=owner._streaming_sqlite_mmap_size,
            id_to_dataset_idx=query_id_to_idx,
            dataset_idx_to_global_row_path=(
                query_row_index_path if query_row_index_path is not None else None
            ),
            shard_size=owner._query_shard_size,
        )
        owner._doc_token_cache = StreamingTokenStore(
            cache_dir=owner._cache_dir,
            prefix="docs",
            max_cached_shards=owner._streaming_max_cached_shards,
            max_cached_rows=owner._streaming_row_cache_size,
            max_cached_index_rows=owner._streaming_index_cache_size,
            sqlite_cache_size_kib=owner._streaming_sqlite_cache_size_kib,
            sqlite_mmap_size=owner._streaming_sqlite_mmap_size,
            id_to_dataset_idx=doc_id_to_idx,
            dataset_idx_to_global_row_path=(
                doc_row_index_path if doc_row_index_path is not None else None
            ),
            shard_size=owner._doc_shard_size,
        )
        owner._load_meta_row_pointer_arrays()
        owner._cache_owner_pid = current_pid

    def build_rerank_inputs_from_meta_row_pointers(
        self,
        *,
        data_idx: int,
        meta_item: MetaItem,
    ) -> RerankInputs | None:
        owner: Any = self._owner
        if (
            not owner._streaming_use_meta_row_pointer
            or not isinstance(owner._query_token_cache, StreamingTokenStore)
            or not isinstance(owner._doc_token_cache, StreamingTokenStore)
        ):
            return None
        query_row_pointers: np.ndarray | None = owner._meta_query_row_pointers
        doc_row_pointers: np.ndarray | None = owner._meta_doc_row_pointers
        doc_counts: np.ndarray | None = owner._meta_doc_counts
        if (
            query_row_pointers is None
            or doc_row_pointers is None
            or doc_counts is None
        ):
            return None
        row_idx: int = int(data_idx)
        if (
            row_idx < 0
            or row_idx >= int(query_row_pointers.shape[0])
            or row_idx >= int(doc_row_pointers.shape[0])
            or row_idx >= int(doc_counts.shape[0])
        ):
            return None
        query_global_row: int = int(query_row_pointers[row_idx])
        if query_global_row < 0:
            return None
        expected_doc_count: int = int(len(meta_item.pos_ids) + len(meta_item.neg_ids))
        mapped_doc_count: int = int(doc_counts[row_idx])
        if expected_doc_count != mapped_doc_count:
            return None
        if expected_doc_count < 0 or expected_doc_count > int(doc_row_pointers.shape[1]):
            return None
        doc_rows_slice: np.ndarray = doc_row_pointers[row_idx][:expected_doc_count]
        if bool((doc_rows_slice < 0).any()):
            return None
        query_tokens: tuple[torch.Tensor, torch.Tensor] | None = (
            owner._query_token_cache.get_by_global_row(query_global_row, default=None)
        )
        if query_tokens is None:
            return None
        query_input_ids: torch.Tensor
        query_attention_mask: torch.Tensor
        query_input_ids, query_attention_mask = query_tokens

        doc_global_rows: list[int] = [int(row_value) for row_value in doc_rows_slice.tolist()]
        doc_tokens_rows: list[tuple[torch.Tensor, torch.Tensor] | None] = (
            owner._doc_token_cache.get_many_by_global_rows(
                doc_global_rows,
                default=None,
            )
        )
        if any(doc_tokens is None for doc_tokens in doc_tokens_rows):
            return None
        doc_rows: list[tuple[torch.Tensor, torch.Tensor]] = [
            doc_tokens
            for doc_tokens in doc_tokens_rows
            if doc_tokens is not None
        ]
        doc_input_rows: list[torch.Tensor] = [doc_row[0] for doc_row in doc_rows]
        doc_mask_rows: list[torch.Tensor] = [doc_row[1] for doc_row in doc_rows]

        if not doc_input_rows:
            doc_input_ids: torch.Tensor = torch.empty(
                (0, owner.max_doc_length), dtype=torch.long
            )
            doc_attention_mask: torch.Tensor = torch.empty(
                (0, owner.max_doc_length), dtype=torch.long
            )
        elif owner.max_padding:
            doc_input_ids = torch.stack(doc_input_rows, dim=0)
            doc_attention_mask = torch.stack(doc_mask_rows, dim=0)
        else:
            doc_input_ids = pad_sequence(
                doc_input_rows,
                batch_first=True,
                padding_value=int(owner.tokenizer.pad_token_id),
            )
            doc_attention_mask = pad_sequence(
                doc_mask_rows,
                batch_first=True,
                padding_value=0,
            )

        num_pos: int = int(len(meta_item.pos_ids))
        doc_mask: torch.Tensor
        pos_mask: torch.Tensor
        doc_mask, pos_mask = build_doc_masks(expected_doc_count, num_pos)
        teacher_scores: torch.Tensor = build_teacher_scores(
            meta_item.pos_scores,
            meta_item.neg_scores,
            num_pos=num_pos,
            num_neg=int(len(meta_item.neg_ids)),
        )
        return RerankInputs(
            qid=meta_item.qid,
            pos_ids=list(meta_item.pos_ids),
            neg_ids=list(meta_item.neg_ids),
            query_text="",
            doc_texts=[""] * expected_doc_count,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            doc_mask=doc_mask,
            pos_mask=pos_mask,
            teacher_scores=teacher_scores,
            num_pos=num_pos,
            num_neg=int(len(meta_item.neg_ids)),
        )
