from bisect import bisect_right
from typing import Callable, Sequence

import torch
import torch.nn.functional as F


def resolve_indptr_values(indptr: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(indptr, torch.Tensor):
        return [int(value) for value in indptr.tolist()]
    return [int(value) for value in indptr]


def resolve_chunk_segments(
    start_idx: int,
    end_idx: int,
    indptr_values: Sequence[int],
    *,
    entity_name: str = "entity",
) -> tuple[list[int], list[int]]:
    entity_indices: list[int] = []
    entity_lengths: list[int] = []
    num_entities: int = max(len(indptr_values) - 1, 0)
    entity_idx: int = max(bisect_right(indptr_values, start_idx) - 1, 0)
    cursor: int = start_idx
    while cursor < end_idx and entity_idx < num_entities:
        entity_end: int = int(indptr_values[entity_idx + 1])
        if entity_end <= cursor:
            entity_idx += 1
            continue
        take: int = min(entity_end, end_idx) - cursor
        if take > 0:
            entity_indices.append(entity_idx)
            entity_lengths.append(take)
            cursor += take
        entity_idx += 1
    if cursor != end_idx:
        raise RuntimeError(
            f"Failed to align window chunk with {entity_name} boundaries."
        )
    return entity_indices, entity_lengths


def encode_in_chunks(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_mask: torch.Tensor | None,
    *,
    encode_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
    chunk_size: int | None,
    mark_step: Callable[[], None] | None = None,
) -> torch.Tensor:
    total_items: int = int(input_ids.shape[0])
    effective_chunk_size: int
    if chunk_size is None or int(chunk_size) <= 0:
        effective_chunk_size = max(total_items, 1)
    else:
        effective_chunk_size = max(int(chunk_size), 1)
    if total_items <= effective_chunk_size:
        if mark_step is not None:
            mark_step()
        return encode_fn(input_ids, attention_mask, pooling_mask)

    rep_chunks: list[torch.Tensor] = []
    start: int
    for start in range(0, total_items, effective_chunk_size):
        end: int = min(start + effective_chunk_size, total_items)
        if mark_step is not None:
            mark_step()
        rep_chunks.append(
            encode_fn(
                input_ids[start:end],
                attention_mask[start:end],
                None if pooling_mask is None else pooling_mask[start:end],
            )
        )
    return torch.cat(rep_chunks, dim=0)


def encode_and_aggregate_windows(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_mask: torch.Tensor | None,
    *,
    indptr: Sequence[int] | torch.Tensor,
    encode_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
    pooling_mode: str,
    output_dim: int,
    output_dtype: torch.dtype,
    pad_token_id: int,
    chunk_size: int | None,
    use_fixed_size_chunks: bool,
    mark_step: Callable[[], None] | None = None,
    entity_name: str = "entity",
) -> torch.Tensor:
    total_windows: int = int(input_ids.shape[0])
    indptr_values: list[int] = resolve_indptr_values(indptr)
    num_entities: int = max(len(indptr_values) - 1, 0)
    if num_entities == 0:
        return torch.empty(
            (0, int(output_dim)), dtype=output_dtype, device=input_ids.device
        )
    if total_windows == 0:
        return torch.zeros(
            (num_entities, int(output_dim)),
            dtype=output_dtype,
            device=input_ids.device,
        )

    normalized_pooling_mode: str = str(pooling_mode).lower()
    if normalized_pooling_mode not in {"sum", "max"}:
        raise ValueError(
            "Unsupported pooling for window aggregation: "
            f"{normalized_pooling_mode}"
        )
    effective_chunk_size: int = (
        total_windows
        if chunk_size is None or int(chunk_size) <= 0
        else int(chunk_size)
    )
    aggregated: torch.Tensor | None = None
    entity_lengths: list[int] = [
        int(indptr_values[idx + 1]) - int(indptr_values[idx])
        for idx in range(num_entities)
    ]

    start_idx: int
    for start_idx in range(0, total_windows, effective_chunk_size):
        end_idx: int = min(start_idx + effective_chunk_size, total_windows)
        real_count: int = end_idx - start_idx
        chunk_input_ids: torch.Tensor = input_ids[start_idx:end_idx]
        chunk_attention_mask: torch.Tensor = attention_mask[start_idx:end_idx]
        chunk_pooling_mask: torch.Tensor | None = (
            None if pooling_mask is None else pooling_mask[start_idx:end_idx]
        )
        if use_fixed_size_chunks and real_count < effective_chunk_size:
            pad_rows: int = effective_chunk_size - real_count
            chunk_input_ids = F.pad(
                chunk_input_ids,
                (0, 0, 0, pad_rows),
                value=int(pad_token_id),
            )
            chunk_attention_mask = F.pad(
                chunk_attention_mask,
                (0, 0, 0, pad_rows),
                value=0,
            )
            if chunk_pooling_mask is not None:
                chunk_pooling_mask = F.pad(
                    chunk_pooling_mask,
                    (0, 0, 0, pad_rows),
                    value=0,
                )
        if mark_step is not None:
            mark_step()
        chunk_representations: torch.Tensor = encode_fn(
            chunk_input_ids,
            chunk_attention_mask,
            chunk_pooling_mask,
        )[:real_count]
        if aggregated is None:
            if normalized_pooling_mode == "sum":
                aggregated = chunk_representations.new_zeros((num_entities, output_dim))
            else:
                aggregated = chunk_representations.new_full(
                    (num_entities, output_dim), float("-inf")
                )
        chunk_entity_indices, chunk_entity_lengths = resolve_chunk_segments(
            start_idx,
            end_idx,
            indptr_values,
            entity_name=entity_name,
        )
        lengths_tensor = torch.tensor(
            chunk_entity_lengths,
            device=chunk_representations.device,
            dtype=torch.long,
        )
        partial_representations = torch.segment_reduce(
            chunk_representations,
            reduce=normalized_pooling_mode,
            lengths=lengths_tensor,
        )
        entity_indices_tensor = torch.tensor(
            chunk_entity_indices,
            device=chunk_representations.device,
            dtype=torch.long,
        )
        if normalized_pooling_mode == "sum":
            aggregated.index_add_(0, entity_indices_tensor, partial_representations)
        else:
            current = aggregated.index_select(0, entity_indices_tensor)
            aggregated.index_copy_(
                0,
                entity_indices_tensor,
                torch.maximum(current, partial_representations),
            )

    if aggregated is None:
        return torch.zeros(
            (num_entities, int(output_dim)),
            dtype=output_dtype,
            device=input_ids.device,
        )
    if normalized_pooling_mode == "max":
        empty_entity_indices: list[int] = [
            entity_idx for entity_idx, count in enumerate(entity_lengths) if count <= 0
        ]
        if empty_entity_indices:
            aggregated[empty_entity_indices] = 0
    return aggregated


__all__ = [
    "encode_and_aggregate_windows",
    "encode_in_chunks",
    "resolve_chunk_segments",
    "resolve_indptr_values",
]
