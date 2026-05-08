from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.utils.logging import get_logger
from src.utils.normalize import normalize_optional_str
from src.utils.sparse_encoder import NativeSparseEncoderAdapter

try:
    from mteb._create_dataloaders import create_dataloader
    from mteb.models import ModelMeta
    from mteb.types import PromptType
except ImportError:
    create_dataloader = None
    ModelMeta = Any  # type: ignore[assignment]
    PromptType = Any  # type: ignore[assignment]

logger = get_logger(__name__, __file__)


def _require_mteb() -> None:
    if create_dataloader is None:
        raise ImportError(
            "mteb is required for true MTEB evaluation. Install it with `pip install mteb`."
        )


def _as_str_list(values: Any) -> list[str]:
    if isinstance(values, str):
        return [values]
    return [str(item) for item in values]


def _extract_batch_texts(batch: dict[str, Any], *, is_query: bool) -> list[str]:
    if is_query and "query" in batch:
        return _as_str_list(batch["query"])
    if "text" not in batch:
        raise KeyError("Expected batched MTEB input to include a `text` field.")
    return _as_str_list(batch["text"])


def _resolve_batch_size(
    default_batch_size: int,
    encode_kwargs: dict[str, Any],
) -> int:
    value: Any = encode_kwargs.get("batch_size", default_batch_size)
    return int(value)


def _resolve_max_active_dims(encode_kwargs: dict[str, Any]) -> int | None:
    value: Any = encode_kwargs.get("max_active_dims")
    if value is None:
        return None
    return int(value)


def _resolve_optional_batch_size(value: int | None, fallback: int) -> int:
    if value is None:
        return int(fallback)
    return int(value)


def _safe_len(values: Any) -> int | None:
    try:
        return int(len(values))
    except (TypeError, ValueError):
        return None


def _build_dataloader_kwargs(encode_kwargs: dict[str, Any]) -> dict[str, Any]:
    excluded_keys: set[str] = {
        "batch_size",
        "max_active_dims",
        "show_progress_bar",
    }
    return {
        key: value for key, value in encode_kwargs.items() if key not in excluded_keys
    }


def _to_sparse_csr(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.layout == torch.sparse_csr:
        return embeddings
    if embeddings.layout == torch.sparse_coo:
        return embeddings.coalesce().to_sparse_csr()
    return embeddings.to_sparse_csr()


def _concat_sparse_csr_rows(chunks: list[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        raise ValueError("Expected at least one sparse chunk to concatenate.")
    if len(chunks) == 1:
        return _to_sparse_csr(chunks[0])

    normalized_chunks: list[torch.Tensor] = [_to_sparse_csr(chunk) for chunk in chunks]
    value_parts: list[torch.Tensor] = []
    column_parts: list[torch.Tensor] = []
    crow_parts: list[torch.Tensor] = []
    total_rows: int = 0
    total_nnz: int = 0
    column_count: int = int(normalized_chunks[0].shape[1])
    dtype: torch.dtype = normalized_chunks[0].dtype
    device: torch.device = normalized_chunks[0].device

    for chunk_index, chunk in enumerate(normalized_chunks):
        if int(chunk.shape[1]) != column_count:
            raise ValueError("All sparse chunks must have the same column count.")
        crow_indices: torch.Tensor = chunk.crow_indices()
        column_indices: torch.Tensor = chunk.col_indices()
        values: torch.Tensor = chunk.values()
        if chunk_index == 0:
            crow_parts.append(crow_indices)
        else:
            crow_parts.append(crow_indices[1:] + int(total_nnz))
        column_parts.append(column_indices)
        value_parts.append(values)
        total_rows += int(chunk.shape[0])
        total_nnz += int(values.numel())

    return torch.sparse_csr_tensor(
        torch.cat(crow_parts, dim=0),
        torch.cat(column_parts, dim=0),
        torch.cat(value_parts, dim=0),
        size=(total_rows, column_count),
        dtype=dtype,
        device=device,
    )


def _merge_topk_state(
    best_scores: torch.Tensor,
    best_positions: torch.Tensor,
    chunk_scores: torch.Tensor,
    chunk_positions: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_scores: torch.Tensor = torch.cat([best_scores, chunk_scores], dim=1)
    combined_positions: torch.Tensor = torch.cat([best_positions, chunk_positions], dim=1)
    merged_scores: torch.Tensor
    merged_order: torch.Tensor
    merged_scores, merged_order = torch.topk(
        combined_scores,
        k=int(top_k),
        dim=1,
        largest=True,
    )
    merged_positions: torch.Tensor = combined_positions.gather(1, merged_order)
    return merged_scores, merged_positions


@dataclass
class _HeartbeatTracker:
    logger: Any
    interval_seconds: float
    _phase_started_at: dict[str, float] = field(default_factory=dict)
    _phase_last_logged_at: dict[str, float] = field(default_factory=dict)

    def _phase_key(self, phase: str, task_name: str) -> str:
        return f"{task_name}:{phase}"

    def _format_progress(
        self,
        *,
        label: str,
        processed: int,
        total: int | None,
    ) -> str:
        if total is None or total <= 0:
            return f"{label}={processed}"
        percent: float = (float(processed) / float(total)) * 100.0
        return f"{label}={processed}/{total} ({percent:.1f}%)"

    def _format_extras(self, extras: dict[str, Any] | None) -> str:
        if not extras:
            return ""
        parts: list[str] = []
        key: str
        value: Any
        for key, value in extras.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        if not parts:
            return ""
        return " " + " ".join(parts)

    def _elapsed_seconds(self, key: str, now: float) -> int:
        started_at: float = self._phase_started_at.get(key, now)
        return max(0, int(now - started_at))

    def start(
        self,
        *,
        phase: str,
        task_name: str,
        label: str,
        total: int | None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        if self.interval_seconds <= 0:
            return
        now: float = time.monotonic()
        key: str = self._phase_key(phase, task_name)
        self._phase_started_at[key] = now
        self._phase_last_logged_at[key] = now
        self.logger.info(
            "Heartbeat[%s][%s] start %s%s",
            phase,
            task_name,
            self._format_progress(label=label, processed=0, total=total),
            self._format_extras(extras),
        )

    def progress(
        self,
        *,
        phase: str,
        task_name: str,
        label: str,
        processed: int,
        total: int | None,
        extras: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        if self.interval_seconds <= 0:
            return
        now: float = time.monotonic()
        key: str = self._phase_key(phase, task_name)
        last_logged_at: float | None = self._phase_last_logged_at.get(key)
        if not force and last_logged_at is not None:
            if (now - last_logged_at) < float(self.interval_seconds):
                return
        self._phase_last_logged_at[key] = now
        elapsed_seconds: int = self._elapsed_seconds(key, now)
        self.logger.info(
            "Heartbeat[%s][%s] %s elapsed=%ss%s",
            phase,
            task_name,
            self._format_progress(label=label, processed=processed, total=total),
            elapsed_seconds,
            self._format_extras(extras),
        )
        if force:
            self._phase_started_at.pop(key, None)
            self._phase_last_logged_at.pop(key, None)


def _resolve_task_name(task_metadata: Any) -> str:
    name: Any = getattr(task_metadata, "name", None)
    if name is None:
        name = getattr(task_metadata, "dataset_name", None)
    resolved: str = str(name).strip() if name is not None else ""
    if resolved:
        return resolved
    return "task"


class MTEBSparseRetrievalAdapter(NativeSparseEncoderAdapter):
    """Retrieval-only MTEB adapter backed by the in-memory sparse encoder."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        *,
        model_cfg: DictConfig,
        device: torch.device,
        batch_size: int,
        query_batch_size: int | None,
        corpus_batch_size: int | None,
        max_query_length: int,
        max_doc_length: int,
        corpus_chunk_size: int,
        heartbeat_interval_seconds: float,
        model_name: str,
        model_revision: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            model_cfg=model_cfg,
            device=device,
            batch_size=batch_size,
            max_query_length=max_query_length,
            max_doc_length=max_doc_length,
        )
        _require_mteb()
        self.query_batch_size: int | None = (
            None if query_batch_size is None else int(query_batch_size)
        )
        self.corpus_batch_size: int | None = (
            None if corpus_batch_size is None else int(corpus_batch_size)
        )
        self.corpus_chunk_size: int = int(corpus_chunk_size)
        self._corpus_embeddings: list[torch.Tensor] = []
        self._corpus_chunk_sizes: list[int] = []
        self._flat_corpus_ids: list[str] = []
        self.score_device: torch.device = (
            device if device.type == "cuda" else torch.device("cpu")
        )
        self.heartbeat: _HeartbeatTracker = _HeartbeatTracker(
            logger=logger,
            interval_seconds=float(heartbeat_interval_seconds),
        )
        self.mteb_model_meta: ModelMeta = ModelMeta.create_empty(
            {
                "name": model_name,
                "revision": model_revision,
                "framework": ["PyTorch", "Transformers"],
                "model_type": ["sparse"],
                "similarity_fn_name": "dot",
                "use_instructions": bool(
                    normalize_optional_str(model_cfg.get("instruction_text"))
                ),
                "embed_dim": int(self.model.encoder.vocab_size),
            }
        )

    def _heartbeat_tracker(self) -> _HeartbeatTracker:
        tracker: Any = getattr(self, "heartbeat", None)
        if isinstance(tracker, _HeartbeatTracker):
            return tracker
        tracker = _HeartbeatTracker(logger=logger, interval_seconds=0.0)
        self.heartbeat = tracker
        return tracker

    def clear_index(self) -> None:
        self._corpus_embeddings.clear()
        self._corpus_chunk_sizes.clear()
        self._flat_corpus_ids.clear()

    def _append_corpus_chunk(
        self,
        *,
        chunk_embeddings: list[torch.Tensor],
        chunk_doc_ids: list[str],
    ) -> None:
        if not chunk_embeddings:
            return
        merged_embeddings: torch.Tensor = _concat_sparse_csr_rows(chunk_embeddings)
        self._corpus_embeddings.append(merged_embeddings)
        self._corpus_chunk_sizes.append(len(chunk_doc_ids))
        self._flat_corpus_ids.extend(chunk_doc_ids)

    def _encode_query_blocks(
        self,
        queries: Any,
        *,
        task_metadata: Any,
        encode_kwargs: dict[str, Any],
        num_proc: int | None,
    ) -> list[tuple[list[str], torch.Tensor]]:
        task_name: str = _resolve_task_name(task_metadata)
        total_queries: int | None = _safe_len(queries)
        query_batch_size: int = _resolve_optional_batch_size(
            self.query_batch_size,
            _resolve_batch_size(self.batch_size, encode_kwargs),
        )
        max_active_dims: int | None = _resolve_max_active_dims(encode_kwargs)
        dataloader_kwargs: dict[str, Any] = _build_dataloader_kwargs(encode_kwargs)
        dataloader: Any = create_dataloader(
            queries,
            task_metadata,
            prompt_type=PromptType.query,
            batch_size=query_batch_size,
            num_proc=num_proc,
            **dataloader_kwargs,
        )
        heartbeat: _HeartbeatTracker = self._heartbeat_tracker()
        heartbeat.start(
            phase="query_encode",
            task_name=task_name,
            label="queries",
            total=total_queries,
            extras={"batch_size": query_batch_size},
        )
        query_blocks: list[tuple[list[str], torch.Tensor]] = []
        processed_queries: int = 0
        block_count: int = 0
        batch: dict[str, Any]
        for batch in dataloader:
            query_ids: list[str] = _as_str_list(batch["id"])
            query_texts: list[str] = _extract_batch_texts(batch, is_query=True)
            query_embeddings: torch.Tensor = self.encode_query(
                query_texts,
                batch_size=len(query_texts),
                show_progress_bar=False,
                convert_to_sparse_tensor=False,
                save_to_cpu=False,
                max_active_dims=max_active_dims,
            )
            query_dense: torch.Tensor = query_embeddings.to(device=self.score_device)
            query_blocks.append((query_ids, query_dense.transpose(0, 1).contiguous()))
            processed_queries += len(query_texts)
            block_count += 1
            heartbeat.progress(
                phase="query_encode",
                task_name=task_name,
                label="queries",
                processed=processed_queries,
                total=total_queries,
                extras={"blocks": block_count},
            )
        heartbeat.progress(
            phase="query_encode",
            task_name=task_name,
            label="queries",
            processed=processed_queries,
            total=total_queries,
            extras={"blocks": block_count, "status": "complete"},
            force=True,
        )
        return query_blocks

    def index(
        self,
        corpus: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        num_proc: int | None,
    ) -> None:
        _ = hf_split, hf_subset
        self.clear_index()
        task_name: str = _resolve_task_name(task_metadata)
        total_docs: int | None = _safe_len(corpus)
        batch_size: int = _resolve_optional_batch_size(
            self.corpus_batch_size,
            _resolve_batch_size(self.batch_size, encode_kwargs),
        )
        max_active_dims: int | None = _resolve_max_active_dims(encode_kwargs)
        dataloader_kwargs: dict[str, Any] = _build_dataloader_kwargs(encode_kwargs)
        dataloader: Any = create_dataloader(
            corpus,
            task_metadata,
            prompt_type=PromptType.document,
            batch_size=batch_size,
            num_proc=num_proc,
            **dataloader_kwargs,
        )
        heartbeat: _HeartbeatTracker = self._heartbeat_tracker()
        heartbeat.start(
            phase="index",
            task_name=task_name,
            label="docs",
            total=total_docs,
            extras={
                "batch_size": batch_size,
                "chunk_size": self.corpus_chunk_size,
            },
        )
        pending_embeddings: list[torch.Tensor] = []
        pending_doc_ids: list[str] = []
        pending_doc_count: int = 0
        processed_docs: int = 0
        chunk_count: int = 0
        flush_threshold: int = max(1, int(self.corpus_chunk_size))
        batch: dict[str, Any]
        for batch in dataloader:
            doc_ids: list[str] = _as_str_list(batch["id"])
            doc_texts: list[str] = _extract_batch_texts(batch, is_query=False)
            doc_embeddings: torch.Tensor = self.encode_document(
                doc_texts,
                batch_size=len(doc_texts),
                show_progress_bar=False,
                convert_to_sparse_tensor=True,
                save_to_cpu=True,
                max_active_dims=max_active_dims,
            )
            pending_embeddings.append(_to_sparse_csr(doc_embeddings))
            pending_doc_ids.extend(doc_ids)
            pending_doc_count += len(doc_ids)
            processed_docs += len(doc_ids)
            if pending_doc_count >= flush_threshold:
                self._append_corpus_chunk(
                    chunk_embeddings=pending_embeddings,
                    chunk_doc_ids=pending_doc_ids,
                )
                chunk_count += 1
                pending_embeddings = []
                pending_doc_ids = []
                pending_doc_count = 0
            heartbeat.progress(
                phase="index",
                task_name=task_name,
                label="docs",
                processed=processed_docs,
                total=total_docs,
                extras={"chunks": chunk_count},
            )
        if pending_embeddings:
            self._append_corpus_chunk(
                chunk_embeddings=pending_embeddings,
                chunk_doc_ids=pending_doc_ids,
            )
            chunk_count += 1
        heartbeat.progress(
            phase="index",
            task_name=task_name,
            label="docs",
            processed=processed_docs,
            total=total_docs,
            extras={"chunks": chunk_count, "status": "complete"},
            force=True,
        )

    def search(
        self,
        queries: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
        top_ranked: dict[str, list[str]] | None = None,
        num_proc: int | None,
    ) -> dict[str, dict[str, float]]:
        _ = hf_split, hf_subset
        if top_ranked is not None:
            raise NotImplementedError(
                "The true MTEB sparse adapter currently supports retrieval only, not reranking."
            )
        if not self._corpus_embeddings:
            raise ValueError("Corpus must be indexed before searching.")
        task_name: str = _resolve_task_name(task_metadata)

        query_blocks: list[tuple[list[str], torch.Tensor]] = self._encode_query_blocks(
            queries,
            task_metadata=task_metadata,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )
        if not query_blocks:
            self.clear_index()
            return {}

        results: dict[str, dict[str, float]] = {}
        top_k_value: int = int(top_k)
        total_queries: int = sum(len(query_ids) for query_ids, _ in query_blocks)
        total_corpus_docs: int = len(self._flat_corpus_ids)
        total_chunks: int = len(self._corpus_embeddings)
        heartbeat: _HeartbeatTracker = self._heartbeat_tracker()
        heartbeat.start(
            phase="search",
            task_name=task_name,
            label="corpus_docs_scored",
            total=total_corpus_docs,
            extras={"queries": total_queries, "chunks": f"0/{total_chunks}"},
        )
        block_states: list[tuple[list[str], torch.Tensor, torch.Tensor]] = []
        query_ids: list[str]
        query_block_t: torch.Tensor
        for query_ids, query_block_t in query_blocks:
            block_size: int = int(query_block_t.shape[1])
            block_states.append(
                (
                    query_ids,
                    torch.full(
                        (block_size, top_k_value),
                        float("-inf"),
                        dtype=query_block_t.dtype,
                        device=self.score_device,
                    ),
                    torch.full(
                        (block_size, top_k_value),
                        -1,
                        dtype=torch.long,
                        device=self.score_device,
                    ),
                )
            )

        corpus_offset: int = 0
        corpus_embeddings: torch.Tensor
        corpus_chunk_size: int
        chunk_index: int
        for chunk_index, (corpus_embeddings, corpus_chunk_size) in enumerate(
            zip(self._corpus_embeddings, self._corpus_chunk_sizes),
            start=1,
        ):
            score_corpus_embeddings: torch.Tensor = corpus_embeddings.to(
                device=self.score_device
            )
            chunk_top_k: int = min(top_k_value, int(score_corpus_embeddings.shape[0]))
            if chunk_top_k <= 0:
                corpus_offset += int(corpus_chunk_size)
                heartbeat.progress(
                    phase="search",
                    task_name=task_name,
                    label="corpus_docs_scored",
                    processed=corpus_offset,
                    total=total_corpus_docs,
                    extras={"queries": total_queries, "chunks": f"{chunk_index}/{total_chunks}"},
                )
                continue
            block_index: int
            for block_index, (query_ids, best_scores, best_positions) in enumerate(
                block_states
            ):
                query_block_t = query_blocks[block_index][1]
                score_matrix: torch.Tensor = torch.sparse.mm(
                    score_corpus_embeddings,
                    query_block_t,
                ).transpose(0, 1)
                chunk_scores: torch.Tensor
                chunk_local_indices: torch.Tensor
                chunk_scores, chunk_local_indices = torch.topk(
                    score_matrix,
                    k=chunk_top_k,
                    dim=1,
                    largest=True,
                )
                chunk_positions: torch.Tensor = chunk_local_indices.to(
                    dtype=torch.long
                ) + int(corpus_offset)
                merged_scores: torch.Tensor
                merged_positions: torch.Tensor
                merged_scores, merged_positions = _merge_topk_state(
                    best_scores,
                    best_positions,
                    chunk_scores,
                    chunk_positions,
                    top_k=top_k_value,
                )
                block_states[block_index] = (
                    query_ids,
                    merged_scores,
                    merged_positions,
                )
            corpus_offset += int(corpus_chunk_size)
            heartbeat.progress(
                phase="search",
                task_name=task_name,
                label="corpus_docs_scored",
                processed=corpus_offset,
                total=total_corpus_docs,
                extras={"queries": total_queries, "chunks": f"{chunk_index}/{total_chunks}"},
            )

        for query_ids, best_scores, best_positions in block_states:
            best_scores_cpu: torch.Tensor = best_scores.to(device=torch.device("cpu"))
            best_positions_cpu: torch.Tensor = best_positions.to(device=torch.device("cpu"))
            row_index: int
            query_id: str
            for row_index, query_id in enumerate(query_ids):
                ranked_results: dict[str, float] = {}
                doc_position: int
                score: float
                for score, doc_position in zip(
                    best_scores_cpu[row_index].tolist(),
                    best_positions_cpu[row_index].tolist(),
                ):
                    if doc_position < 0:
                        continue
                    ranked_results[self._flat_corpus_ids[int(doc_position)]] = float(score)
                results[query_id] = ranked_results

        heartbeat.progress(
            phase="search",
            task_name=task_name,
            label="corpus_docs_scored",
            processed=corpus_offset,
            total=total_corpus_docs,
            extras={
                "queries": total_queries,
                "chunks": f"{total_chunks}/{total_chunks}",
                "status": "complete",
            },
            force=True,
        )
        self.clear_index()
        return results
