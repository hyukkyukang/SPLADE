import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import Dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from src.data.collator import UniversalCollator
from src.data.dataclass import MetaItem, TrainingDataItem
from src.data.pd_module import PDModule
from src.data.pd_module.pretokenize_lifecycle import PretokenizeCacheLifecycleManager
from src.data.pd_module.pretokenize_row_materializer import TrainingRowMaterializer
from src.data.pd_module.pretokenize_runtime_reader import (
    PretokenizeRuntimeCacheReader,
)
from src.data.pd_module.pretokenize_writer import PretokenizeCacheStorageWriter
from src.data.pd_module.pretokenize import (
    normalize_storage_format,
    make_id_key,
    make_text_key,
    resolve_row_index_path,
    STORAGE_FORMAT_HYBRID,
    STORAGE_FORMAT_SIDECAR_ONLY,
)
from src.data.pd_module.utils import (
    RerankInputs,
)
from src.data.utils import resolve_dataset_column
from src.utils import get_rank, maybe_barrier
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.script_setup import normalize_optional_str

logger = get_logger("TrainingPDModule")

TOKENIZE_BATCH_SIZE: int = 2048
ARROW_UNIQUE_SLICE_ROWS: int = 5000000
META_QUERY_GLOBAL_ROW_FILENAME: str = "meta.query_global_rows.npy"
META_DOC_GLOBAL_ROWS_FILENAME: str = "meta.doc_global_rows.npy"
META_DOC_COUNT_FILENAME: str = "meta.doc_counts.npy"


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
        cache_namespace: str | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            tokenizer=tokenizer,
            seed=seed,
            load_teacher_scores=load_teacher_scores,
            require_teacher_scores=require_teacher_scores,
        )
        self._collator: UniversalCollator | None = None
        pretokenize_cfg: DictConfig | None = (
            self.cfg.get("pretokenize") if "pretokenize" in self.cfg else None
        )
        self._pretokenize_enabled: bool = bool(
            pretokenize_cfg.enabled if pretokenize_cfg is not None else False
        )
        self._pretokenize_overwrite: bool = bool(
            pretokenize_cfg.overwrite if pretokenize_cfg is not None else False
        )
        self._allow_runtime_tokenize_fallback: bool = bool(
            pretokenize_cfg.allow_runtime_tokenize_fallback
            if pretokenize_cfg is not None
            else False
        )
        self._require_cache_complete: bool = bool(
            pretokenize_cfg.require_cache_complete
            if pretokenize_cfg is not None
            else True
        )
        self._query_shard_size: int = int(
            pretokenize_cfg.query_shard_size if pretokenize_cfg is not None else 200000
        )
        self._doc_shard_size: int = int(
            pretokenize_cfg.doc_shard_size if pretokenize_cfg is not None else 200000
        )
        self._write_dtype: str = str(
            pretokenize_cfg.write_dtype if pretokenize_cfg is not None else "int32"
        )
        self._pretokenize_storage_format: str = normalize_storage_format(
            pretokenize_cfg.get("storage_format", STORAGE_FORMAT_HYBRID)
            if pretokenize_cfg is not None
            else STORAGE_FORMAT_HYBRID
        )
        loading_mode_value: str = str(
            pretokenize_cfg.get("loading_mode", "streaming")
            if pretokenize_cfg is not None
            else "streaming"
        )
        loading_mode_normalized: str = loading_mode_value.strip().lower()
        if loading_mode_normalized not in {"eager", "streaming"}:
            raise ValueError(
                "pretokenize.loading_mode must be one of: eager, streaming. "
                f"Got: {loading_mode_value!r}"
            )
        self._pretokenize_loading_mode: str = loading_mode_normalized
        self._streaming_index_backend: str = str(
            pretokenize_cfg.get("streaming_index_backend", "sqlite")
            if pretokenize_cfg is not None
            else "sqlite"
        )
        self._streaming_max_cached_shards: int = int(
            pretokenize_cfg.get("streaming_max_cached_shards", 2)
            if pretokenize_cfg is not None
            else 2
        )
        self._streaming_row_cache_size: int = int(
            pretokenize_cfg.get("streaming_row_cache_size", 200000)
            if pretokenize_cfg is not None
            else 200000
        )
        streaming_index_cache_size_value: Any | None = (
            pretokenize_cfg.get("streaming_index_cache_size")
            if pretokenize_cfg is not None
            else None
        )
        if streaming_index_cache_size_value is None:
            self._streaming_index_cache_size: int = int(self._streaming_row_cache_size)
        else:
            self._streaming_index_cache_size = int(streaming_index_cache_size_value)
        self._streaming_sqlite_cache_size_kib: int = int(
            pretokenize_cfg.get("streaming_sqlite_cache_size_kib", 131072)
            if pretokenize_cfg is not None
            else 131072
        )
        self._streaming_sqlite_mmap_size: int = int(
            pretokenize_cfg.get("streaming_sqlite_mmap_size", 1073741824)
            if pretokenize_cfg is not None
            else 1073741824
        )
        self._streaming_use_dataset_row_index: bool = bool(
            pretokenize_cfg.get("streaming_use_dataset_row_index", False)
            if pretokenize_cfg is not None
            else False
        )
        self._streaming_use_meta_row_pointer: bool = bool(
            pretokenize_cfg.get("streaming_use_meta_row_pointer", True)
            if pretokenize_cfg is not None
            else True
        )
        self._streaming_numpy_sidecar: bool = bool(
            pretokenize_cfg.get("streaming_numpy_sidecar", False)
            if pretokenize_cfg is not None
            else False
        )
        if self._pretokenize_storage_format == STORAGE_FORMAT_SIDECAR_ONLY:
            if self._pretokenize_loading_mode != "streaming":
                raise ValueError(
                    "pretokenize.storage_format='sidecar_only' requires "
                    "pretokenize.loading_mode='streaming'."
                )
            self._streaming_numpy_sidecar = True
        if self._streaming_use_meta_row_pointer:
            self._streaming_use_dataset_row_index = True
        parquet_row_group_size_value: Any | None = (
            pretokenize_cfg.get("parquet_row_group_size")
            if pretokenize_cfg is not None
            else None
        )
        self._parquet_row_group_size: int | None = (
            None
            if parquet_row_group_size_value is None
            else int(parquet_row_group_size_value)
        )
        if (
            self._parquet_row_group_size is not None
            and self._parquet_row_group_size <= 0
        ):
            raise ValueError(
                "pretokenize.parquet_row_group_size must be a positive integer "
                f"when set. Got: {parquet_row_group_size_value!r}"
            )
        self._use_streaming_cache: bool = bool(
            self._pretokenize_enabled
            and self._pretokenize_loading_mode == "streaming"
        )
        self._enable_pretokenize_tokenizers_parallelism: bool = bool(
            pretokenize_cfg.get("tokenizers_parallelism", True)
            if pretokenize_cfg is not None
            else True
        )
        configured_output_dir: str | None = (
            normalize_optional_str(pretokenize_cfg.output_dir)
            if pretokenize_cfg is not None
            else None
        )
        if configured_output_dir is None:
            configured_output_dir = f"data/cache/pretokenized/{self.name}"
        self._pretokenize_output_dir: Path = Path(configured_output_dir).expanduser()
        split_name: str = normalize_optional_str(self.cfg.get("split")) or "default"
        self._cache_namespace: str = cache_namespace or split_name
        self._cache_dir: Path = self._pretokenize_output_dir / self._cache_namespace
        self._query_row_index_path: Path = resolve_row_index_path(
            self._cache_dir, "queries"
        )
        self._doc_row_index_path: Path = resolve_row_index_path(self._cache_dir, "docs")
        self._meta_query_row_pointer_path: Path = (
            self._cache_dir / META_QUERY_GLOBAL_ROW_FILENAME
        )
        self._meta_doc_row_pointers_path: Path = (
            self._cache_dir / META_DOC_GLOBAL_ROWS_FILENAME
        )
        self._meta_doc_count_path: Path = self._cache_dir / META_DOC_COUNT_FILENAME
        self._query_token_cache: Mapping[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._doc_token_cache: Mapping[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._meta_query_row_pointers: np.ndarray | None = None
        self._meta_doc_row_pointers: np.ndarray | None = None
        self._meta_doc_counts: np.ndarray | None = None
        self._cache_owner_pid: int | None = None
        self._cache_ready: bool = not self._pretokenize_enabled
        self._pretokenize_runtime_reader = PretokenizeRuntimeCacheReader(owner=self)
        self._pretokenize_writer = PretokenizeCacheStorageWriter(
            owner=self, logger=logger
        )
        self._row_materializer = TrainingRowMaterializer(owner=self)
        self._pretokenize_lifecycle = PretokenizeCacheLifecycleManager(
            owner=self, logger=logger
        )

    def __getitem__(self, idx: int) -> TrainingDataItem:
        return self._row_materializer.materialize(int(idx))

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

    def _requires_query_text_dataset(self) -> bool:
        return not self._uses_strict_pretokenized_cache()

    def _requires_corpus_text_dataset(self) -> bool:
        return not self._uses_strict_pretokenized_cache()

    def _requires_query_id_to_idx(self) -> bool:
        return not self._uses_strict_pretokenized_cache()

    def _requires_corpus_id_to_idx(self) -> bool:
        return not self._uses_strict_pretokenized_cache()

    def _uses_strict_pretokenized_cache(self) -> bool:
        return bool(
            self._pretokenize_enabled and not self._allow_runtime_tokenize_fallback
        )

    @staticmethod
    def _to_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            raw_values: list[Any] = list(value)
        else:
            raw_values = [value]
        output: list[str] = []
        raw_value: Any
        for raw_value in raw_values:
            if raw_value is None:
                continue
            text: str = str(raw_value).strip()
            if text:
                output.append(text)
        return output

    @staticmethod
    def _extract_row_qid(row: dict[str, Any]) -> str | None:
        for key in ("query_id", "qid", "_id", "id"):
            value: Any | None = row.get(key)
            if value is None:
                continue
            text: str = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _extract_row_query_text(row: dict[str, Any]) -> str | None:
        for key in ("query_text", "query", "anchor"):
            value: Any | None = row.get(key)
            if value is None:
                continue
            text: str = str(value).strip()
            if text:
                return text
        return None

    def _extract_row_doc_ids(self, row: dict[str, Any]) -> set[str]:
        doc_ids: set[str] = set()
        for key in (
            "positive_id",
            "negative_id",
            "pos_doc_ids",
            "neg_doc_ids",
            "doc_ids",
        ):
            for doc_id in self._to_str_list(row.get(key)):
                doc_ids.add(doc_id)
        pos_values: Any | None = row.get("pos")
        neg_values: Any | None = row.get("neg")
        for value in (pos_values, neg_values):
            if value is None:
                continue
            if isinstance(value, str):
                text: str = str(value).strip()
                if text and " " not in text:
                    doc_ids.add(text)
                continue
            for doc_id in self._to_str_list(value):
                doc_ids.add(doc_id)
        return doc_ids

    def _extract_row_inline_doc_texts(self, row: dict[str, Any]) -> set[str]:
        texts: set[str] = set()
        for key in ("positive", "negative", "pos_texts", "neg_texts"):
            value: Any | None = row.get(key)
            if value is None:
                continue
            for text in self._to_str_list(value):
                texts.add(text)
        return texts

    @staticmethod
    def _extract_fixed_triplet_doc_ids(row: dict[str, Any]) -> list[str] | None:
        if "positive_id" not in row or "negative_id" not in row:
            return None
        positive_id: str = str(row.get("positive_id") or "").strip()
        negative_id: str = str(row.get("negative_id") or "").strip()
        doc_ids: list[str] = []
        if positive_id:
            doc_ids.append(positive_id)
        if negative_id:
            doc_ids.append(negative_id)
        return doc_ids

    def _iter_meta_rows(self) -> Iterable[tuple[int, dict[str, Any]]]:
        idx: int
        row: Any
        for idx, row in enumerate(self.meta_dataset):
            yield idx, dict(row)

    @contextmanager
    def _tokenizers_parallelism_context(self) -> Iterator[None]:
        if not self._enable_pretokenize_tokenizers_parallelism:
            yield
            return
        previous_value: str | None = os.environ.get("TOKENIZERS_PARALLELISM")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        try:
            yield
        finally:
            if previous_value is None:
                os.environ.pop("TOKENIZERS_PARALLELISM", None)
            else:
                os.environ["TOKENIZERS_PARALLELISM"] = previous_value

    def _close_token_stores(self) -> None:
        self._pretokenize_runtime_reader.close_token_stores()

    def _ensure_streaming_cache_for_worker(self) -> None:
        self._pretokenize_runtime_reader.ensure_streaming_cache_for_worker()

    @staticmethod
    def _load_numpy_array(path: Path) -> np.ndarray | None:
        return PretokenizeCacheLifecycleManager.load_numpy_array(path)

    @staticmethod
    def _write_numpy_array(path: Path, array: np.ndarray) -> None:
        PretokenizeCacheLifecycleManager.write_numpy_array(path, array)

    def _load_meta_row_pointer_arrays(self) -> None:
        self._pretokenize_lifecycle.load_meta_row_pointer_arrays()

    def _build_rerank_inputs_from_meta_row_pointers(
        self,
        *,
        data_idx: int,
        meta_item: MetaItem,
    ) -> RerankInputs | None:
        return self._pretokenize_runtime_reader.build_rerank_inputs_from_meta_row_pointers(
            data_idx=data_idx,
            meta_item=meta_item,
        )

    def _collect_unique_ids_from_scalar_column(
        self,
        *,
        column_name: str,
        progress_desc: str,
    ) -> list[str]:
        column: pa.Array | pa.ChunkedArray = resolve_dataset_column(
            self.meta_dataset, column_name
        )
        total_rows: int = int(len(column))
        if total_rows == 0:
            return []
        progress_bar: tqdm | None = (
            tqdm(
                total=total_rows,
                desc=progress_desc,
                unit="row",
                dynamic_ncols=True,
                mininterval=1.0,
            )
            if int(get_rank()) == 0
            else None
        )
        values: list[str] = []
        seen_values: set[str] = set()
        start: int
        try:
            for start in range(0, total_rows, ARROW_UNIQUE_SLICE_ROWS):
                batch_size: int = min(ARROW_UNIQUE_SLICE_ROWS, total_rows - start)
                unique_values_array: pa.Array = pc.unique(column.slice(start, batch_size))
                unique_values: list[Any] = unique_values_array.to_pylist()
                raw_value: Any
                for raw_value in unique_values:
                    if raw_value is None:
                        continue
                    text: str = str(raw_value).strip()
                    if not text or text in seen_values:
                        continue
                    seen_values.add(text)
                    values.append(text)
                if progress_bar is not None:
                    progress_bar.update(batch_size)
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return values

    def _build_cache_items_from_ids(
        self,
        *,
        query_ids: list[str],
        doc_ids: list[str],
        inline_query_text_by_qid: dict[str, str],
        inline_query_texts: list[str],
        inline_doc_texts: list[str],
    ) -> tuple[dict[str, str], dict[str, str], int, int]:
        query_items: dict[str, str] = {}
        query_id_to_idx: dict[str, int] = self.dataset.query_dataset_id_to_idx
        missing_query_ids: list[str] = []
        query_progress: tqdm | None = (
            tqdm(
                total=len(query_ids),
                desc="Resolve query texts",
                unit="qid",
                dynamic_ncols=True,
                mininterval=1.0,
            )
            if int(get_rank()) == 0
            else None
        )
        try:
            qid: str
            for qid in query_ids:
                query_idx: int = int(query_id_to_idx.get(qid, -1))
                if query_idx < 0:
                    inline_query_text: str | None = inline_query_text_by_qid.get(qid)
                    if inline_query_text is None:
                        missing_query_ids.append(qid)
                    else:
                        query_items[make_id_key(qid)] = inline_query_text
                else:
                    query_items[make_id_key(qid)] = self.dataset.query_text(query_idx)
                if query_progress is not None:
                    query_progress.update(1)
        finally:
            if query_progress is not None:
                query_progress.close()

        if missing_query_ids:
            fallback_query_texts: dict[str, str] = self.dataset.lookup_query_texts(
                missing_query_ids
            )
            fallback_qid: str
            for fallback_qid, query_text in fallback_query_texts.items():
                query_items[make_id_key(fallback_qid)] = query_text
            if fallback_query_texts:
                log_if_rank_zero(
                    logger,
                    "Pretokenize query lookup fallback resolved "
                    f"{len(fallback_query_texts):,}/{len(missing_query_ids):,} "
                    "missing query ids.",
                )
            fallback_qid_set: set[str] = set(fallback_query_texts.keys())
            missing_query_ids = [
                qid for qid in missing_query_ids if qid not in fallback_qid_set
            ]

        inline_query_text: str
        for inline_query_text in inline_query_texts:
            query_items[make_text_key(inline_query_text)] = inline_query_text

        doc_items: dict[str, str] = {}
        corpus_id_to_idx: dict[str, int] = self.dataset.corpus_dataset_id_to_idx
        missing_doc_ids: list[str] = []
        doc_progress: tqdm | None = (
            tqdm(
                total=len(doc_ids),
                desc="Resolve doc texts",
                unit="did",
                dynamic_ncols=True,
                mininterval=1.0,
            )
            if int(get_rank()) == 0
            else None
        )
        try:
            doc_id: str
            for doc_id in doc_ids:
                corpus_idx: int = int(corpus_id_to_idx.get(doc_id, -1))
                if corpus_idx < 0:
                    missing_doc_ids.append(doc_id)
                else:
                    doc_items[make_id_key(doc_id)] = self.dataset.corpus_text(corpus_idx)
                if doc_progress is not None:
                    doc_progress.update(1)
        finally:
            if doc_progress is not None:
                doc_progress.close()

        inline_doc_text: str
        for inline_doc_text in inline_doc_texts:
            doc_items[make_text_key(inline_doc_text)] = inline_doc_text

        if (missing_query_ids or missing_doc_ids) and self._require_cache_complete:
            query_missing_preview: list[str] = missing_query_ids[:5]
            doc_missing_preview: list[str] = missing_doc_ids[:5]
            raise ValueError(
                "Pretokenization cache build found missing query/doc ids. "
                f"missing_query_ids={query_missing_preview} "
                f"missing_doc_ids={doc_missing_preview}"
            )

        return query_items, doc_items, len(query_ids), len(doc_ids)

    def _collect_cache_inputs_triplet_columns(
        self,
    ) -> tuple[dict[str, str], dict[str, str], int, int] | None:
        meta_dataset: Dataset = self.meta_dataset
        column_names: set[str] = set(meta_dataset.column_names)
        required_columns: set[str] = {"query_id", "positive_id", "negative_id"}
        if not required_columns.issubset(column_names):
            return None
        if any(
            key in column_names
            for key in ("query_text", "query", "anchor", "positive", "negative")
        ):
            return None
        log_if_rank_zero(
            logger,
            "Pretokenize input scan: using Arrow triplet-column fast path.",
        )

        query_ids: list[str] = self._collect_unique_ids_from_scalar_column(
            column_name="query_id",
            progress_desc="Collect query ids",
        )
        positive_doc_ids: list[str] = self._collect_unique_ids_from_scalar_column(
            column_name="positive_id",
            progress_desc="Collect positive doc ids",
        )
        negative_doc_ids: list[str] = self._collect_unique_ids_from_scalar_column(
            column_name="negative_id",
            progress_desc="Collect negative doc ids",
        )

        doc_ids: list[str] = list(positive_doc_ids)
        seen_doc_ids: set[str] = set(doc_ids)
        doc_id: str
        for doc_id in negative_doc_ids:
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            doc_ids.append(doc_id)

        return self._build_cache_items_from_ids(
            query_ids=query_ids,
            doc_ids=doc_ids,
            inline_query_text_by_qid={},
            inline_query_texts=[],
            inline_doc_texts=[],
        )

    def _collect_cache_inputs(
        self,
    ) -> tuple[dict[str, str], dict[str, str], int, int]:
        triplet_result: tuple[dict[str, str], dict[str, str], int, int] | None = (
            self._collect_cache_inputs_triplet_columns()
        )
        if triplet_result is not None:
            return triplet_result

        total_rows: int = int(len(self.meta_dataset))
        progress_bar: tqdm | None = (
            tqdm(
                total=total_rows,
                desc="Pretokenize inputs",
                unit="row",
                dynamic_ncols=True,
                mininterval=1.0,
            )
            if int(get_rank()) == 0
            else None
        )
        pending_progress: int = 0
        query_ids: set[str] = set()
        doc_ids: set[str] = set()
        inline_query_text_by_qid: dict[str, str] = {}
        inline_query_texts: set[str] = set()
        inline_doc_texts: set[str] = set()

        try:
            _row_idx: int
            row: dict[str, Any]
            for _row_idx, row in self._iter_meta_rows():
                qid: str | None = self._extract_row_qid(row)
                if qid is not None:
                    query_ids.add(qid)
                query_text: str | None = self._extract_row_query_text(row)
                if query_text is not None:
                    inline_query_texts.add(query_text)
                    if qid is not None and qid not in inline_query_text_by_qid:
                        inline_query_text_by_qid[qid] = query_text
                doc_ids.update(self._extract_row_doc_ids(row))
                inline_doc_texts.update(self._extract_row_inline_doc_texts(row))
                if progress_bar is not None:
                    pending_progress += 1
                    if pending_progress >= TOKENIZE_BATCH_SIZE:
                        progress_bar.update(pending_progress)
                        pending_progress = 0
            if progress_bar is not None and pending_progress > 0:
                progress_bar.update(pending_progress)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return self._build_cache_items_from_ids(
            query_ids=sorted(query_ids),
            doc_ids=sorted(doc_ids),
            inline_query_text_by_qid=inline_query_text_by_qid,
            inline_query_texts=sorted(inline_query_texts),
            inline_doc_texts=sorted(inline_doc_texts),
        )

    def _tokenize_rows(
        self,
        *,
        items: dict[str, str],
        max_length: int,
        phase_name: str,
    ) -> Iterable[tuple[str, list[int], list[int]]]:
        keys: list[str] = list(items.keys())
        total_items: int = len(keys)
        if total_items == 0:
            return
        progress_bar: tqdm | None = (
            tqdm(
                total=total_items,
                desc=f"Pretokenize {phase_name}",
                unit="item",
                dynamic_ncols=True,
                mininterval=1.0,
            )
            if int(get_rank()) == 0
            else None
        )
        padding_mode: str | bool = "max_length" if self.max_padding else False
        try:
            start: int
            for start in range(0, total_items, TOKENIZE_BATCH_SIZE):
                batch_keys: list[str] = keys[start : start + TOKENIZE_BATCH_SIZE]
                texts: list[str] = [items[key] for key in batch_keys]
                tokens: dict[str, Any] = self.tokenizer(
                    texts,
                    padding=padding_mode,
                    truncation=True,
                    max_length=max_length,
                    return_attention_mask=True,
                )
                input_ids_rows: list[list[int]] = [
                    [int(token_id) for token_id in row] for row in tokens["input_ids"]
                ]
                attention_mask_rows: list[list[int]] = [
                    [int(mask_value) for mask_value in row]
                    for row in tokens["attention_mask"]
                ]
                batch_idx: int
                key: str
                for batch_idx, key in enumerate(batch_keys):
                    yield (
                        key,
                        input_ids_rows[batch_idx],
                        attention_mask_rows[batch_idx],
                    )
                if progress_bar is not None:
                    progress_bar.update(len(batch_keys))
        finally:
            if progress_bar is not None:
                progress_bar.close()

    @staticmethod
    def _build_dataset_row_index(
        *,
        keys: Iterable[str],
        id_to_idx: Mapping[str, int],
        dataset_size: int,
    ) -> np.ndarray:
        return PretokenizeCacheStorageWriter.build_dataset_row_index(
            keys=keys,
            id_to_idx=id_to_idx,
            dataset_size=dataset_size,
        )

    def _build_dataset_row_index_from_sqlite(
        self,
        *,
        prefix: str,
        id_to_idx: Mapping[str, int],
        dataset_size: int,
        shard_size: int,
    ) -> np.ndarray:
        return self._pretokenize_writer.build_dataset_row_index_from_sqlite(
            prefix=prefix,
            id_to_idx=id_to_idx,
            dataset_size=dataset_size,
            shard_size=shard_size,
        )

    @staticmethod
    def _resolve_global_row_for_id(
        *,
        row_index: np.ndarray,
        id_to_dataset_idx: Mapping[str, int],
        identifier: str,
    ) -> int:
        dataset_idx_value: int | None = id_to_dataset_idx.get(str(identifier))
        if dataset_idx_value is None:
            return -1
        dataset_idx: int = int(dataset_idx_value)
        if dataset_idx < 0 or dataset_idx >= int(row_index.shape[0]):
            return -1
        return int(row_index[dataset_idx])

    def _build_meta_row_pointer_arrays(
        self,
        *,
        query_row_index: np.ndarray,
        doc_row_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_rows: int = int(len(self.meta_dataset))
        max_docs: int = int(self.num_positives + self.num_negatives)
        query_pointers: np.ndarray = np.full(total_rows, -1, dtype=np.int64)
        doc_pointers: np.ndarray = np.full((total_rows, max_docs), -1, dtype=np.int64)
        doc_counts: np.ndarray = np.zeros(total_rows, dtype=np.int32)
        query_id_to_idx: Mapping[str, int] = self.dataset.query_dataset_id_to_idx
        doc_id_to_idx: Mapping[str, int] = self.dataset.corpus_dataset_id_to_idx

        row_idx: int
        row: dict[str, Any]
        for row_idx, row in self._iter_meta_rows():
            qid: str | None = self._extract_row_qid(row)
            if qid:
                query_pointers[row_idx] = int(
                    self._resolve_global_row_for_id(
                        row_index=query_row_index,
                        id_to_dataset_idx=query_id_to_idx,
                        identifier=qid,
                    )
                )
            doc_ids: list[str] | None = self._extract_fixed_triplet_doc_ids(row)
            if doc_ids is None:
                continue
            doc_count: int = min(int(len(doc_ids)), int(max_docs))
            doc_counts[row_idx] = int(doc_count)
            doc_slot: int
            for doc_slot in range(doc_count):
                doc_id: str = str(doc_ids[doc_slot]).strip()
                if not doc_id:
                    continue
                doc_pointers[row_idx, doc_slot] = int(
                    self._resolve_global_row_for_id(
                        row_index=doc_row_index,
                        id_to_dataset_idx=doc_id_to_idx,
                        identifier=doc_id,
                    )
                )
        return query_pointers, doc_pointers, doc_counts

    def _ensure_row_index_artifacts(self) -> None:
        self._pretokenize_lifecycle.ensure_row_index_artifacts()

    def _ensure_meta_row_pointer_artifacts(self) -> None:
        self._pretokenize_lifecycle.ensure_meta_row_pointer_artifacts()

    def _ensure_numpy_sidecar_artifacts(self) -> None:
        self._pretokenize_lifecycle.ensure_numpy_sidecar_artifacts()

    def _expected_manifest(self) -> dict[str, Any]:
        return self._pretokenize_lifecycle.expected_manifest()

    def _cache_is_ready(self, expected_manifest: dict[str, Any]) -> bool:
        return self._pretokenize_lifecycle.cache_is_ready(expected_manifest)

    def _build_or_validate_cache(self) -> None:
        self._pretokenize_lifecycle.build_or_validate_cache()

    def _load_cache(self) -> None:
        self._pretokenize_lifecycle.load_cache()

    # --- Public methods ---
    def prepare_data(self) -> None:
        super().prepare_data()
        if not self._pretokenize_enabled:
            return
        if int(get_rank()) != 0:
            return
        self._build_or_validate_cache()

    def setup(self) -> None:
        super().setup()
        if not self._pretokenize_enabled:
            self._cache_ready = True
            return
        maybe_barrier()
        self._load_cache()
