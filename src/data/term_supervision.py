from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, cast

from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.dataset import BaseDataset
from src.utils.logging import get_logger

logger = get_logger("OrderedMaskSlotTermSupervisor")

_WORKER_DATASET: Any | None = None
_WORKER_TEXT_COLUMN_NAME: str | None = None
_WORKER_TOKENIZER: PreTrainedTokenizerBase | None = None
_WORKER_VOCAB_SIZE: int = 0
_WORKER_EXCLUDED_TOKEN_ID_MASK: np.ndarray | None = None
_WORKER_BATCH_SIZE: int = 0


@dataclass(frozen=True)
class _DatasetLoadSpec:
    hf_name: str
    hf_subset: str | None
    split: str
    cache_dir: str | None
    data_files: dict[str, Any] | None
    text_column_name: str


def _load_dataset_from_spec(spec: _DatasetLoadSpec) -> Any:
    if spec.data_files:
        return load_dataset(
            spec.hf_name,
            name=spec.hf_subset,
            split=spec.split,
            cache_dir=spec.cache_dir,
            data_files=dict(spec.data_files),
        )
    return load_dataset(
        spec.hf_name,
        name=spec.hf_subset,
        split=spec.split,
        cache_dir=spec.cache_dir,
    )


def _filtered_token_ids_np_static(
    token_ids: list[int],
    *,
    vocab_size: int,
    excluded_token_id_mask: np.ndarray,
) -> np.ndarray:
    if not token_ids:
        return np.empty((0,), dtype=np.int32)
    token_array = np.fromiter(token_ids, dtype=np.int32, count=len(token_ids))
    valid_mask = (token_array >= 0) & (token_array < vocab_size)
    token_array = token_array[valid_mask]
    if token_array.size == 0:
        return token_array
    if excluded_token_id_mask.any():
        token_array = token_array[~excluded_token_id_mask[token_array]]
    return token_array


def _idf_worker_init(
    spec: _DatasetLoadSpec,
    tokenizer_name_or_path: str,
    vocab_size: int,
    excluded_token_ids: list[int],
    batch_size: int,
) -> None:
    global _WORKER_DATASET
    global _WORKER_TEXT_COLUMN_NAME
    global _WORKER_TOKENIZER
    global _WORKER_VOCAB_SIZE
    global _WORKER_EXCLUDED_TOKEN_ID_MASK
    global _WORKER_BATCH_SIZE

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    _WORKER_DATASET = _load_dataset_from_spec(spec)
    _WORKER_TEXT_COLUMN_NAME = spec.text_column_name
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        local_files_only=True,
    )
    _WORKER_VOCAB_SIZE = int(vocab_size)
    excluded_mask = np.zeros(_WORKER_VOCAB_SIZE, dtype=np.bool_)
    if excluded_token_ids:
        excluded_mask[np.asarray(excluded_token_ids, dtype=np.int64)] = True
    _WORKER_EXCLUDED_TOKEN_ID_MASK = excluded_mask
    _WORKER_BATCH_SIZE = max(int(batch_size), 1)


def _idf_worker_compute_doc_freq(
    start_index: int,
    end_index: int,
) -> tuple[np.ndarray, int]:
    if (
        _WORKER_DATASET is None
        or _WORKER_TEXT_COLUMN_NAME is None
        or _WORKER_TOKENIZER is None
        or _WORKER_EXCLUDED_TOKEN_ID_MASK is None
    ):
        raise RuntimeError("Ordered mask-slot IDF worker was not initialized.")
    doc_freq = np.zeros(_WORKER_VOCAB_SIZE, dtype=np.float64)
    processed_docs: int = 0
    local_start: int
    for local_start in range(int(start_index), int(end_index), _WORKER_BATCH_SIZE):
        local_end: int = min(local_start + _WORKER_BATCH_SIZE, int(end_index))
        batch_rows: Any = _WORKER_DATASET[local_start:local_end]
        batch_texts: list[str] = [
            str(text) for text in batch_rows[_WORKER_TEXT_COLUMN_NAME]
        ]
        encoded = _WORKER_TOKENIZER(
            batch_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        unique_token_batches: list[np.ndarray] = []
        token_id_row: list[int]
        for token_id_row in encoded["input_ids"]:
            filtered_token_ids = _filtered_token_ids_np_static(
                token_id_row,
                vocab_size=_WORKER_VOCAB_SIZE,
                excluded_token_id_mask=_WORKER_EXCLUDED_TOKEN_ID_MASK,
            )
            if filtered_token_ids.size == 0:
                processed_docs += 1
                continue
            unique_token_batches.append(np.unique(filtered_token_ids))
            processed_docs += 1
        if unique_token_batches:
            flattened_unique_ids = np.concatenate(unique_token_batches)
            doc_freq += np.bincount(flattened_unique_ids, minlength=_WORKER_VOCAB_SIZE)
    return doc_freq, processed_docs


class OrderedMaskSlotTermSupervisor:
    """Build fixed-vocabulary ordered TF-IDF targets for mask-slot supervision."""

    def __init__(
        self,
        *,
        dataset: BaseDataset,
        tokenizer: PreTrainedTokenizerBase,
        cache_dir: str | None,
        excluded_token_ids: torch.Tensor | None = None,
        idf_batch_size: int = 1024,
        idf_log_interval: int = 100_000,
        cache_wait_timeout_seconds: float = 7200.0,
        idf_num_workers: int = 0,
        idf_shards_per_worker: int = 4,
    ) -> None:
        self.dataset: BaseDataset = dataset
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.vocab_size: int = int(len(tokenizer))
        self.cache_dir: Path = Path(
            cache_dir or ".cache/ordered_mask_slot_term_supervision"
        ).expanduser()
        excluded_ids: torch.Tensor
        if excluded_token_ids is None:
            excluded_ids = torch.tensor(
                [int(token_id) for token_id in tokenizer.all_special_ids],
                dtype=torch.long,
            )
        else:
            excluded_ids = excluded_token_ids.to(dtype=torch.long).flatten().cpu()
        self.excluded_token_ids: torch.Tensor = torch.unique(excluded_ids)
        self.idf_batch_size: int = max(int(idf_batch_size), 1)
        self.idf_log_interval: int = max(int(idf_log_interval), 1)
        self.cache_wait_timeout_seconds: float = max(
            float(cache_wait_timeout_seconds), 1.0
        )
        self.idf_num_workers: int = int(idf_num_workers)
        self.idf_shards_per_worker: int = max(int(idf_shards_per_worker), 1)
        self._query_idf: torch.Tensor | None = None
        self._doc_idf: torch.Tensor | None = None
        excluded_mask = np.zeros(self.vocab_size, dtype=np.bool_)
        if int(self.excluded_token_ids.numel()) > 0:
            excluded_mask[self.excluded_token_ids.numpy()] = True
        self._excluded_token_id_mask: np.ndarray = excluded_mask

    def prepare(self) -> None:
        if self._query_idf is not None and self._doc_idf is not None:
            return
        cache_path: Path = self._cache_path()
        if cache_path.is_file():
            self._load_from_cache(cache_path)
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path: Path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
        if self._try_acquire_lock(lock_path):
            try:
                if not cache_path.is_file():
                    self._build_and_store_cache(cache_path)
            finally:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
        else:
            self._wait_for_cache(cache_path=cache_path, lock_path=lock_path)
        self._load_from_cache(cache_path)

    def top_k_query_target_ids(
        self,
        text: str,
        *,
        k: int,
        ignore_index: int,
    ) -> torch.Tensor:
        self.prepare()
        return self._top_k_target_ids(
            text,
            idf=cast(torch.Tensor, self._query_idf),
            k=k,
            ignore_index=ignore_index,
        )

    def top_k_doc_target_ids(
        self,
        text: str,
        *,
        k: int,
        ignore_index: int,
    ) -> torch.Tensor:
        self.prepare()
        return self._top_k_target_ids(
            text,
            idf=cast(torch.Tensor, self._doc_idf),
            k=k,
            ignore_index=ignore_index,
        )

    def _cache_path(self) -> Path:
        query_spec: _DatasetLoadSpec | None = self._query_dataset_spec()
        corpus_spec: _DatasetLoadSpec | None = self._corpus_dataset_spec()
        cache_key_payload: dict[str, Any] = {
            "query_dataset": (
                None
                if query_spec is None
                else {
                    "hf_name": query_spec.hf_name,
                    "hf_subset": query_spec.hf_subset,
                    "split": query_spec.split,
                    "cache_dir": query_spec.cache_dir,
                    "data_files": query_spec.data_files,
                    "text_column_name": query_spec.text_column_name,
                }
            ),
            "corpus_dataset": (
                None
                if corpus_spec is None
                else {
                    "hf_name": corpus_spec.hf_name,
                    "hf_subset": corpus_spec.hf_subset,
                    "split": corpus_spec.split,
                    "cache_dir": corpus_spec.cache_dir,
                    "data_files": corpus_spec.data_files,
                    "text_column_name": corpus_spec.text_column_name,
                }
            ),
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", None),
            "vocab_size": self.vocab_size,
            "excluded_token_ids": [
                int(token_id) for token_id in self.excluded_token_ids.tolist()
            ],
        }
        digest: str = hashlib.sha256(
            json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return self.cache_dir / f"tfidf_{digest}.pt"

    def _load_from_cache(self, cache_path: Path) -> None:
        payload: dict[str, Any] = cast(
            dict[str, Any], torch.load(cache_path, map_location="cpu")
        )
        self._query_idf = payload["query_idf"].to(dtype=torch.float32)
        self._doc_idf = payload["doc_idf"].to(dtype=torch.float32)

    @staticmethod
    def _try_acquire_lock(lock_path: Path) -> bool:
        try:
            fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            return False
        try:
            os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        finally:
            os.close(fd)
        return True

    @staticmethod
    def _read_lock_owner_pid(lock_path: Path) -> int | None:
        try:
            raw_value: str = lock_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if not raw_value:
            return None
        try:
            return int(raw_value)
        except ValueError:
            return None

    @staticmethod
    def _pid_is_alive(pid: int | None) -> bool:
        if pid is None or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _clear_stale_lock(self, lock_path: Path) -> bool:
        owner_pid: int | None = self._read_lock_owner_pid(lock_path)
        if self._pid_is_alive(owner_pid):
            return False
        try:
            lock_path.unlink()
        except FileNotFoundError:
            return False
        logger.warning(
            "Removed stale ordered-mask-slot TF-IDF cache lock at %s (pid=%s)",
            lock_path,
            "unknown" if owner_pid is None else str(owner_pid),
        )
        return True

    def _build_and_store_cache(self, cache_path: Path) -> None:
        logger.info(
            "Building ordered-mask-slot TF-IDF cache at %s "
            "(query_batch_size=%d, corpus_batch_size=%d)",
            cache_path,
            self.idf_batch_size,
            self.idf_batch_size,
        )
        query_idf: torch.Tensor = self._build_idf_for_queries()
        doc_idf: torch.Tensor = self._build_idf_for_corpus()
        payload = {
            "query_idf": query_idf,
            "doc_idf": doc_idf,
            "vocab_size": self.vocab_size,
            "excluded_token_ids": self.excluded_token_ids,
        }
        tmp_path: Path = cache_path.with_suffix(f"{cache_path.suffix}.tmp.{os.getpid()}")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
        logger.info("Finished building ordered-mask-slot TF-IDF cache at %s", cache_path)

    def _wait_for_cache(self, *, cache_path: Path, lock_path: Path) -> None:
        deadline: float = time.monotonic() + self.cache_wait_timeout_seconds
        while time.monotonic() < deadline:
            if cache_path.is_file():
                return
            if lock_path.exists():
                self._clear_stale_lock(lock_path)
            if not lock_path.exists() and self._try_acquire_lock(lock_path):
                try:
                    if not cache_path.is_file():
                        self._build_and_store_cache(cache_path)
                    return
                finally:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
            time.sleep(1.0)
        raise TimeoutError(
            "Timed out waiting for ordered mask-slot TF-IDF cache at "
            f"{cache_path}."
        )

    def _filtered_token_ids_np(self, token_ids: list[int]) -> np.ndarray:
        return _filtered_token_ids_np_static(
            token_ids,
            vocab_size=self.vocab_size,
            excluded_token_id_mask=self._excluded_token_id_mask,
        )

    def _token_ids_for_text(self, text: str) -> torch.Tensor:
        encoded: dict[str, list[int]] = self.tokenizer(
            str(text),
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        token_ids_np: np.ndarray = self._filtered_token_ids_np(encoded["input_ids"])
        if token_ids_np.size == 0:
            return torch.empty((0,), dtype=torch.long)
        return torch.from_numpy(token_ids_np.astype(np.int64, copy=False))

    def _resolve_worker_count(self, total_docs: int) -> int:
        if total_docs <= 0:
            return 1
        if not bool(getattr(self.dataset, "use_hf", False)):
            return 1
        tokenizer_name_or_path: str = str(
            getattr(self.tokenizer, "name_or_path", "")
        ).strip()
        if not tokenizer_name_or_path:
            return 1
        configured_workers: int = int(self.idf_num_workers)
        if configured_workers > 0:
            return max(1, min(configured_workers, total_docs))
        cpu_count: int = os.cpu_count() or 1
        return max(1, min(cpu_count, total_docs))

    def _query_dataset_spec(self) -> _DatasetLoadSpec | None:
        if not bool(getattr(self.dataset, "use_hf", False)):
            return None
        hf_name: str = str(self.dataset.huggingface_name)
        cache_dir: str | None = (
            self.dataset.query_corpus_hf_cache_dir
            if self.dataset.query_corpus_hf_cache_dir is not None
            else self.dataset.hf_cache_dir
        )
        data_files = (
            self.dataset.query_hf_data_files
            if self.dataset.query_hf_data_files is not None
            else self.dataset.query_corpus_hf_data_files
        )
        return _DatasetLoadSpec(
            hf_name=hf_name,
            hf_subset=self.dataset.query_column_names["query_subset_name"],
            split=self.dataset.query_column_names["query_split_name"],
            cache_dir=cache_dir,
            data_files=None if data_files is None else dict(data_files),
            text_column_name=self.dataset.query_text_column_name,
        )

    def _corpus_dataset_spec(self) -> _DatasetLoadSpec | None:
        if not bool(getattr(self.dataset, "use_hf", False)):
            return None
        hf_name: str = str(self.dataset.huggingface_name)
        cache_dir: str | None = (
            self.dataset.query_corpus_hf_cache_dir
            if self.dataset.query_corpus_hf_cache_dir is not None
            else self.dataset.hf_cache_dir
        )
        data_files = (
            self.dataset.corpus_hf_data_files
            if self.dataset.corpus_hf_data_files is not None
            else self.dataset.query_corpus_hf_data_files
        )
        return _DatasetLoadSpec(
            hf_name=hf_name,
            hf_subset=self.dataset.corpus_column_names["corpus_subset_name"],
            split=self.dataset.corpus_column_names["corpus_split_name"],
            cache_dir=cache_dir,
            data_files=None if data_files is None else dict(data_files),
            text_column_name=self.dataset.corpus_text_column_name,
        )

    def _log_idf_progress(
        self,
        *,
        description: str,
        processed_count: int,
        total_docs: int,
        started_at: float,
    ) -> None:
        elapsed_seconds: float = max(time.monotonic() - started_at, 1e-6)
        docs_per_second: float = processed_count / elapsed_seconds
        logger.info(
            "Building %s IDF cache: %d / %d texts (%.1f%%, %.1f texts/s)",
            description,
            processed_count,
            total_docs,
            100.0 * processed_count / total_docs,
            docs_per_second,
        )

    def _build_idf_from_loaded_dataset(
        self,
        *,
        dataset: Any,
        text_column_name: str,
        count: int,
        description: str,
    ) -> torch.Tensor:
        total_docs: int = int(count)
        doc_freq = np.zeros(self.vocab_size, dtype=np.float64)
        started_at: float = time.monotonic()
        for start_index in range(0, total_docs, self.idf_batch_size):
            end_index: int = min(start_index + self.idf_batch_size, total_docs)
            batch_rows: Any = dataset[start_index:end_index]
            batch_texts: list[str] = [str(text) for text in batch_rows[text_column_name]]
            encoded = self.tokenizer(
                batch_texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            unique_token_batches: list[np.ndarray] = []
            token_id_row: list[int]
            for token_id_row in encoded["input_ids"]:
                filtered_token_ids = self._filtered_token_ids_np(token_id_row)
                if filtered_token_ids.size == 0:
                    continue
                unique_token_batches.append(np.unique(filtered_token_ids))
            if unique_token_batches:
                flattened_unique_ids = np.concatenate(unique_token_batches)
                doc_freq += np.bincount(flattened_unique_ids, minlength=self.vocab_size)
            processed_count: int = end_index
            if (
                processed_count == total_docs
                or processed_count % self.idf_log_interval == 0
            ):
                self._log_idf_progress(
                    description=description,
                    processed_count=processed_count,
                    total_docs=total_docs,
                    started_at=started_at,
                )
        num_docs_value: float = float(total_docs)
        idf = np.log((1.0 + num_docs_value) / (1.0 + doc_freq)) + 1.0
        if int(self.excluded_token_ids.numel()) > 0:
            idf[self.excluded_token_ids.numpy()] = 0.0
        return torch.from_numpy(idf.astype(np.float32, copy=False))

    def _build_idf_from_text_iterator(
        self,
        *,
        dataset: Any,
        dataset_spec: _DatasetLoadSpec | None,
        text_column_name: str,
        count: int,
        description: str,
    ) -> torch.Tensor:
        total_docs: int = int(count)
        if total_docs <= 0:
            idf = np.ones(self.vocab_size, dtype=np.float32)
            if int(self.excluded_token_ids.numel()) > 0:
                idf[self.excluded_token_ids.numpy()] = 0.0
            return torch.from_numpy(idf)
        worker_count: int = self._resolve_worker_count(total_docs)
        tokenizer_name_or_path: str = str(
            getattr(self.tokenizer, "name_or_path", "")
        ).strip()
        if worker_count <= 1 or dataset_spec is None or not tokenizer_name_or_path:
            return self._build_idf_from_loaded_dataset(
                dataset=dataset,
                text_column_name=text_column_name,
                count=total_docs,
                description=description,
            )

        task_count: int = min(
            total_docs,
            max(worker_count * self.idf_shards_per_worker, worker_count),
        )
        chunk_size: int = max((total_docs + task_count - 1) // task_count, 1)
        shard_ranges: list[tuple[int, int]] = [
            (start_index, min(start_index + chunk_size, total_docs))
            for start_index in range(0, total_docs, chunk_size)
        ]
        logger.info(
            "Building %s IDF cache with %d workers across %d shards",
            description,
            worker_count,
            len(shard_ranges),
        )
        doc_freq = np.zeros(self.vocab_size, dtype=np.float64)
        processed_count: int = 0
        started_at: float = time.monotonic()
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
            initializer=_idf_worker_init,
            initargs=(
                dataset_spec,
                tokenizer_name_or_path,
                self.vocab_size,
                [int(token_id) for token_id in self.excluded_token_ids.tolist()],
                self.idf_batch_size,
            ),
        ) as executor:
            futures = [
                executor.submit(_idf_worker_compute_doc_freq, start_index, end_index)
                for start_index, end_index in shard_ranges
            ]
            for future in as_completed(futures):
                shard_doc_freq, shard_processed_count = future.result()
                doc_freq += shard_doc_freq
                previous_processed_count: int = processed_count
                processed_count += int(shard_processed_count)
                if (
                    processed_count == total_docs
                    or processed_count // self.idf_log_interval
                    > previous_processed_count // self.idf_log_interval
                ):
                    self._log_idf_progress(
                        description=description,
                        processed_count=processed_count,
                        total_docs=total_docs,
                        started_at=started_at,
                    )

        num_docs_value: float = float(total_docs)
        idf = np.log((1.0 + num_docs_value) / (1.0 + doc_freq)) + 1.0
        if int(self.excluded_token_ids.numel()) > 0:
            idf[self.excluded_token_ids.numpy()] = 0.0
        return torch.from_numpy(idf.astype(np.float32, copy=False))

    def _build_idf_for_queries(self) -> torch.Tensor:
        query_dataset = self.dataset.query_dataset
        query_count: int = int(len(query_dataset))
        return self._build_idf_from_text_iterator(
            dataset=query_dataset,
            dataset_spec=self._query_dataset_spec(),
            text_column_name=self.dataset.query_text_column_name,
            count=query_count,
            description="query",
        )

    def _build_idf_for_corpus(self) -> torch.Tensor:
        corpus_dataset = self.dataset.corpus_dataset
        corpus_count: int = int(len(corpus_dataset))
        return self._build_idf_from_text_iterator(
            dataset=corpus_dataset,
            dataset_spec=self._corpus_dataset_spec(),
            text_column_name=self.dataset.corpus_text_column_name,
            count=corpus_count,
            description="corpus",
        )

    def _top_k_target_ids(
        self,
        text: str,
        *,
        idf: torch.Tensor,
        k: int,
        ignore_index: int,
    ) -> torch.Tensor:
        resolved_k: int = max(int(k), 0)
        if resolved_k <= 0:
            return torch.empty((0,), dtype=torch.long)
        target_ids: torch.Tensor = torch.full(
            (resolved_k,),
            int(ignore_index),
            dtype=torch.long,
        )
        token_ids: torch.Tensor = self._token_ids_for_text(text)
        if int(token_ids.numel()) == 0:
            return target_ids
        tf: torch.Tensor = torch.bincount(
            token_ids,
            minlength=self.vocab_size,
        ).to(dtype=torch.float32)
        scores: torch.Tensor = tf * idf
        if int(self.excluded_token_ids.numel()) > 0:
            scores = scores.clone()
            scores.index_fill_(0, self.excluded_token_ids, 0.0)
        positive_mask: torch.Tensor = scores > 0
        if not bool(positive_mask.any()):
            return target_ids
        candidate_indices: torch.Tensor = torch.nonzero(
            positive_mask,
            as_tuple=False,
        ).flatten()
        candidate_scores: torch.Tensor = scores[candidate_indices]
        sorted_order: torch.Tensor = torch.argsort(candidate_scores, descending=True)
        top_candidate_indices: torch.Tensor = candidate_indices[
            sorted_order[:resolved_k]
        ]
        target_count: int = int(top_candidate_indices.numel())
        if target_count > 0:
            target_ids[:target_count] = top_candidate_indices.to(dtype=torch.long)
        return target_ids
