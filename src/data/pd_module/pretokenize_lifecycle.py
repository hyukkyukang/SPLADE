import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.data.pd_module.pretokenize_backend import (
    FileSystemPretokenizeCacheBackend,
    PretokenizeCacheBackend,
)
from src.data.pd_module.pretokenize import (
    STORAGE_FORMAT_SIDECAR_ONLY,
    acquire_build_lock,
    build_manifest,
    clear_done,
    index_exists,
    load_manifest,
    load_token_cache,
    manifests_compatible,
    mark_done,
    release_build_lock,
    remove_index,
    remove_row_index,
    remove_shards,
    resolve_done_path,
    resolve_lock_path,
    write_manifest,
    write_numpy_sidecar_from_parquet_shard,
    write_row_index,
    wait_for_done,
)
from src.utils.logging import log_if_rank_zero


class PretokenizeCacheLifecycleManager:
    """Handle pretokenized cache build/validate/load lifecycle."""

    def __init__(self, *, owner: Any, logger: logging.Logger) -> None:
        self._owner: Any = owner
        self._logger: logging.Logger = logger
        self._backend: PretokenizeCacheBackend = FileSystemPretokenizeCacheBackend(
            cache_dir=Path(owner._cache_dir)
        )

    @staticmethod
    def load_numpy_array(path: Path) -> np.ndarray | None:
        if not path.is_file():
            return None
        return np.load(str(path), mmap_mode="r", allow_pickle=False)

    @staticmethod
    def write_numpy_array(path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path = path.with_suffix(".tmp")
        with open(tmp_path, "wb") as output_file:
            np.save(output_file, array, allow_pickle=False)
        tmp_path.replace(path)

    def load_meta_row_pointer_arrays(self) -> None:
        owner: Any = self._owner
        if not owner._streaming_use_meta_row_pointer:
            owner._meta_query_row_pointers = None
            owner._meta_doc_row_pointers = None
            owner._meta_doc_counts = None
            return
        owner._meta_query_row_pointers = self.load_numpy_array(
            owner._meta_query_row_pointer_path
        )
        owner._meta_doc_row_pointers = self.load_numpy_array(
            owner._meta_doc_row_pointers_path
        )
        owner._meta_doc_counts = self.load_numpy_array(owner._meta_doc_count_path)

    def ensure_row_index_artifacts(self) -> None:
        owner: Any = self._owner
        if not owner._use_streaming_cache or not owner._streaming_use_dataset_row_index:
            return
        if not owner._query_row_index_path.is_file():
            log_if_rank_zero(
                self._logger,
                "Building query row-index map for streaming pretokenized cache.",
            )
            query_row_index: np.ndarray = (
                owner._pretokenize_writer.build_dataset_row_index_from_sqlite(
                prefix="queries",
                id_to_idx=owner.dataset.query_dataset_id_to_idx,
                dataset_size=len(owner.dataset.query_dataset),
                shard_size=owner._query_shard_size,
                )
            )
            write_row_index(
                cache_dir=owner._cache_dir,
                prefix="queries",
                row_index=query_row_index,
            )
        if not owner._doc_row_index_path.is_file():
            log_if_rank_zero(
                self._logger,
                "Building doc row-index map for streaming pretokenized cache.",
            )
            doc_row_index: np.ndarray = (
                owner._pretokenize_writer.build_dataset_row_index_from_sqlite(
                prefix="docs",
                id_to_idx=owner.dataset.corpus_dataset_id_to_idx,
                dataset_size=len(owner.dataset.corpus_dataset),
                shard_size=owner._doc_shard_size,
                )
            )
            write_row_index(
                cache_dir=owner._cache_dir,
                prefix="docs",
                row_index=doc_row_index,
            )

    def ensure_meta_row_pointer_artifacts(self) -> None:
        owner: Any = self._owner
        if not owner._use_streaming_cache or not owner._streaming_use_meta_row_pointer:
            return
        if (
            owner._meta_query_row_pointer_path.is_file()
            and owner._meta_doc_row_pointers_path.is_file()
            and owner._meta_doc_count_path.is_file()
        ):
            return
        self.ensure_row_index_artifacts()
        query_row_index: np.ndarray = self._backend.load_row_index(
            path=owner._query_row_index_path
        )
        doc_row_index: np.ndarray = self._backend.load_row_index(
            path=owner._doc_row_index_path
        )
        log_if_rank_zero(
            self._logger,
            "Building per-meta-row global pointers for streaming pretokenized cache.",
        )
        (
            meta_query_row_pointers,
            meta_doc_row_pointers,
            meta_doc_counts,
        ) = owner._build_meta_row_pointer_arrays(
            query_row_index=query_row_index,
            doc_row_index=doc_row_index,
        )
        self.write_numpy_array(owner._meta_query_row_pointer_path, meta_query_row_pointers)
        self.write_numpy_array(owner._meta_doc_row_pointers_path, meta_doc_row_pointers)
        self.write_numpy_array(owner._meta_doc_count_path, meta_doc_counts)

    def ensure_numpy_sidecar_artifacts(self) -> None:
        owner: Any = self._owner
        if not owner._use_streaming_cache or not owner._streaming_numpy_sidecar:
            return
        if owner._pretokenize_storage_format == STORAGE_FORMAT_SIDECAR_ONLY:
            prefix: str
            for prefix in ("queries", "docs"):
                missing_sidecars: list[Path] = self._backend.missing_sidecar_shards(
                    prefix=prefix
                )
                if missing_sidecars:
                    raise FileNotFoundError(
                        "sidecar_only cache is missing sidecar files for "
                        f"{prefix}: {[path.name for path in missing_sidecars[:3]]}"
                    )
            return
        for prefix in ("queries", "docs"):
            missing_sidecar_shards: list[Path] = self._backend.missing_sidecar_shards(
                prefix=prefix
            )
            if not missing_sidecar_shards:
                continue
            log_if_rank_zero(
                self._logger,
                "Building NumPy sidecars for streaming pretokenized cache "
                f"({prefix}: {len(missing_sidecar_shards):,} shards).",
            )
            shard_path: Path
            for shard_path in missing_sidecar_shards:
                write_numpy_sidecar_from_parquet_shard(shard_path)

    def expected_manifest(self) -> dict[str, Any]:
        owner: Any = self._owner
        return build_manifest(
            {
                "dataset_name": owner.name,
                "dataset_split": str(owner.cfg.get("split")),
                "cache_namespace": owner._cache_namespace,
                "meta_len": int(len(owner.meta_dataset)),
                "model_name": str(owner.tokenizer.name_or_path),
                "use_fast_tokenizer": bool(owner.tokenizer.is_fast),
                "max_query_length": int(owner.max_query_length),
                "max_doc_length": int(owner.max_doc_length),
                "max_padding": bool(owner.max_padding),
                "num_positives": int(owner.num_positives),
                "num_negatives": int(owner.num_negatives),
                "pretokenize_loading_mode": owner._pretokenize_loading_mode,
                "pretokenize_query_shard_size": int(owner._query_shard_size),
                "pretokenize_doc_shard_size": int(owner._doc_shard_size),
                "pretokenize_write_dtype": owner._write_dtype,
                "pretokenize_storage_format": owner._pretokenize_storage_format,
                "pretokenize_index_backend": owner._streaming_index_backend,
                "pretokenize_streaming_use_dataset_row_index": bool(
                    owner._streaming_use_dataset_row_index
                ),
                "pretokenize_streaming_use_meta_row_pointer": bool(
                    owner._streaming_use_meta_row_pointer
                ),
                "pretokenize_parquet_row_group_size": owner._parquet_row_group_size,
            }
        )

    def cache_is_ready(self, expected_manifest: dict[str, Any]) -> bool:
        owner: Any = self._owner
        existing_manifest: dict[str, Any] | None = load_manifest(owner._cache_dir)
        done_path: Path = resolve_done_path(owner._cache_dir)
        if existing_manifest is None or not done_path.is_file():
            return False
        if not manifests_compatible(existing_manifest, expected_manifest):
            return False
        has_query_shards: bool = bool(self._backend.shard_paths(prefix="queries"))
        has_doc_shards: bool = bool(self._backend.shard_paths(prefix="docs"))
        if not (has_query_shards and has_doc_shards):
            return False
        if owner._use_streaming_cache:
            has_query_index: bool = index_exists(owner._cache_dir, "queries")
            has_doc_index: bool = index_exists(owner._cache_dir, "docs")
            return has_query_index and has_doc_index
        return True

    def build_or_validate_cache(self) -> None:
        owner: Any = self._owner
        expected_manifest: dict[str, Any] = self.expected_manifest()
        existing_manifest: dict[str, Any] | None = load_manifest(owner._cache_dir)
        done_path: Path = resolve_done_path(owner._cache_dir)
        lock_path: Path = resolve_lock_path(owner._cache_dir)

        if self.cache_is_ready(expected_manifest):
            log_if_rank_zero(
                self._logger, f"Pretokenized cache hit for {owner._cache_dir.as_posix()}."
            )
            self.ensure_row_index_artifacts()
            self.ensure_meta_row_pointer_artifacts()
            self.ensure_numpy_sidecar_artifacts()
            return
        if (
            existing_manifest is not None
            and not manifests_compatible(existing_manifest, expected_manifest)
            and not owner._pretokenize_overwrite
        ):
            raise ValueError(
                "Pretokenize manifest mismatch and overwrite is disabled. "
                f"cache_dir={owner._cache_dir.as_posix()}"
            )

        if not acquire_build_lock(lock_path):
            log_if_rank_zero(
                self._logger,
                "Pretokenized cache lock is held by another process. "
                f"Waiting for completion: {done_path.as_posix()}",
            )
            wait_for_done(done_path)
            return

        try:
            if self.cache_is_ready(expected_manifest):
                return
            log_if_rank_zero(
                self._logger, f"Building pretokenized cache at {owner._cache_dir.as_posix()}."
            )
            clear_done(done_path)
            remove_shards(owner._cache_dir, "queries")
            remove_shards(owner._cache_dir, "docs")
            remove_index(owner._cache_dir, "queries")
            remove_index(owner._cache_dir, "docs")
            if owner._streaming_use_dataset_row_index:
                remove_row_index(owner._cache_dir, "queries")
                remove_row_index(owner._cache_dir, "docs")
            if owner._streaming_use_meta_row_pointer:
                for pointer_path in (
                    owner._meta_query_row_pointer_path,
                    owner._meta_doc_row_pointers_path,
                    owner._meta_doc_count_path,
                ):
                    if pointer_path.is_file():
                        pointer_path.unlink()

            query_items: dict[str, str]
            doc_items: dict[str, str]
            num_query_ids: int
            num_doc_ids: int
            query_items, doc_items, num_query_ids, num_doc_ids = (
                owner._collect_cache_inputs()
            )
            query_count: int
            doc_count: int
            query_row_index: np.ndarray | None
            doc_row_index: np.ndarray | None
            (
                query_count,
                doc_count,
                query_row_index,
                doc_row_index,
            ) = owner._pretokenize_writer.write_cache_entries(
                query_items=query_items,
                doc_items=doc_items,
            )
            if (
                owner._streaming_use_meta_row_pointer
                and query_row_index is not None
                and doc_row_index is not None
            ):
                (
                    meta_query_row_pointers,
                    meta_doc_row_pointers,
                    meta_doc_counts,
                ) = owner._build_meta_row_pointer_arrays(
                    query_row_index=query_row_index,
                    doc_row_index=doc_row_index,
                )
                self.write_numpy_array(
                    owner._meta_query_row_pointer_path, meta_query_row_pointers
                )
                self.write_numpy_array(
                    owner._meta_doc_row_pointers_path, meta_doc_row_pointers
                )
                self.write_numpy_array(owner._meta_doc_count_path, meta_doc_counts)
            manifest_to_write: dict[str, Any] = dict(expected_manifest)
            manifest_to_write["query_cache_count"] = int(query_count)
            manifest_to_write["doc_cache_count"] = int(doc_count)
            manifest_to_write["query_id_candidates"] = int(num_query_ids)
            manifest_to_write["doc_id_candidates"] = int(num_doc_ids)
            write_manifest(owner._cache_dir, manifest_to_write)
            mark_done(done_path)
            log_if_rank_zero(
                self._logger,
                "Pretokenized cache build complete: "
                f"{owner._cache_dir.as_posix()} "
                f"(queries={query_count:,}, docs={doc_count:,}).",
            )
        finally:
            release_build_lock(lock_path)

    def load_cache(self) -> None:
        owner: Any = self._owner
        expected_manifest: dict[str, Any] = self.expected_manifest()
        if not self.cache_is_ready(expected_manifest):
            if owner._require_cache_complete:
                raise ValueError(
                    "Pretokenized cache is missing or incompatible. "
                    f"cache_dir={owner._cache_dir.as_posix()}"
                )
            owner._query_token_cache = {}
            owner._doc_token_cache = {}
            owner._cache_ready = False
            owner._cache_owner_pid = None
            owner._meta_query_row_pointers = None
            owner._meta_doc_row_pointers = None
            owner._meta_doc_counts = None
            return
        owner._close_token_stores()
        if owner._use_streaming_cache:
            self.ensure_row_index_artifacts()
            self.ensure_meta_row_pointer_artifacts()
            self.ensure_numpy_sidecar_artifacts()
            owner._query_token_cache = {}
            owner._doc_token_cache = {}
            self.load_meta_row_pointer_arrays()
            owner._cache_owner_pid = None
        else:
            owner._query_token_cache = load_token_cache(
                cache_dir=owner._cache_dir, prefix="queries"
            )
            owner._doc_token_cache = load_token_cache(
                cache_dir=owner._cache_dir, prefix="docs"
            )
            owner._meta_query_row_pointers = None
            owner._meta_doc_row_pointers = None
            owner._meta_doc_counts = None
        owner._cache_ready = True
