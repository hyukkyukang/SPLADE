from pathlib import Path
from typing import Protocol

import numpy as np

from src.data.pd_module.pretokenize import iter_shard_paths, sidecar_exists


class PretokenizeCacheBackend(Protocol):
    """Protocol for row-index/sidecar artifact access."""

    def shard_paths(self, *, prefix: str) -> list[Path]:
        ...

    def missing_sidecar_shards(self, *, prefix: str) -> list[Path]:
        ...

    def load_row_index(self, *, path: Path) -> np.ndarray:
        ...


class FileSystemPretokenizeCacheBackend:
    """Filesystem implementation for cache artifact operations."""

    def __init__(self, *, cache_dir: Path) -> None:
        self._cache_dir: Path = cache_dir

    def shard_paths(self, *, prefix: str) -> list[Path]:
        return iter_shard_paths(self._cache_dir, prefix)

    def missing_sidecar_shards(self, *, prefix: str) -> list[Path]:
        return [
            shard_path
            for shard_path in self.shard_paths(prefix=prefix)
            if not sidecar_exists(shard_path)
        ]

    def load_row_index(self, *, path: Path) -> np.ndarray:
        return np.load(str(path), mmap_mode="r", allow_pickle=False)
