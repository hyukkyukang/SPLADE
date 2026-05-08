"""Loader for ``cfli/bge-full-data`` — the multi-task training mix used by
LENS (arXiv 2501.09749) and BGE-EN-ICL.

The HF dataset ships with multiple splits, each keyed by a *task family*
(retrieval, reranking, STS, clustering, classification, …). Every row has the
common schema::

    {
      "query":  str,
      "pos":    list[str],
      "neg":    list[str],
      "pos_scores": list[float] | None,
      "neg_scores": list[float] | None,
      "prompt": str,              # per-row instruction
      "type":   str,              # family identifier used by sampling/loss logic
    }

We concatenate all splits into a single flat table to make ``__getitem__``
O(1) and cheap, while remembering which row range belongs to each task (so
the :class:`SameDatasetBatchSampler` can carve task-pure batches).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BgeFullDataMeta:
    """Metadata describing how rows are grouped by task family."""

    data_names: List[str]
    row_ranges: List[np.ndarray]  # one int64 array per task, global-row indices

    def total_rows(self) -> int:
        return int(sum(arr.size for arr in self.row_ranges))


class BgeFullDataset:
    """Thin wrapper around the concatenated HF dataset.

    Loading order:

    1. If ``local_dir`` exists and contains a ``dataset_info.json`` / ``load_from_disk``
       compatible layout, use it (no network).
    2. Otherwise fall back to ``datasets.load_dataset("cfli/bge-full-data",
       cache_dir=cache_dir)`` and, if ``save_to_disk`` is true, materialize it.

    The resulting object exposes ``__len__`` and ``__getitem__`` plus a
    :attr:`meta` attribute with per-task row ranges, mirroring the bookkeeping
    the official :class:`SameDatasetTrainDataset` performs on-the-fly.
    """

    DEFAULT_HF_NAME: str = "cfli/bge-full-data"

    def __init__(
        self,
        local_dir: str | os.PathLike[str] | None = None,
        *,
        hf_name: str = DEFAULT_HF_NAME,
        cache_dir: str | os.PathLike[str] | None = None,
        save_to_disk: bool = False,
        save_to_disk_path: str | os.PathLike[str] | None = None,
    ) -> None:
        import datasets  # lazy import

        self.hf_name: str = hf_name
        self.local_dir: Path | None = (
            Path(local_dir).expanduser() if local_dir is not None else None
        )
        self.cache_dir: str | None = (
            str(Path(cache_dir).expanduser()) if cache_dir is not None else None
        )

        loaded_from_disk: bool = False
        if self.local_dir is not None and self.local_dir.is_dir():
            try:
                concat = datasets.load_from_disk(str(self.local_dir))
                if isinstance(concat, datasets.DatasetDict):
                    self._raw_dict = concat
                    self._dataset = datasets.concatenate_datasets(list(concat.values()))
                    self._splits_order: List[str] = list(concat.keys())
                    loaded_from_disk = True
                elif isinstance(concat, datasets.Dataset):
                    # Pre-concatenated Dataset; require a sibling meta.json for ranges.
                    self._dataset = concat
                    self._splits_order = ["_all_"]
                    loaded_from_disk = True
                else:
                    raise TypeError(
                        f"Unexpected load_from_disk payload: {type(concat).__name__}"
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to load BGE-full-data from %s (%s); falling back to HF hub",
                    self.local_dir,
                    exc,
                )
                loaded_from_disk = False

        if not loaded_from_disk:
            logger.info("Loading %s from HuggingFace hub", hf_name)
            ds_dict = datasets.load_dataset(hf_name, cache_dir=self.cache_dir)
            if not isinstance(ds_dict, datasets.DatasetDict):
                raise TypeError(
                    "cfli/bge-full-data is expected to be a DatasetDict."
                )
            self._raw_dict = ds_dict
            self._splits_order = list(ds_dict.keys())
            self._dataset = datasets.concatenate_datasets(
                list(ds_dict.values())
            )
            if save_to_disk:
                target: Path = Path(
                    save_to_disk_path or (self.local_dir or ".")
                ).expanduser()
                target.mkdir(parents=True, exist_ok=True)
                ds_dict.save_to_disk(str(target))
                logger.info("Saved cfli/bge-full-data to %s", target)

        # Build the row-range metadata
        ranges: List[np.ndarray] = []
        offset: int = 0
        name: str
        for name in self._splits_order:
            count: int = (
                len(self._raw_dict[name])
                if hasattr(self, "_raw_dict")
                else len(self._dataset)
            )
            ranges.append(np.arange(offset, offset + count, dtype=np.int64))
            offset += count
        self.meta: BgeFullDataMeta = BgeFullDataMeta(
            data_names=list(self._splits_order),
            row_ranges=ranges,
        )

    # --- basic sequence interface -------------------------------------------

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int | Sequence[int]) -> Mapping[str, Any]:
        return self._dataset[idx]

    @property
    def column_names(self) -> List[str]:
        return list(self._dataset.column_names)

    def raw_split(self, name: str) -> Any:
        if not hasattr(self, "_raw_dict"):
            raise RuntimeError("Dataset was loaded pre-concatenated; raw splits unavailable.")
        return self._raw_dict[name]


__all__ = ["BgeFullDataset", "BgeFullDataMeta"]
