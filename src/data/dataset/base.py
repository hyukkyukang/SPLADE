from __future__ import annotations

import abc
import os
from typing import Any, Optional
from omegaconf import DictConfig
from torch.utils.data import get_worker_info

from src.data.dataclass import DataTuple, RerankingDataItem, RetrievalDataItem


class BaseDataset(abc.ABC):
    """Abstract base class for datasets."""

    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        tokenizer,
    ) -> None:
        self.cfg: DictConfig = cfg
        self.global_cfg: DictConfig = global_cfg
        self.tokenizer: Any = tokenizer
        self.name: str = str(self.cfg.name)
        self.data_instances: Optional[list[DataTuple]] = None

    def __len__(self) -> int:
        return len(self.data_instances or [])

    @abc.abstractmethod
    def __getitem__(
        self, idx: int
    ) -> RerankingDataItem | RetrievalDataItem | dict[str, Any]:
        raise NotImplementedError

    @property
    def rank_id(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def worker_id(self) -> int:
        worker_info: Any | None = get_worker_info()
        return int(worker_info.id) if worker_info is not None else 0

    @property
    @abc.abstractmethod
    def collator(self) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_data(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def setup(self) -> None:
        raise NotImplementedError
