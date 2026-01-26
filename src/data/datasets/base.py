from __future__ import annotations

import abc
import os
from typing import Any, Dict, Optional

from datasets import Dataset
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
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.tokenizer = tokenizer
        self.name = str(self.cfg.name)
        self.data_instances: Optional[list[DataTuple]] = None

    @property
    def rank_id(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def worker_id(self) -> int:
        worker_info = get_worker_info()
        return worker_info.id if worker_info is not None else 0

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

    def __len__(self) -> int:
        return len(self.data_instances or [])

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> RerankingDataItem | RetrievalDataItem:
        raise NotImplementedError


class BaseRetrievalDataset(BaseDataset, abc.ABC):
    """Base dataset interface for retrieval evaluation datasets."""

    @property
    @abc.abstractmethod
    def query_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def corpus_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def qrels_dict(self) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def corpus_text_column(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def query_id_column(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def corpus_id_column(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def qrel_query_column(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def qrel_doc_column(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def qrel_score_column(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def all_qids(self) -> set[str]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def all_doc_ids(self) -> set[str]:
        raise NotImplementedError

    def get_relevance_judgments(self, qid: str) -> Dict[str, float]:
        return self.qrels_dict.get(qid, {})
