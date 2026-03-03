from typing import Any, Iterable

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.pd_module.scoring import (
    ScoringItem,
    ScoringPDModule,
    _resolve_local_files_only,
)
from src.data.sampler import NonPaddingDistributedSampler
from src.utils.script_setup import normalize_optional_str


class ScoringCollator:
    """Collate scoring rows and batch tokenize pairs."""

    def __init__(
        self,
        *,
        model_name: str,
        tokenizer_name: str | None,
        max_length: int,
        tokenize_chunk_size: int,
        local_files_only: bool,
        trust_remote_code: bool = False,
        use_fast_tokenizer: bool = True,
        require_fast_tokenizer: bool = False,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self._model_name: str = model_name
        self._tokenizer_name: str | None = tokenizer_name
        self._max_length: int = max_length
        self._tokenize_chunk_size: int = tokenize_chunk_size
        self._local_files_only: bool = local_files_only
        self._trust_remote_code: bool = trust_remote_code
        self._use_fast_tokenizer: bool = use_fast_tokenizer
        self._require_fast_tokenizer: bool = require_fast_tokenizer
        self._tokenizer: PreTrainedTokenizerBase | None = tokenizer

    def _resolve_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            name = self._tokenizer_name or self._model_name
            self._tokenizer = AutoTokenizer.from_pretrained(
                name,
                local_files_only=self._local_files_only,
                use_fast=bool(self._use_fast_tokenizer),
                trust_remote_code=self._trust_remote_code,
            )
            if bool(self._require_fast_tokenizer) and not bool(self._tokenizer.is_fast):
                raise ValueError(
                    "Fast tokenizer is required for scoring but a slow tokenizer "
                    f"was loaded: {name}"
                )
        return self._tokenizer

    def __call__(self, batch: Iterable[ScoringItem | None]) -> dict[str, Any] | None:
        items: list[ScoringItem] = [item for item in batch if item is not None]
        if not items:
            return None
        output: dict[str, Any] = {
            "rows": [item.row for item in items],
            "qids": [item.qid for item in items],
            "doc_ids": [item.doc_ids for item in items],
            "labels": [item.labels for item in items],
            "doc_sources": [item.doc_sources for item in items],
        }
        pair_row_ids: list[int] = []
        pair_doc_idxs: list[int] = []
        pair_queries: list[str] = []
        pair_docs: list[str] = []
        for row_idx, item in enumerate(items):
            doc_count: int = len(item.doc_texts)
            if doc_count <= 0:
                continue
            pair_row_ids.extend([row_idx] * doc_count)
            pair_doc_idxs.extend(list(range(doc_count)))
            pair_queries.extend([item.query_text] * doc_count)
            pair_docs.extend(item.doc_texts)
        if not pair_row_ids:
            output["pair_row_ids"] = []
            output["pair_doc_idxs"] = []
            output["pair_tokens"] = {}
            return output
        tokenizer = self._resolve_tokenizer()
        tokenize_chunk_size: int = int(self._tokenize_chunk_size)
        if tokenize_chunk_size <= 0:
            tokenize_chunk_size = len(pair_queries)
        pair_tokens_by_key: dict[str, list[torch.Tensor]] = {}
        for start in range(0, len(pair_queries), tokenize_chunk_size):
            end = start + tokenize_chunk_size
            tokens = tokenizer(
                pair_queries[start:end],
                pair_docs[start:end],
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            for key, value in tokens.items():
                pair_tokens_by_key.setdefault(key, []).append(value)
        pair_tokens: dict[str, torch.Tensor] = {
            key: torch.cat(values, dim=0) for key, values in pair_tokens_by_key.items()
        }
        output["pair_row_ids"] = pair_row_ids
        output["pair_doc_idxs"] = pair_doc_idxs
        output["pair_tokens"] = pair_tokens
        return output


class ScoringDataModule(L.LightningDataModule):
    """LightningDataModule for cross-encoder scoring."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self._dataset: ScoringPDModule | None = None

    @property
    def dataset(self) -> ScoringPDModule:
        if self._dataset is None:
            positives_cfg: DictConfig | None = (
                self.cfg.positives if "positives" in self.cfg else None
            )
            self._dataset = ScoringPDModule(
                score_dataset_cfg=self.cfg.score_dataset,
                scoring_cfg=self.cfg.scoring,
                mining_cfg=self.cfg.mining,
                positives_cfg=positives_cfg,
            )
        return self._dataset

    def prepare_data(self) -> None:
        self.dataset.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.dataset.setup()

    def predict_dataloader(self) -> DataLoader:
        scoring_cfg: DictConfig = self.cfg.scoring
        num_workers: int = int(scoring_cfg.num_workers)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_primary = int(torch.distributed.get_rank()) == 0
        else:
            is_primary = True
        local_files_only = _resolve_local_files_only(scoring_cfg, is_primary=is_primary)
        collator = ScoringCollator(
            model_name=str(scoring_cfg.model_name),
            tokenizer_name=normalize_optional_str(scoring_cfg.tokenizer_name),
            max_length=int(scoring_cfg.max_length),
            tokenize_chunk_size=int(scoring_cfg.tokenize_chunk_size),
            local_files_only=local_files_only,
            trust_remote_code=bool(scoring_cfg.trust_remote_code),
            use_fast_tokenizer=bool(scoring_cfg.use_fast_tokenizer),
            require_fast_tokenizer=bool(scoring_cfg.require_fast_tokenizer),
        )
        sampler: Sampler[int] | None = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = NonPaddingDistributedSampler(self.dataset, shuffle=False)
        prefetch_factor: int | None = None
        if num_workers > 0:
            prefetch_factor = int(scoring_cfg.prefetch_factor)
        batch_size: int = int(scoring_cfg.row_batch_size)
        if batch_size <= 0:
            raise ValueError("scoring.row_batch_size must be a positive integer.")
        use_cpu: bool = bool(scoring_cfg.use_cpu)
        pin_memory: bool = not use_cpu
        persistent_workers: bool = num_workers > 0
        dataloader_kwargs: dict[str, Any] = {
            "dataset": self.dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collator,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
        if sampler is not None:
            dataloader_kwargs["sampler"] = sampler
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(**dataloader_kwargs)
