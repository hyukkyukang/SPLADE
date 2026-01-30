from __future__ import annotations

import logging
from typing import Any, Iterable

from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.dataset.base import BaseDataset
from src.utils.logging import log_if_rank_zero

logger: logging.Logger = logging.getLogger("CorpusDataset")


class CorpusDataset(BaseDataset):
    """Dataset wrapper for encoding a corpus from Hugging Face datasets."""

    # --- Special methods ---
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig | None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        resolved_global_cfg: DictConfig = global_cfg or cfg
        BaseDataset.__init__(
            self, cfg=cfg, global_cfg=resolved_global_cfg, tokenizer=tokenizer
        )

        self._hf_name: str = self._resolve_hf_name()
        self._hf_subset: str | None = getattr(self.cfg, "hf_subset", None)
        self._hf_split: str = str(getattr(self.cfg, "hf_split", "train"))
        self._hf_cache_dir: str | None = getattr(self.cfg, "hf_cache_dir", None)
        self._hf_max_samples: int | None = getattr(self.cfg, "hf_max_samples", None)
        self._hf_skip_samples: int = int(getattr(self.cfg, "hf_skip_samples", 0))

        self._corpus_split: str = self._resolve_corpus_split()
        self._dataset: Dataset | None = None
        self._doc_id_column: str | None = None
        self._title_column: str | None = None
        self._text_column: str | None = None

    # --- Public methods ---
    def __len__(self) -> int:
        if self._dataset is None:
            return 0
        total: int = len(self._dataset)
        start: int = min(self._hf_skip_samples, total)
        if self._hf_max_samples is None:
            return total - start
        return max(0, min(total - start, int(self._hf_max_samples)))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._dataset is None:
            raise RuntimeError("Dataset is not initialized. Call setup() first.")
        total: int = len(self._dataset)
        start: int = min(self._hf_skip_samples, total)
        raw_record: dict[str, Any] = dict(self._dataset[start + idx])
        doc_id: str = self._extract_doc_id(raw_record)
        text: str = self._build_doc_text(raw_record)
        return {"doc_id": doc_id, "text": text}

    @property
    def collator(self) -> Any:
        return self._collate

    def prepare_data(self) -> None:
        _ = self._load_corpus_dataset()

    def setup(self) -> None:
        self._dataset = self._load_corpus_dataset()
        self._resolve_columns()
        log_if_rank_zero(
            logger,
            f"Loaded {len(self)} corpus documents from {self._hf_name}.",
        )

    # --- Protected methods ---
    def _resolve_hf_name(self) -> str:
        hf_name: str | None = getattr(self.cfg, "hf_name", None)
        if hf_name:
            return str(hf_name)
        beir_dataset: str | None = getattr(self.cfg, "beir_dataset", None)
        if beir_dataset:
            return f"BeIR/{beir_dataset}"
        raise ValueError("dataset.hf_name or dataset.beir_dataset must be set.")

    def _resolve_corpus_split(self) -> str:
        configured_split: str | None = getattr(self.cfg, "corpus_split", None)
        if configured_split is not None:
            return str(configured_split)
        if getattr(self.cfg, "beir_dataset", None):
            return "train"
        return self._hf_split

    def _load_corpus_dataset(self) -> Dataset:
        try:
            return load_dataset(
                self._hf_name,
                "corpus",
                split=self._corpus_split,
                cache_dir=self._hf_cache_dir,
            )
        except Exception:  # pylint: disable=broad-except
            if self._hf_subset:
                return load_dataset(
                    self._hf_name,
                    self._hf_subset,
                    split=self._corpus_split,
                    cache_dir=self._hf_cache_dir,
                )
            return load_dataset(
                self._hf_name,
                split=self._corpus_split,
                cache_dir=self._hf_cache_dir,
            )

    def _resolve_columns(self) -> None:
        if self._dataset is None:
            raise RuntimeError("Dataset is not initialized. Call setup() first.")
        column_names: Iterable[str] = self._dataset.column_names
        self._doc_id_column = self._resolve_column(
            column_names, ("_id", "doc_id", "corpus_id", "pid", "id")
        )
        self._title_column = self._resolve_optional_column(
            column_names, ("title", "headline")
        )
        self._text_column = self._resolve_column(
            column_names, ("text", "contents", "body", "document")
        )

    def _resolve_column(
        self, column_names: Iterable[str], candidates: tuple[str, ...]
    ) -> str:
        for name in candidates:
            if name in column_names:
                return name
        raise ValueError(f"Unable to resolve column from {list(column_names)}")

    def _resolve_optional_column(
        self, column_names: Iterable[str], candidates: tuple[str, ...]
    ) -> str | None:
        for name in candidates:
            if name in column_names:
                return name
        return None

    def _extract_doc_id(self, record: dict[str, Any]) -> str:
        if self._doc_id_column is None:
            raise RuntimeError("doc_id column not resolved.")
        return str(record.get(self._doc_id_column))

    def _build_doc_text(self, record: dict[str, Any]) -> str:
        if self._text_column is None:
            raise RuntimeError("text column not resolved.")
        title: str = ""
        if self._title_column is not None:
            title = str(record.get(self._title_column, "") or "").strip()
        text: str = str(record.get(self._text_column, "") or "").strip()
        if title and text:
            return f"{title} {text}"
        return title or text

    def _collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        doc_ids: list[str] = [str(item["doc_id"]) for item in batch]
        texts: list[str] = [str(item["text"]) for item in batch]
        return {"doc_ids": doc_ids, "doc_texts": texts}
