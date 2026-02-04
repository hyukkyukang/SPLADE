from typing import Any, Dict

import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.collator import UniversalCollator
from src.data.dataclass import RetrievalDataItem
from src.data.pd_module import PDModule


class RetrievalPDModule(PDModule):
    """Retrieval PyTorch datasets module for evaluation/inference."""

    # --- Special methods ---
    @staticmethod
    def _normalize_optional_str(value: Any | None) -> str | None:
        if value is None:
            return None
        normalized: str = str(value).strip()
        return normalized if normalized else None

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        *,
        seed: int,
        load_teacher_scores: bool | None = None,
        require_teacher_scores: bool | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            tokenizer=tokenizer,
            seed=seed,
            load_teacher_scores=load_teacher_scores,
            require_teacher_scores=require_teacher_scores,
        )
        self.hf_name: str = (
            str(self.cfg.query_corpus_hf_name)
            if self.cfg.query_corpus_hf_name is not None
            else str(self.cfg.hf_name)
        )
        self.hf_split: str = str(self.cfg.split)
        self._beir_mode: bool = (
            str(getattr(self.cfg, "type", "")).lower() == "beir"
            or self.cfg.get("beir_dataset") is not None
        )
        text_cache_dir: str | None = self.cfg.query_corpus_hf_cache_dir
        self.hf_cache_dir: str | None = (
            None
            if text_cache_dir is None and self.cfg.hf_cache_dir is None
            else str(
                text_cache_dir if text_cache_dir is not None else self.cfg.hf_cache_dir
            )
        )
        self.qrels_hf_name: str | None = self._normalize_optional_str(
            getattr(self.cfg, "qrels_hf_name", None)
        )
        self.qrels_hf_subset: str | None = self._normalize_optional_str(
            getattr(self.cfg, "qrels_hf_subset", None)
        )
        qrels_split_override: str | None = self._normalize_optional_str(
            getattr(self.cfg, "qrels_hf_split", None)
        )
        self.qrels_hf_split: str = qrels_split_override or self.hf_split
        qrels_cache_override: str | None = self._normalize_optional_str(
            getattr(self.cfg, "qrels_hf_cache_dir", None)
        )
        self.qrels_hf_cache_dir: str | None = (
            qrels_cache_override if qrels_cache_override is not None else self.hf_cache_dir
        )
        self.qrels_hf_data_files: Any | None = getattr(
            self.cfg, "qrels_hf_data_files", None
        )
        self._query_ids: list[str] = []
        self._query_id_to_idx: dict[str, int] = {}
        self._qrels: Dict[str, Dict[str, float]] = {}
        self._qrels_dataset: Dataset | None = None

    def __len__(self) -> int:
        self._ensure_query_index()
        return len(self._query_ids)

    def __getitem__(self, idx: int) -> RetrievalDataItem:
        self._ensure_query_index()
        qid: str = self._query_ids[int(idx)]
        query_idx: int = self._query_id_to_idx[qid]
        query_text: str = self.dataset.query_text(query_idx)
        tokens: dict[str, torch.Tensor] = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        query_input_ids: torch.Tensor = tokens["input_ids"].squeeze(0)
        query_attention_mask: torch.Tensor = tokens["attention_mask"].squeeze(0)
        return RetrievalDataItem(
            data_idx=int(idx),
            qid=qid,
            relevance_judgments=self.get_relevance_judgments(qid),
            query_text=query_text,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
        )

    # --- Property methods ---
    @property
    def collator(self) -> UniversalCollator:
        return UniversalCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_padding=self.max_padding,
            max_query_length=self.max_query_length,
        )

    @property
    def qrels_dict(self) -> Dict[str, Dict[str, float]]:
        return self._qrels

    # --- Protected methods ---
    def _ensure_query_index(self) -> None:
        if self._query_ids:
            return
        self._query_id_to_idx = dict(self.dataset.query_dataset_id_to_idx)
        self._query_ids = list(self._query_id_to_idx.keys())

    def _load_hf_split(
        self, hf_name: str, config: str, split: str, cache_dir: str | None
    ) -> Dataset:
        if self._beir_mode:
            # BEIR datasets use a single config and expose corpus/queries/qrels as splits.
            return load_dataset(hf_name, split=config, cache_dir=cache_dir)
        return load_dataset(hf_name, config, split=split, cache_dir=cache_dir)

    def _load_qrels_dataset(self) -> Dataset:
        qrels_name: str = self.qrels_hf_name or self.hf_name
        qrels_subset: str | None = self.qrels_hf_subset
        qrels_split: str = self.qrels_hf_split
        qrels_cache_dir: str | None = self.qrels_hf_cache_dir
        qrels_data_files: Any | None = self.qrels_hf_data_files

        if (
            self.qrels_hf_name is None
            and qrels_subset is None
            and qrels_data_files is None
        ):
            return self._load_hf_split(qrels_name, "qrels", qrels_split, qrels_cache_dir)

        data_files: dict[str, Any] | None = (
            dict(qrels_data_files) if qrels_data_files is not None else None
        )
        if qrels_subset is not None:
            return load_dataset(
                qrels_name,
                qrels_subset,
                split=qrels_split,
                cache_dir=qrels_cache_dir,
                data_files=data_files,
            )
        return load_dataset(
            qrels_name,
            split=qrels_split,
            cache_dir=qrels_cache_dir,
            data_files=data_files,
        )

    # --- Public methods ---
    def prepare_data(self) -> None:
        _ = self.dataset.query_dataset
        _ = self._load_qrels_dataset()

    def setup(self) -> None:
        self._ensure_query_index()
        self._qrels_dataset = self._load_qrels_dataset()

        self._qrels = {}
        if self._qrels_dataset is not None:
            allowed_queries: set[str] = set(self._query_ids)
            for raw_row in self._qrels_dataset:
                row: dict[str, Any] = raw_row
                qid: str = str(
                    row.get("query-id")
                    or row.get("query_id")
                    or row.get("qid")
                    or row.get("_id")
                )
                if qid not in allowed_queries:
                    continue
                doc_id: str = str(
                    row.get("corpus-id")
                    or row.get("doc_id")
                    or row.get("pid")
                    or row.get("docid")
                )
                score: float = float(
                    row.get("score") or row.get("relevance") or row.get("rel") or 0
                )
                self._qrels.setdefault(qid, {})[doc_id] = score
            # Restrict evaluation to queries with qrels to avoid scoring unlabeled queries.
            qrels_query_ids: set[str] = set(self._qrels.keys())
            if qrels_query_ids:
                self._query_ids = [qid for qid in self._query_ids if qid in qrels_query_ids]

    def get_relevance_judgments(self, qid: str) -> Dict[str, float]:
        return self._qrels.get(qid, {})
