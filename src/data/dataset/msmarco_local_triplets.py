from __future__ import annotations

import logging
import os
from typing import Any, Iterable

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.collators import RerankingCollator
from src.data.dataclass import RerankingDataItem
from src.data.dataset.base import BaseDataset

logger: logging.Logger = logging.getLogger("MSMARCOLocalTriplets")


class MSMARCOLocalTriplets(BaseDataset):
    """Load MS MARCO-style local triplets from raw.tsv."""

    # --- Special methods ---
    def __init__(
        self,
        cfg: DictConfig,
        global_cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__(cfg=cfg, global_cfg=global_cfg, tokenizer=tokenizer)
        self.data_dir: str = str(cfg.local_triplets_dir)
        self.raw_path: str = os.path.join(self.data_dir, "raw.tsv")
        self.max_query_length: int = int(cfg.max_query_length)
        self.max_doc_length: int = int(cfg.max_doc_length)
        self.max_padding: bool = bool(cfg.max_padding)
        self.num_positives: int = int(cfg.num_positives)
        self.num_negatives: int = int(cfg.num_negatives)
        if self.num_positives != 1 or self.num_negatives != 1:
            raise ValueError(
                "Local triplets dataset supports exactly one positive and one negative."
            )
        self._collator: RerankingCollator | None = None
        self._triplets: list[tuple[str, str, str, str]] = []

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, idx: int) -> RerankingDataItem:
        qid: str
        query_text: str
        pos_text: str
        neg_text: str
        qid, query_text, pos_text, neg_text = self._triplets[idx]
        query_input_ids: torch.Tensor
        query_attention_mask: torch.Tensor
        query_input_ids, query_attention_mask = self._tokenize_text(
            query_text, max_length=self.max_query_length
        )
        doc_input_ids: torch.Tensor
        doc_attention_mask: torch.Tensor
        doc_input_ids, doc_attention_mask = self._tokenize_docs(
            [pos_text, neg_text], max_length=self.max_doc_length
        )
        doc_mask: torch.Tensor = torch.tensor([True, True], dtype=torch.bool)
        pos_mask: torch.Tensor = torch.tensor([True, False], dtype=torch.bool)
        teacher_scores: torch.Tensor = torch.full(
            (2,), float("nan"), dtype=torch.float
        )

        return RerankingDataItem(
            data_idx=idx,
            qid=qid,
            pos_ids=[""],
            neg_ids=[""],
            query_text=query_text,
            doc_texts=[pos_text, neg_text],
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            doc_mask=doc_mask,
            pos_mask=pos_mask,
            teacher_scores=teacher_scores,
        )

    # --- Property methods ---
    @property
    def collator(self) -> RerankingCollator:
        if self._collator is None:
            self._collator = RerankingCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                require_teacher_scores=False,
                max_padding=self.max_padding,
                max_query_length=self.max_query_length,
                max_doc_length=self.max_doc_length,
                max_docs=2,
            )
        return self._collator

    # --- Protected methods ---
    def _parse_triplet_line(
        self, line: str, row_idx: int
    ) -> tuple[str, str, str, str] | None:
        stripped: str = line.strip()
        if not stripped:
            return None
        parts: list[str] = stripped.split("\t")
        if len(parts) == 3:
            query_text: str
            pos_text: str
            neg_text: str
            query_text, pos_text, neg_text = parts
            qid: str = str(row_idx)
        elif len(parts) == 4:
            qid = parts[0].strip()
            query_text = parts[1]
            pos_text = parts[2]
            neg_text = parts[3]
        else:
            return None
        return qid.strip(), query_text.strip(), pos_text.strip(), neg_text.strip()

    def _tokenize_text(
        self, text: str, *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding: str | bool = "max_length" if self.max_padding else True
        tokens: Any = self.tokenizer(
            text,
            padding=padding,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        input_ids: torch.Tensor = tokens["input_ids"].squeeze(0)
        attention_mask: torch.Tensor = tokens["attention_mask"].squeeze(0)
        return input_ids, attention_mask

    def _tokenize_docs(
        self, docs: Iterable[str], *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding: str | bool = "max_length" if self.max_padding else True
        tokens: Any = self.tokenizer(
            list(docs),
            padding=padding,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        input_ids: torch.Tensor = tokens["input_ids"]
        attention_mask: torch.Tensor = tokens["attention_mask"]
        return input_ids, attention_mask

    # --- Public methods ---
    def prepare_data(self) -> None:
        if not os.path.isfile(self.raw_path):
            raise FileNotFoundError(
                f"Missing local triplets file: {self.raw_path}"
            )

    def setup(self) -> None:
        skipped_lines: int = 0
        triplets: list[tuple[str, str, str, str]] = []
        with open(self.raw_path, "r", encoding="utf-8") as reader:
            for row_idx, line in enumerate(reader):
                parsed: tuple[str, str, str, str] | None = self._parse_triplet_line(
                    line, row_idx
                )
                if parsed is None:
                    skipped_lines += 1
                    continue
                triplets.append(parsed)
        if skipped_lines:
            logger.warning(
                "Skipped %d malformed triplet lines in %s",
                skipped_lines,
                self.raw_path,
            )
        self._triplets = triplets
