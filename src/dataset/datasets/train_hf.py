from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from src.dataset.datasets.train import TrainSample


class HFMSMarcoTrainDataset:
    def __init__(
        self,
        hf_name: str,
        hf_subset: str,
        hf_split: str,
        num_positives: int,
        num_negatives: int,
        require_teacher_scores: bool,
        teacher_score_key: str,
        hf_text_name: str | None = None,
        hf_teacher_name: str | None = None,
        hf_teacher_subset: str | None = None,
        hf_teacher_split: str = "train",
        hf_cache_dir: str | None = None,
        hf_max_samples: int | None = None,
        hf_teacher_cache_dir: str | None = None,
        hf_teacher_max_samples: int | None = None,
    ) -> None:
        self.hf_name = hf_name
        self.hf_subset = hf_subset
        self.hf_split = hf_split
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.require_teacher_scores = require_teacher_scores
        self.teacher_score_key = teacher_score_key
        self.hf_teacher_subset = hf_teacher_subset
        self.hf_teacher_name = hf_teacher_name
        self.hf_teacher_split = hf_teacher_split

        self.dataset = load_dataset(
            hf_name, hf_subset, split=hf_split, cache_dir=hf_cache_dir
        )
        if hf_max_samples is not None:
            self.dataset = self.dataset.select(range(min(hf_max_samples, len(self.dataset))))

        text_name = hf_text_name or hf_name
        self.query_map, self.corpus_map = self._load_queries_and_corpus(
            text_name, hf_cache_dir
        )
        self.teacher_scores = self._load_teacher_scores(
            hf_teacher_name,
            hf_teacher_subset,
            hf_teacher_split,
            hf_teacher_cache_dir or hf_cache_dir,
            hf_teacher_max_samples,
        )

    @staticmethod
    def _load_queries_and_corpus(
        hf_name: str, hf_cache_dir: str | None
    ) -> tuple[dict[str, str], dict[str, str]]:
        queries_ds = load_dataset(hf_name, "queries", split="train", cache_dir=hf_cache_dir)
        corpus_ds = load_dataset(hf_name, "corpus", split="train", cache_dir=hf_cache_dir)

        query_map: dict[str, str] = {}
        for row in queries_ds:
            qid = str(row.get("query_id") or row.get("_id") or row.get("id"))
            text = row.get("text") or row.get("query") or ""
            query_map[qid] = text

        corpus_map: dict[str, str] = {}
        for row in corpus_ds:
            doc_id = str(
                row.get("doc_id")
                or row.get("passage_id")
                or row.get("_id")
                or row.get("id")
            )
            text = row.get("text") or row.get("passage") or row.get("contents") or ""
            corpus_map[doc_id] = text

        return query_map, corpus_map

    @staticmethod
    def _load_teacher_scores(
        hf_name: str | None,
        subset: str | None,
        split: str,
        hf_cache_dir: str | None,
        max_samples: int | None,
    ) -> dict[tuple[str, str], float]:
        if hf_name is None or subset is None:
            return {}
        teacher_ds = load_dataset(hf_name, subset, split=split, cache_dir=hf_cache_dir)
        if max_samples is not None:
            teacher_ds = teacher_ds.select(range(min(max_samples, len(teacher_ds))))
        scores: dict[tuple[str, str], float] = {}
        for row in teacher_ds:
            qid = str(row.get("query_id") or row.get("qid") or row.get("_id"))
            if subset == "pair":
                doc_id = str(
                    row.get("corpus_id")
                    or row.get("doc_id")
                    or row.get("passage_id")
                    or row.get("pid")
                    or row.get("_id")
                )
                score = row.get("score")
                if score is None:
                    continue
                scores[(qid, doc_id)] = float(score)
            elif subset == "triplet":
                pos_id = str(
                    row.get("positive_id") or row.get("pos_id") or row.get("doc_pos_id")
                )
                neg_id = str(
                    row.get("negative_id") or row.get("neg_id") or row.get("doc_neg_id")
                )
                score = row.get("score") or row.get("scores")
                if isinstance(score, (list, tuple)) and len(score) == 2:
                    scores[(qid, pos_id)] = float(score[0])
                    scores[(qid, neg_id)] = float(score[1])
            elif subset == "list":
                doc_ids = row.get("corpus_id") or row.get("doc_ids") or []
                score_list = row.get("score") or row.get("scores") or []
                for doc_id, score in zip(doc_ids, score_list):
                    scores[(qid, str(doc_id))] = float(score)
        return scores

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TrainSample:
        row = self.dataset[index]

        if "query" in row and "positive" in row and "negative" in row:
            query_text = row["query"]
            pos_texts = [row["positive"]]
            neg_texts = [row["negative"]]
            pos_ids = [None]
            neg_ids = [None]
            qid = str(row.get("query_id") or row.get("qid") or index)
        elif "query_id" in row and "positive_id" in row:
            qid = str(row["query_id"])
            query_text = self.query_map.get(qid, "")
            pos_ids = [str(row["positive_id"])]
            neg_ids = [str(row["negative_id"])]
            pos_texts = [self.corpus_map.get(pos_ids[0], "")]
            neg_texts = [self.corpus_map.get(neg_ids[0], "")]
            score_values = row.get("score") or row.get("scores")
            if isinstance(score_values, (list, tuple)) and len(score_values) == 2:
                row_teacher_scores = [float(score_values[0]), float(score_values[1])]
            else:
                row_teacher_scores = None
        elif "query_id" in row and "doc_ids" in row and "labels" in row:
            qid = str(row["query_id"])
            query_text = self.query_map.get(qid, "")
            doc_ids = [str(doc_id) for doc_id in row["doc_ids"]]
            labels = row["labels"]
            pos_ids = [doc_id for doc_id, label in zip(doc_ids, labels) if label > 0]
            neg_ids = [doc_id for doc_id, label in zip(doc_ids, labels) if label <= 0]
            pos_ids = self._sample_ids(pos_ids, self.num_positives)
            neg_ids = self._sample_ids(neg_ids, self.num_negatives)
            pos_texts = [self.corpus_map.get(doc_id, "") for doc_id in pos_ids]
            neg_texts = [self.corpus_map.get(doc_id, "") for doc_id in neg_ids]
            row_teacher_scores = None
        else:
            raise ValueError(f"Unsupported MSMARCO HF row format: {row.keys()}")

        docs = pos_texts + neg_texts
        doc_ids = pos_ids + neg_ids
        if row_teacher_scores is not None and len(row_teacher_scores) == len(doc_ids):
            teacher_scores = row_teacher_scores
        else:
            teacher_scores = [
                self._get_teacher_score(qid, doc_id) for doc_id in doc_ids
            ]

        if self.require_teacher_scores and any(
            score != score for score in teacher_scores
        ):
            raise ValueError(f"Missing teacher score in HF sample for query {qid}")

        return TrainSample(
            query=query_text,
            docs=docs,
            pos_count=len(pos_texts),
            teacher_scores=teacher_scores,
        )

    def _sample_ids(self, items: list[str], count: int) -> list[str]:
        if count <= 0:
            return []
        if len(items) <= count:
            return items
        return random.sample(items, count)

    def _get_teacher_score(self, qid: str, doc_id: str | None) -> float:
        if doc_id is None:
            return float("nan")
        score = self.teacher_scores.get((qid, str(doc_id)))
        if score is None:
            return float("nan")
        return float(score)
