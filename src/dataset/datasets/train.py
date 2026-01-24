from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.file_io import read_jsonl


@dataclass
class TrainSample:
    query: str
    docs: list[str]
    pos_count: int
    teacher_scores: list[float]


class TrainDataset:
    def __init__(
        self,
        train_path: str,
        num_positives: int,
        num_negatives: int,
        require_teacher_scores: bool,
        teacher_score_key: str,
        teacher_scores_path: str | None = None,
    ) -> None:
        if not Path(train_path).exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        self.data = read_jsonl(train_path)
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.require_teacher_scores = require_teacher_scores
        self.teacher_score_key = teacher_score_key
        self.teacher_scores = (
            self._load_teacher_scores(teacher_scores_path) if teacher_scores_path else {}
        )

    @staticmethod
    def _load_teacher_scores(path: str) -> dict[tuple[str, str], float]:
        if path is None:
            return {}
        if not Path(path).exists():
            raise FileNotFoundError(f"Teacher score file not found: {path}")
        scores: dict[tuple[str, str], float] = {}
        for row in read_jsonl(path):
            qid = str(row.get("query_id") or row.get("qid") or row.get("queryId"))
            doc_id = str(row.get("doc_id") or row.get("pid") or row.get("docId"))
            score = float(row.get("score"))
            scores[(qid, doc_id)] = score
        return scores

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> TrainSample:
        item = self.data[index]
        query = item.get("query")
        if query is None:
            raise ValueError("Training example missing 'query' field")

        qid = str(item.get("query_id") or item.get("qid") or index)
        positives = self._extract_docs(item.get("positives", []), qid)
        negatives = self._extract_docs(item.get("negatives", []), qid)

        positives = self._sample(positives, self.num_positives)
        negatives = self._sample(negatives, self.num_negatives)

        docs = positives + negatives
        pos_count = len(positives)
        teacher_scores = [doc["teacher_score"] for doc in docs]
        doc_texts = [doc["text"] for doc in docs]

        return TrainSample(
            query=query,
            docs=doc_texts,
            pos_count=pos_count,
            teacher_scores=teacher_scores,
        )

    def _sample(self, items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        if len(items) <= count:
            return items
        return random.sample(items, count)

    def _extract_docs(self, entries: list[Any], qid: str) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, str):
                text = entry
                doc_id = None
                teacher_score = None
            elif isinstance(entry, dict):
                text = entry.get("text") or entry.get("doc") or entry.get("body")
                doc_id = entry.get("doc_id") or entry.get("pid") or entry.get("docId")
                teacher_score = entry.get(self.teacher_score_key)
            else:
                raise ValueError(f"Unsupported doc entry type: {type(entry)}")

            if text is None:
                raise ValueError("Doc entry missing text field")

            if teacher_score is None and doc_id is not None:
                teacher_score = self.teacher_scores.get((qid, str(doc_id)))

            if teacher_score is None and self.require_teacher_scores:
                raise ValueError(
                    f"Missing teacher score for query {qid} doc {doc_id}"
                )

            if teacher_score is None:
                teacher_score = float("nan")

            docs.append(
                {
                    "text": text,
                    "doc_id": doc_id,
                    "teacher_score": float(teacher_score),
                }
            )
        return docs
