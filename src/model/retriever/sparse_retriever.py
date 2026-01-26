from __future__ import annotations

import random
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from src.data.dataclass import CorpusDataset, Document, Query, QueryDataset


class SparseRetriever:
    def __init__(
        self,
        model,
        tokenizer,
        max_query_length: int,
        max_doc_length: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.batch_size = batch_size
        self.device = device

    def _encode_queries(self, queries: list[Query]) -> tuple[list[str], torch.Tensor]:
        dataset = QueryDataset(queries)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        ids: list[str] = []
        reps: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                texts = [q.text for q in batch]
                ids.extend([q.query_id for q in batch])
                tokens = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_query_length,
                    return_tensors="pt",
                )
                q_reps = self.model.encode_queries(
                    tokens["input_ids"].to(self.device),
                    tokens["attention_mask"].to(self.device),
                )
                reps.append(q_reps.cpu())
        return ids, torch.cat(reps, dim=0)

    def _encode_corpus(self, docs: list[Document]) -> tuple[list[str], torch.Tensor]:
        dataset = CorpusDataset(docs)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        ids: list[str] = []
        reps: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                texts = [d.text for d in batch]
                ids.extend([d.doc_id for d in batch])
                tokens = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_doc_length,
                    return_tensors="pt",
                )
                d_reps = self.model.encode_docs(
                    tokens["input_ids"].to(self.device),
                    tokens["attention_mask"].to(self.device),
                )
                reps.append(d_reps.cpu())
        return ids, torch.cat(reps, dim=0)

    def encode_queries(self, queries: list[Query]) -> tuple[list[str], torch.Tensor]:
        """Public wrapper for query encoding."""
        return self._encode_queries(queries)

    def encode_corpus(self, docs: list[Document]) -> tuple[list[str], torch.Tensor]:
        """Public wrapper for corpus encoding."""
        return self._encode_corpus(docs)

    def encode_queries_from_loader(
        self, loader: DataLoader
    ) -> tuple[list[str], torch.Tensor]:
        ids: list[str] = []
        reps: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                ids.extend(batch["qids"])
                q_reps = self.model.encode_queries(
                    batch["query_input_ids"].to(self.device),
                    batch["query_attention_mask"].to(self.device),
                )
                reps.append(q_reps.cpu())
        return ids, torch.cat(reps, dim=0)

    def retrieve(
        self,
        queries: list[Query],
        docs: list[Document],
        top_k: int,
        max_docs: int | None = None,
    ) -> dict[str, list[str]]:
        if max_docs is not None and len(docs) > max_docs:
            docs = random.sample(docs, max_docs)

        doc_ids, doc_reps = self._encode_corpus(docs)
        query_ids, query_reps = self._encode_queries(queries)

        doc_reps_t = doc_reps.T
        results: dict[str, list[str]] = {}
        for qid, q_rep in zip(query_ids, query_reps):
            scores = torch.mv(doc_reps_t, q_rep)
            topk = torch.topk(scores, k=min(top_k, scores.size(0))).indices
            results[qid] = [doc_ids[i] for i in topk.tolist()]
        return results
