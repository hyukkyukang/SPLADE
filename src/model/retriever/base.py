import random
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.data.dataclass import CorpusDataset, Document, Query, QueryDataset


class BaseRetriever:
    """Base class providing shared sparse retrieval helpers."""

    # --- Special methods ---
    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        max_query_length: int,
        max_doc_length: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        # Store model components for reuse across encoders.
        self.model: Any = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_query_length: int = max_query_length
        self.max_doc_length: int = max_doc_length
        self.batch_size: int = batch_size
        self.device: torch.device = device

    # --- Protected methods ---
    def _encode_queries(self, queries: list[Query]) -> tuple[list[str], torch.Tensor]:
        dataset: QueryDataset = QueryDataset(queries)
        loader: DataLoader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
        ids: list[str] = []
        reps: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch_queries: list[Query] = list(batch)
                texts: list[str] = [query.text for query in batch_queries]
                ids.extend([query.query_id for query in batch_queries])
                # Tokenize and move tensors to the target device.
                tokens: dict[str, torch.Tensor] = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_query_length,
                    return_tensors="pt",
                )
                query_input_ids: torch.Tensor = tokens["input_ids"].to(self.device)
                query_attention_mask: torch.Tensor = tokens["attention_mask"].to(
                    self.device
                )
                q_reps: torch.Tensor = self.model.encode_queries(
                    query_input_ids, query_attention_mask
                )
                reps.append(q_reps.cpu())
        return ids, torch.cat(reps, dim=0)

    def _encode_corpus(self, docs: list[Document]) -> tuple[list[str], torch.Tensor]:
        dataset: CorpusDataset = CorpusDataset(docs)
        loader: DataLoader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
        ids: list[str] = []
        reps: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch_docs: list[Document] = list(batch)
                texts: list[str] = [doc.text for doc in batch_docs]
                ids.extend([doc.doc_id for doc in batch_docs])
                # Tokenize and move tensors to the target device.
                tokens: dict[str, torch.Tensor] = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_doc_length,
                    return_tensors="pt",
                )
                doc_input_ids: torch.Tensor = tokens["input_ids"].to(self.device)
                doc_attention_mask: torch.Tensor = tokens["attention_mask"].to(
                    self.device
                )
                d_reps: torch.Tensor = self.model.encode_docs(
                    doc_input_ids, doc_attention_mask
                )
                reps.append(d_reps.cpu())
        return ids, torch.cat(reps, dim=0)

    # --- Public methods ---
    def encode_queries(self, queries: list[Query]) -> tuple[list[str], torch.Tensor]:
        """Public wrapper for query encoding."""
        return self._encode_queries(queries)

    def encode_corpus(self, docs: list[Document]) -> tuple[list[str], torch.Tensor]:
        """Public wrapper for corpus encoding."""
        return self._encode_corpus(docs)

    def retrieve(
        self,
        queries: list[Query],
        docs: list[Document],
        top_k: int,
        max_docs: int | None = None,
    ) -> dict[str, list[str]]:
        if max_docs is not None and len(docs) > max_docs:
            docs = random.sample(docs, max_docs)

        doc_ids: list[str]
        doc_reps: torch.Tensor
        doc_ids, doc_reps = self._encode_corpus(docs)
        query_ids: list[str]
        query_reps: torch.Tensor
        query_ids, query_reps = self._encode_queries(queries)

        doc_reps_t: torch.Tensor = doc_reps.T
        results: dict[str, list[str]] = {}
        for qid, q_rep in zip(query_ids, query_reps):
            scores: torch.Tensor = torch.mv(doc_reps_t, q_rep)
            topk: torch.Tensor = torch.topk(
                scores, k=min(top_k, scores.size(0))
            ).indices
            results[qid] = [doc_ids[i] for i in topk.tolist()]
        return results
