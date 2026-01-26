from __future__ import annotations

import random

import torch

from src.dataset.datasets.retrieval import RetrievalDataset
from src.metric.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.model.retriever.sparse_retriever import SparseRetriever


class BEIREvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        max_query_length: int,
        max_doc_length: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.retriever = SparseRetriever(
            model=model,
            tokenizer=tokenizer,
            max_query_length=max_query_length,
            max_doc_length=max_doc_length,
            batch_size=batch_size,
            device=device,
        )

    def evaluate_hf(
        self,
        hf_name: str,
        split: str,
        metrics: list[str],
        top_k: int = 100,
        sample_size: int | None = None,
        max_docs: int | None = None,
        cache_dir: str | None = None,
    ) -> dict[str, float]:
        dataset = RetrievalDataset.from_hf(
            hf_name=hf_name, split=split, cache_dir=cache_dir
        )
        return self._evaluate_dataset(
            dataset=dataset,
            metrics=metrics,
            top_k=top_k,
            sample_size=sample_size,
            max_docs=max_docs,
        )

    def _evaluate_dataset(
        self,
        dataset: RetrievalDataset,
        metrics: list[str],
        top_k: int,
        sample_size: int | None,
        max_docs: int | None,
    ) -> dict[str, float]:

        queries = dataset.queries
        if sample_size is not None and sample_size < len(queries):
            queries = random.sample(queries, sample_size)

        results = self.retriever.retrieve(
            queries=queries,
            docs=dataset.corpus,
            top_k=top_k,
            max_docs=max_docs,
        )

        metrics_out: dict[str, float] = {}
        for metric in metrics:
            if metric.lower().startswith("mrr"):
                k = int(metric.split("@")[1])
                metrics_out[metric] = mrr_at_k(dataset.qrels, results, k)
            elif metric.lower().startswith("ndcg"):
                k = int(metric.split("@")[1])
                metrics_out[metric] = ndcg_at_k(dataset.qrels, results, k)
            elif metric.lower().startswith("recall"):
                k = int(metric.split("@")[1])
                metrics_out[metric] = recall_at_k(dataset.qrels, results, k)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        return metrics_out
