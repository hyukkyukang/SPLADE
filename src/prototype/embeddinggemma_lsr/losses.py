from typing import Iterable

import torch
from torch.nn import functional as F


def info_nce_in_batch(
    query_reps: torch.Tensor,
    doc_reps: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute in-batch InfoNCE where each row i matches doc i."""
    if query_reps.ndim != 2 or doc_reps.ndim != 2:
        raise ValueError("query_reps and doc_reps must be [batch, dim].")
    if query_reps.shape != doc_reps.shape:
        raise ValueError(
            "query_reps and doc_reps must have identical shape for in-batch loss."
        )
    temperature_value: float = max(float(temperature), 1e-8)
    scores: torch.Tensor = torch.matmul(query_reps.float(), doc_reps.float().transpose(0, 1))
    logits: torch.Tensor = scores / temperature_value
    labels: torch.Tensor = torch.arange(logits.shape[0], device=logits.device)
    loss: torch.Tensor = F.cross_entropy(logits, labels)
    return loss, logits


def flops_regularization(reps: torch.Tensor) -> torch.Tensor:
    """SPLADE FLOPs-style penalty: ||E[x]||_2^2 over vocabulary dimensions."""
    if reps.ndim != 2:
        raise ValueError("reps must be [batch, vocab_dim].")
    mean_activation: torch.Tensor = reps.float().mean(dim=0)
    return torch.sum(mean_activation.pow(2))


def compute_ranking_metrics(
    scores: torch.Tensor,
    *,
    positive_indices: torch.Tensor,
    k_values: Iterable[int] = (10,),
) -> dict[str, float]:
    """Compute MRR/Recall/nDCG at requested cutoffs.

    Args:
        scores: [num_queries, num_docs]
        positive_indices: [num_queries] index of the positive doc per query
    """
    if scores.ndim != 2:
        raise ValueError("scores must be a rank-2 tensor [num_queries, num_docs].")
    if positive_indices.ndim != 1:
        raise ValueError("positive_indices must be rank-1.")
    if scores.shape[0] != positive_indices.shape[0]:
        raise ValueError("scores and positive_indices must have same query dimension.")

    num_queries: int = int(scores.shape[0])
    if num_queries == 0:
        return {"mrr@10": 0.0, "recall@10": 0.0, "ndcg@10": 0.0}

    sorted_indices: torch.Tensor = torch.argsort(scores, dim=1, descending=True)
    positives: torch.Tensor = positive_indices.to(device=scores.device, dtype=torch.long)
    hits: torch.Tensor = sorted_indices.eq(positives.unsqueeze(1))
    positive_ranks: torch.Tensor = hits.float().argmax(dim=1) + 1

    reciprocal_ranks: torch.Tensor = 1.0 / positive_ranks.to(dtype=torch.float32)
    metrics: dict[str, float] = {}

    for k in k_values:
        cutoff: int = max(int(k), 1)
        hit_at_k: torch.Tensor = (positive_ranks <= cutoff).to(dtype=torch.float32)
        recall_k: float = float(hit_at_k.mean().item())
        mrr_k: float = float((reciprocal_ranks * hit_at_k).mean().item())

        # With one relevant doc per query, DCG@k is 1/log2(rank+1) if hit, else 0.
        dcg: torch.Tensor = hit_at_k / torch.log2(positive_ranks.to(torch.float32) + 1.0)
        ndcg_k: float = float(dcg.mean().item())

        metrics[f"recall@{cutoff}"] = recall_k
        metrics[f"mrr@{cutoff}"] = mrr_k
        metrics[f"ndcg@{cutoff}"] = ndcg_k

    return metrics
