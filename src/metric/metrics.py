from __future__ import annotations

import math
from typing import Iterable


def mrr_at_k(qrels: dict[str, dict[str, int]], results: dict[str, list[str]], k: int) -> float:
    total = 0.0
    count = 0
    for qid, ranking in results.items():
        rel_docs = qrels.get(qid, {})
        rr = 0.0
        for rank, doc_id in enumerate(ranking[:k], start=1):
            if rel_docs.get(doc_id, 0) > 0:
                rr = 1.0 / rank
                break
        total += rr
        count += 1
    return total / max(count, 1)


def recall_at_k(
    qrels: dict[str, dict[str, int]], results: dict[str, list[str]], k: int
) -> float:
    total = 0.0
    count = 0
    for qid, ranking in results.items():
        rel_docs = {doc_id for doc_id, rel in qrels.get(qid, {}).items() if rel > 0}
        if not rel_docs:
            continue
        retrieved = set(ranking[:k])
        total += len(rel_docs & retrieved) / len(rel_docs)
        count += 1
    return total / max(count, 1)


def ndcg_at_k(
    qrels: dict[str, dict[str, int]], results: dict[str, list[str]], k: int
) -> float:
    total = 0.0
    count = 0
    for qid, ranking in results.items():
        rel_docs = qrels.get(qid, {})
        dcg = 0.0
        for rank, doc_id in enumerate(ranking[:k], start=1):
            rel = rel_docs.get(doc_id, 0)
            if rel > 0:
                dcg += (2**rel - 1) / math.log2(rank + 1)
        ideal_rels = sorted(rel_docs.values(), reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels[:k], start=1):
            if rel > 0:
                idcg += (2**rel - 1) / math.log2(rank + 1)
        if idcg > 0:
            total += dcg / idcg
            count += 1
    return total / max(count, 1)
