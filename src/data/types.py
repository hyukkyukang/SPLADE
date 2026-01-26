from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class DataTuple:
    qid: str
    pos_ids: List[str]
    pos_scores: List[float]
    neg_ids: List[str]


@dataclass
class RerankingDataItem:
    data_idx: int
    qid: str
    pos_ids: List[str]
    neg_ids: List[str]
    query_text: str
    doc_texts: List[str]
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    doc_mask: torch.Tensor
    pos_mask: torch.Tensor
    teacher_scores: torch.Tensor


@dataclass
class RetrievalDataItem:
    data_idx: int
    qid: str
    relevance_judgments: Dict[str, float]
    query_text: str
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
