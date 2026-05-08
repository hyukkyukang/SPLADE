"""LENS training collator.

Consumes one batch-worth of raw rows from :class:`src.data.dataset.bge_full_data.BgeFullDataset`
(all of the same ``type``) and produces the structure the LENS training step
expects::

    {
      "query":        [sub_batch_dict, ...],   # each: input_ids, attention_mask, pooling_mask
      "passage":      [sub_batch_dict, ...],   # same keys
      "messages":     "normal" | "not in-batch",
      "teacher_scores": torch.FloatTensor | None,  # shape (B * train_group_size,)
      "task_type":    str,
    }

Mirrors ``Yibin-Lei/LENS/finetune/data.py::SameEmbedCollator`` semantics
exactly: queries use the ``<instruct>{prompt}\\n<query>{text}`` template and a
``<query>``-position pooling mask; passages zero out the last two attention
positions; sub-batches allow fitting large per-step passage counts on smaller
GPUs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


_QUERY_TOKEN: str = "<query>"
_INSTRUCT_TOKEN: str = "<instruct>"


def _format_instructed_query(prompt: str, query: str, use_special_tokens: bool) -> str:
    if use_special_tokens:
        return f"{_INSTRUCT_TOKEN}{prompt}\n{_QUERY_TOKEN}{query}"
    return f"Instruct: {prompt}\nQuery: {query}"


def _build_query_pooling_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    query_token_id: int,
) -> Tensor:
    """For each row: 1 between the last <query> token and -2 (exclusive of last 2)."""
    mask: Tensor = torch.zeros_like(attention_mask)
    for idx in range(input_ids.size(0)):
        seq = input_ids[idx]
        special_pos = (seq == query_token_id).nonzero()
        if len(special_pos) > 0:
            last_pos: int = int(special_pos[-1].item())
            mask[idx, last_pos:-2] = 1
        else:
            mask[idx] = attention_mask[idx]
    return mask


def _build_doc_pooling_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    query_token_id: int,
) -> Tensor:
    """For docs: either (a) no ``<query>`` token present → attention mask with
    last 2 positions zeroed, or (b) symmetric task that wraps docs in the same
    query template → apply query-style mask."""
    mask: Tensor = torch.zeros_like(attention_mask)
    for idx in range(input_ids.size(0)):
        seq = input_ids[idx]
        special_pos = (seq == query_token_id).nonzero()
        if len(special_pos) > 0:
            last_pos: int = int(special_pos[-1].item())
            mask[idx, last_pos:-2] = 1
        else:
            mask[idx] = attention_mask[idx]
            mask[idx, -2:] = 0
    return mask


@dataclass
class LENSCollator:
    """Build one LENS training step from a batch of raw rows.

    Parameters
    ----------
    tokenizer
        The LENS tokenizer (strict official: slow, left-pad, add_eos_token).
    train_group_size
        Number of passages per query (1 positive + N-1 negatives) for
        asymmetric tasks (retrieval, reranking).
    symmetric_train_group_size
        Group size for STS/clustering/class tasks whose passages also get
        wrapped in the query template.
    max_class_neg
        Max negatives per query in class-wise symmetric tasks.
    query_max_len, passage_max_len
        Token truncation lengths.
    sub_batch_size
        Max sequences per encoder forward pass. Each sub-batch is tokenized /
        padded independently.
    use_special_tokens
        Whether to use the ``<instruct>``/``<query>`` tokens (True matches the
        paper).
    pad_to_multiple_of
        Pad to a multiple of this many tokens for tensor-core efficiency.
    """

    tokenizer: PreTrainedTokenizerBase
    train_group_size: int = 8
    symmetric_train_group_size: int = 8
    max_class_neg: int = 7
    query_max_len: int = 512
    passage_max_len: int = 512
    sub_batch_size: int = 8
    use_special_tokens: bool = True
    pad_to_multiple_of: int = 8
    _query_token_id: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        self._query_token_id = int(
            self.tokenizer.convert_tokens_to_ids(_QUERY_TOKEN)
        )

    # --- Entry point -----------------------------------------------------------

    def __call__(self, raw_batch: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> Dict[str, Any]:
        # DataLoader with batch_sampler passes us a list of rows already
        # comprising a full batch. Some callers pre-collate into a dict-of-lists.
        if isinstance(raw_batch, Mapping):
            rows: List[Mapping[str, Any]] = _dict_of_lists_to_rows(raw_batch)
        else:
            rows = list(raw_batch)
        if not rows:
            raise ValueError("Empty batch passed to LENSCollator.")

        finetune_type: str = str(rows[0].get("type", ""))
        effective_group_size: int = self._resolve_group_size(finetune_type)

        queries: List[str] = []
        passages: List[str] = []
        scores: List[float] = []
        is_symmetric: bool = ("symmetric" in finetune_type) and (
            "sts" in finetune_type or "clustering" in finetune_type
        )

        for row in rows:
            raw_query: str = str(row["query"])
            prompt: str = str(row.get("prompt", ""))
            q_text: str = _format_instructed_query(
                raw_query, prompt, self.use_special_tokens
            )
            queries.append(q_text)

            pos_list: List[str] = list(row["pos"])
            neg_list: List[str] = list(row["neg"])
            pos_scores: Optional[List[float]] = row.get("pos_scores")
            neg_scores: Optional[List[float]] = row.get("neg_scores")

            # Pick one positive.
            pos_idx: int = random.randrange(len(pos_list))
            pos_doc: str = pos_list[pos_idx]
            if pos_scores is not None and pos_idx < len(pos_scores) and pos_scores[pos_idx] is not None:
                scores.append(float(pos_scores[pos_idx]))

            # Sample ``effective_group_size - 1`` negatives.
            needed_neg: int = effective_group_size - 1
            neg_idxs: List[int]
            if len(neg_list) == 0:
                neg_idxs = []
            elif len(neg_list) < needed_neg:
                repeats: int = math.ceil(needed_neg / max(len(neg_list), 1))
                pool: List[int] = list(range(len(neg_list))) * repeats
                neg_idxs = random.sample(pool, needed_neg)
            else:
                neg_idxs = random.sample(range(len(neg_list)), needed_neg)
            negatives: List[str] = [neg_list[i] for i in neg_idxs]

            if neg_scores is not None:
                for i in neg_idxs:
                    if i < len(neg_scores) and neg_scores[i] is not None:
                        scores.append(float(neg_scores[i]))

            row_passages: List[str] = [pos_doc] + negatives
            if is_symmetric:
                # Wrap passages with the query template too.
                row_passages = [
                    _format_instructed_query(p, prompt, self.use_special_tokens)
                    for p in row_passages
                ]
            passages.extend(row_passages)

        is_not_in_batch: bool = ("symmetric" in finetune_type) and (
            "class" in finetune_type or "clustering" in finetune_type
        )
        message: str = "not in-batch" if is_not_in_batch else "normal"

        teacher_scores_tensor: Optional[Tensor] = None
        if len(scores) == len(queries) * effective_group_size:
            teacher_scores_tensor = torch.tensor(scores, dtype=torch.float32)

        # Tokenize the full list of queries/passages in one pass, then chunk
        # into sub-batches for padding + mask construction.
        q_tok = self.tokenizer(
            queries,
            max_length=self.query_max_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=False,
        )
        p_tok = self.tokenizer(
            passages,
            max_length=self.passage_max_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=False,
        )
        q_subs: List[Dict[str, Tensor]] = self._chunk_and_pad(
            q_tok,
            max_len=self.query_max_len,
            mode="query",
        )
        p_subs: List[Dict[str, Tensor]] = self._chunk_and_pad(
            p_tok,
            max_len=self.passage_max_len,
            mode="doc",
        )
        return {
            "query": q_subs,
            "passage": p_subs,
            "messages": message,
            "teacher_scores": teacher_scores_tensor,
            "task_type": finetune_type,
            "train_group_size": effective_group_size,
        }

    # --- Internals ------------------------------------------------------------

    def _resolve_group_size(self, finetune_type: str) -> int:
        t: str = str(finetune_type)
        if "symmetric" in t and ("sts" in t or "clustering" in t):
            return self.symmetric_train_group_size
        if "only_1neg" in t:
            return 2
        if "symmetric" in t and "class" in t:
            return self.max_class_neg + 1
        return self.train_group_size

    def _chunk_and_pad(
        self,
        encoded: Any,
        *,
        max_len: int,
        mode: str,
    ) -> List[Dict[str, Tensor]]:
        result: List[Dict[str, Tensor]] = []
        total: int = len(encoded["input_ids"])
        sub_bs: int = max(self.sub_batch_size, 1)
        for start in range(0, total, sub_bs):
            end: int = min(start + sub_bs, total)
            features: Dict[str, List[List[int]]] = {
                "input_ids": encoded["input_ids"][start:end],
                "attention_mask": encoded["attention_mask"][start:end],
            }
            padded: Dict[str, Tensor] = self.tokenizer.pad(
                features,
                padding=True,
                max_length=max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            pooling_mask: Tensor
            if mode == "query":
                pooling_mask = _build_query_pooling_mask(
                    padded["input_ids"],
                    padded["attention_mask"],
                    self._query_token_id,
                )
            else:
                pooling_mask = _build_doc_pooling_mask(
                    padded["input_ids"],
                    padded["attention_mask"],
                    self._query_token_id,
                )
            padded["pooling_mask"] = pooling_mask
            result.append(padded)
        return result


def _dict_of_lists_to_rows(batch: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if not batch:
        return []
    keys: List[str] = list(batch.keys())
    length: int = len(batch[keys[0]])
    return [{k: batch[k][i] for k in keys} for i in range(length)]


# `Mapping` used inside this module; imported lazily to keep the public
# signature of the file minimal.
from typing import Mapping  # noqa: E402  (deliberate late import)


__all__ = ["LENSCollator"]
