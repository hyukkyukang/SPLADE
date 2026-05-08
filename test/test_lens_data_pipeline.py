"""End-to-end smoke test for the LENS data pipeline on synthetic data.

Exercises:
- ``SameDatasetBatchSampler`` — task-pure batches, deterministic per-epoch
  shuffle, per-rank slicing.
- ``LENSCollator`` — query template, ``<query>``-position pooling mask,
  doc-side last-two-zero mask, teacher_scores tensor, sub-batching.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

from src.data.sampler import SameDatasetBatchSampler


def test_same_dataset_batch_sampler_yields_task_pure_batches() -> None:
    # Two tasks of different sizes.
    task_a = np.arange(0, 24)       # 24 rows
    task_b = np.arange(24, 40)      # 16 rows
    sampler = SameDatasetBatchSampler(
        row_ranges=[task_a, task_b],
        per_task_batch_size=[4, 2],  # per-rank
        num_replicas=2,
        rank=0,
        seed=0,
        drop_last=True,
    )
    batches: List[List[int]] = list(sampler)
    # Each batch should have indices all in one task range.
    for batch in batches:
        assert len(batch) in (4, 2), f"unexpected local batch size: {len(batch)}"
        in_a = all(24 > i >= 0 for i in batch)
        in_b = all(40 > i >= 24 for i in batch)
        assert in_a ^ in_b, f"mixed-task batch: {batch}"


def test_same_dataset_batch_sampler_ranks_are_disjoint() -> None:
    task = np.arange(0, 64)
    sampler0 = SameDatasetBatchSampler(
        row_ranges=[task], per_task_batch_size=[8], num_replicas=2, rank=0, seed=42,
    )
    sampler1 = SameDatasetBatchSampler(
        row_ranges=[task], per_task_batch_size=[8], num_replicas=2, rank=1, seed=42,
    )
    b0 = sum(sampler0, [])
    b1 = sum(sampler1, [])
    assert len(b0) == len(b1)
    # No overlap expected given the same seed + different rank slices.
    assert set(b0).isdisjoint(set(b1))


def test_same_dataset_batch_sampler_set_epoch_changes_order() -> None:
    task = np.arange(0, 32)
    sampler = SameDatasetBatchSampler(
        row_ranges=[task], per_task_batch_size=[4], num_replicas=1, rank=0, seed=7,
    )
    first = [list(b) for b in sampler]
    sampler.set_epoch(1)
    second = [list(b) for b in sampler]
    # Same rows in both epochs (all task indices), but order differs.
    assert sorted(sum(first, [])) == sorted(sum(second, []))
    assert first != second


# --- LENSCollator on synthetic raw rows ---------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub with the subset of the API LENSCollator uses."""

    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "left"

    def __init__(self) -> None:
        # Vocabulary: word → id. We'll tokenize by word split and map unknown
        # words to a shared UNK id so the test doesn't depend on HF.
        self._vocab: dict[str, int] = {
            "<pad>": 0,
            "<instruct>": 1,
            "<query>": 2,
            "<response>": 3,
            "</s>": 4,
            "UNK": 5,
        }
        self._next_id: int = 6

    def _encode_word(self, w: str) -> int:
        if w not in self._vocab:
            self._vocab[w] = self._next_id
            self._next_id += 1
        return self._vocab[w]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, 5)

    def __call__(
        self,
        texts: list[str],
        *,
        max_length: int,
        truncation: bool,
        add_special_tokens: bool,
        return_attention_mask: bool,
        padding: bool,
    ) -> dict[str, list[list[int]]]:
        input_ids: list[list[int]] = []
        for t in texts:
            # Split on whitespace; treat '<..>' sub-strings as single tokens.
            tokens: list[str] = []
            for piece in t.replace("<instruct>", " <instruct> ") \
                          .replace("<query>", " <query> ") \
                          .replace("</s>", " </s> ") \
                          .split():
                tokens.append(piece)
            ids: list[int] = [self._encode_word(w) for w in tokens]
            ids = ids[:max_length]
            input_ids.append(ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(seq) for seq in input_ids],
        }

    def pad(self, features, padding=True, max_length=None,
            pad_to_multiple_of=None, return_tensors="pt"):
        max_len: int = max(len(x) for x in features["input_ids"])
        if pad_to_multiple_of:
            max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        padded_ids: list[list[int]] = []
        padded_am: list[list[int]] = []
        for seq, am in zip(features["input_ids"], features["attention_mask"]):
            pad_amt: int = max_len - len(seq)
            # Left pad (padding_side='left').
            padded_ids.append([self.pad_token_id] * pad_amt + list(seq))
            padded_am.append([0] * pad_amt + list(am))
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_am, dtype=torch.long),
        }


def test_lens_collator_produces_expected_structure() -> None:
    from src.data.lens_collator import LENSCollator

    tok = _FakeTokenizer()
    collator = LENSCollator(
        tokenizer=tok,
        train_group_size=3,
        sub_batch_size=2,
        query_max_len=32,
        passage_max_len=32,
        pad_to_multiple_of=1,
    )

    rows = [
        {
            "query": f"query {i}",
            "pos": [f"pos {i} doc"],
            "neg": [f"neg {i}a", f"neg {i}b"],
            "pos_scores": [1.0],
            "neg_scores": [0.1, 0.2],
            "prompt": "test prompt",
            "type": "retrieval",
        }
        for i in range(4)
    ]
    out = collator(rows)
    assert out["messages"] == "normal"
    assert out["train_group_size"] == 3
    # 4 queries × 3 passages = 12; with sub_batch=2 → 6 doc chunks.
    assert len(out["passage"]) == 6
    # 4 queries with sub_batch=2 → 2 query chunks.
    assert len(out["query"]) == 2

    # teacher_scores should be (4 * 3,) since each row provides pos_score + neg_scores for all negs.
    assert out["teacher_scores"] is not None
    assert out["teacher_scores"].shape == (12,)

    # Each sub-batch dict has the three expected keys with batched tensors.
    for sub in out["query"] + out["passage"]:
        assert set(sub.keys()) == {"input_ids", "attention_mask", "pooling_mask"}
        assert sub["input_ids"].shape == sub["attention_mask"].shape == sub["pooling_mask"].shape

    # Symmetric clustering task should flip messages -> not in-batch.
    out_cls = collator([{**rows[0], "type": "symmetric_class"}])
    assert out_cls["messages"] == "not in-batch"
