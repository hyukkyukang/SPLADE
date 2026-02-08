import unittest

import torch

from src.data.pd_module.scoring import ScoringItem
from src.data.pl_module.scoring import ScoringCollator


class DummyTokenizer:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.docs: list[str] = []

    def __call__(
        self,
        queries: list[str],
        docs: list[str],
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        _ = (padding, truncation, max_length, return_tensors)
        self.queries = list(queries)
        self.docs = list(docs)
        batch = len(self.queries)
        input_ids = torch.arange(batch * 2, dtype=torch.long).reshape(batch, 2)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ScoringCollatorTest(unittest.TestCase):
    def test_batch_tokenization_ordering(self) -> None:
        tokenizer = DummyTokenizer()
        collator = ScoringCollator(
            model_name="dummy",
            max_length=8,
            tokenize_chunk_size=2,
            local_files_only=True,
            tokenizer=tokenizer,
        )
        items = [
            ScoringItem(
                row={},
                qid="q1",
                doc_ids=["d1", "d2"],
                labels=[1.0, 0.0],
                doc_sources=["pos", "neg"],
                query_text="q1",
                doc_texts=["t1", "t2"],
            ),
            ScoringItem(
                row={},
                qid="q2",
                doc_ids=["d3"],
                labels=[0.0],
                doc_sources=["neg"],
                query_text="q2",
                doc_texts=["t3"],
            ),
        ]
        batch = collator(items)
        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch["pair_row_ids"], [0, 0, 1])
        self.assertEqual(batch["pair_doc_idxs"], [0, 1, 0])
        self.assertEqual(tokenizer.queries, ["q1", "q1", "q2"])
        self.assertEqual(tokenizer.docs, ["t1", "t2", "t3"])
        pair_tokens = batch["pair_tokens"]
        self.assertEqual(pair_tokens["input_ids"].shape[0], 3)


if __name__ == "__main__":
    unittest.main()
