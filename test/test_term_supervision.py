import tempfile
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset

from src.data.term_supervision import OrderedMaskSlotTermSupervisor


class _DummyTokenizer:
    def __init__(self) -> None:
        self.name_or_path = "dummy-tokenizer"
        self.all_special_ids = [0, 99]
        self.mask_token_id = 99
        self._vocab = {
            "[PAD]": 0,
            "[MASK]": 99,
            "alpha": 1,
            "beta": 2,
            "gamma": 3,
            "delta": 4,
            "epsilon": 5,
        }

    def __len__(self) -> int:
        return 128

    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = False,
        padding=False,
        truncation: bool = False,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
        max_length: int | None = None,
    ):
        _ = add_special_tokens
        _ = padding
        _ = truncation
        _ = return_attention_mask
        _ = return_tensors
        _ = max_length
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [[self._vocab.get(piece, 0) for piece in row.split()] for row in texts]
        if isinstance(text, str):
            return {"input_ids": encoded[0]}
        return {"input_ids": encoded}


class _DummyDataset:
    def __init__(self) -> None:
        self.name = "dummy"
        self.hf_name = "dummy-hf"
        self.query_corpus_hf_name = "dummy-qc"
        self.hf_split = "train"
        self.query_text_column_name = "query"
        self.corpus_text_column_name = "passage"
        self.query_dataset = Dataset.from_dict(
            {
                "query": [
                    "alpha beta",
                    "alpha gamma",
                    "delta epsilon",
                ]
            }
        )
        self.corpus_dataset = Dataset.from_dict(
            {
                "passage": [
                    "alpha beta beta",
                    "gamma delta",
                    "epsilon epsilon alpha",
                ]
            }
        )


class OrderedMaskSlotTermSupervisorTest(unittest.TestCase):
    def test_prepare_builds_cache_and_reuses_it(self) -> None:
        dataset = _DummyDataset()
        tokenizer = _DummyTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = OrderedMaskSlotTermSupervisor(
                dataset=dataset,
                tokenizer=tokenizer,
                cache_dir=tmpdir,
                idf_batch_size=2,
                idf_log_interval=1,
            )
            supervisor.prepare()
            cache_path = supervisor._cache_path()
            self.assertTrue(cache_path.is_file())

            cached_supervisor = OrderedMaskSlotTermSupervisor(
                dataset=dataset,
                tokenizer=tokenizer,
                cache_dir=tmpdir,
                idf_batch_size=2,
                idf_log_interval=1,
            )
            with patch.object(
                cached_supervisor,
                "_build_idf_for_queries",
                side_effect=AssertionError("should not rebuild query idf"),
            ), patch.object(
                cached_supervisor,
                "_build_idf_for_corpus",
                side_effect=AssertionError("should not rebuild corpus idf"),
            ):
                cached_supervisor.prepare()

    def test_top_k_targets_use_global_idf_with_local_tf(self) -> None:
        dataset = _DummyDataset()
        tokenizer = _DummyTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            supervisor = OrderedMaskSlotTermSupervisor(
                dataset=dataset,
                tokenizer=tokenizer,
                cache_dir=tmpdir,
                idf_batch_size=2,
                idf_log_interval=1,
            )
            # beta is rarer than alpha in the query set, so it should outrank alpha.
            query_targets = supervisor.top_k_query_target_ids(
                "alpha beta beta",
                k=2,
                ignore_index=-100,
            )
            self.assertTrue(torch.equal(query_targets, torch.tensor([2, 1])))

            # epsilon has both higher TF and higher IDF than alpha in the corpus set.
            doc_targets = supervisor.top_k_doc_target_ids(
                "epsilon epsilon alpha",
                k=2,
                ignore_index=-100,
            )
            self.assertTrue(torch.equal(doc_targets, torch.tensor([5, 1])))


if __name__ == "__main__":
    unittest.main()
