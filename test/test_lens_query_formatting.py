import unittest

import torch
from omegaconf import OmegaConf

from src.data.lens_formatting import (
    build_doc_pooling_mask,
    build_query_pooling_mask,
    format_query_text,
    validate_lens_tokenizer,
)


class _DummyTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {
            "<instruct>": 10,
            "<query>": 11,
            "<response>": 12,
            "find": 20,
            "cats": 21,
            "dogs": 22,
        }
        self.all_special_ids: list[int] = [0, 10, 11, 12]

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


class LensQueryFormattingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _DummyTokenizer()
        self.model_cfg = OmegaConf.create(
            {
                "family": "lens",
                "instruction_text": "Find a relevant document.",
                "instruction_template": (
                    "{instruction_token}{instruction}\n{query_token}{query}"
                ),
                "instruction_token": "<instruct>",
                "query_token": "<query>",
                "response_token": "<response>",
                "query_mask_mode": "after_query_token",
                "doc_trim_last_tokens": 2,
            }
        )

    def test_format_query_text_applies_lens_template(self) -> None:
        formatted: str = format_query_text("cats", self.model_cfg)
        self.assertEqual(
            formatted,
            "<instruct>Find a relevant document.\n<query>cats",
        )

    def test_build_query_pooling_mask_starts_after_query_token(self) -> None:
        input_ids = torch.tensor([[10, 30, 11, 20, 21, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long)

        pooling_mask = build_query_pooling_mask(
            input_ids,
            attention_mask,
            self.tokenizer,
            self.model_cfg,
        )

        expected = torch.tensor([[0, 0, 0, 1, 1, 0]], dtype=torch.long)
        self.assertTrue(torch.equal(pooling_mask, expected))

    def test_build_query_pooling_mask_keeps_continuation_windows(self) -> None:
        input_ids = torch.tensor([[10, 30, 11, 20], [0, 21, 22, 12]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long)

        pooling_mask = build_query_pooling_mask(
            input_ids,
            attention_mask,
            self.tokenizer,
            self.model_cfg,
            missing_query_token_mode="keep",
        )

        expected = torch.tensor([[0, 0, 0, 1], [0, 1, 1, 0]], dtype=torch.long)
        self.assertTrue(torch.equal(pooling_mask, expected))

    def test_build_doc_pooling_mask_trims_last_active_tokens(self) -> None:
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
        pooling_mask = build_doc_pooling_mask(attention_mask, self.model_cfg)
        expected = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.long)
        self.assertTrue(torch.equal(pooling_mask, expected))

    def test_validate_lens_tokenizer_rejects_missing_special_tokens(self) -> None:
        incomplete_tokenizer = _DummyTokenizer()
        incomplete_tokenizer._vocab.pop("<response>")
        with self.assertRaisesRegex(ValueError, "Missing: <response>"):
            validate_lens_tokenizer(incomplete_tokenizer, self.model_cfg)


if __name__ == "__main__":
    unittest.main()
