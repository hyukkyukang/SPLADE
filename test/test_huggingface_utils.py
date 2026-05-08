import os
import unittest
from unittest.mock import patch

from src.utils.huggingface import resolve_hf_token


class HuggingFaceUtilsTest(unittest.TestCase):
    def test_resolve_hf_token_prefers_hf_token(self) -> None:
        with patch.dict(
            os.environ,
            {"HF_TOKEN": " token-a ", "HUGGINGFACE_HUB_TOKEN": "token-b"},
            clear=False,
        ):
            self.assertEqual(resolve_hf_token(), "token-a")

    def test_resolve_hf_token_falls_back_to_hub_token(self) -> None:
        with patch.dict(
            os.environ,
            {"HUGGINGFACE_HUB_TOKEN": " token-b "},
            clear=True,
        ):
            self.assertEqual(resolve_hf_token(), "token-b")

    def test_resolve_hf_token_falls_back_to_legacy_hub_token(self) -> None:
        with patch.dict(
            os.environ,
            {"HUGGING_FACE_HUB_TOKEN": " token-c "},
            clear=True,
        ):
            self.assertEqual(resolve_hf_token(), "token-c")

    def test_resolve_hf_token_returns_none_when_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_hf_token())


if __name__ == "__main__":
    unittest.main()
