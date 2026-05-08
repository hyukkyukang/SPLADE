import unittest
from unittest.mock import patch

from src.utils.transformers import build_masked_lm_model, build_tokenizer


class _DummyTokenizer:
    def __init__(self) -> None:
        self.is_fast: bool = True
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.cls_token = "[CLS]"


class _DummyLoader:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: object) -> object:
        cls.calls.append((model_name_or_path, dict(kwargs)))
        return object()


class TransformersDiffusionLoaderTest(unittest.TestCase):
    def test_build_masked_lm_model_forwards_diffusion_loader_kwargs(self) -> None:
        _DummyLoader.calls.clear()
        with patch(
            "src.utils.transformers._resolve_hf_model_loader",
            return_value=("AutoModelForMaskedLM", _DummyLoader),
        ):
            _ = build_masked_lm_model(
                "org/diffusion-backbone",
                trust_remote_code=True,
                revision="exp-branch",
                local_files_only=True,
            )

        self.assertEqual(len(_DummyLoader.calls), 1)
        model_name_or_path, kwargs = _DummyLoader.calls[0]
        self.assertEqual(model_name_or_path, "org/diffusion-backbone")
        self.assertTrue(bool(kwargs["trust_remote_code"]))
        self.assertEqual(kwargs["revision"], "exp-branch")
        self.assertTrue(bool(kwargs["local_files_only"]))

    def test_build_tokenizer_forwards_revision_and_local_files_only(self) -> None:
        dummy_tokenizer = _DummyTokenizer()

        with patch(
            "src.utils.transformers.AutoTokenizer.from_pretrained",
            return_value=dummy_tokenizer,
        ) as mocked_from_pretrained:
            tokenizer = build_tokenizer(
                "org/diffusion-tokenizer",
                trust_remote_code=True,
                local_files_only=True,
                revision="tok-rev",
            )

        self.assertIs(tokenizer, dummy_tokenizer)
        _, kwargs = mocked_from_pretrained.call_args
        self.assertTrue(bool(kwargs["trust_remote_code"]))
        self.assertTrue(bool(kwargs["local_files_only"]))
        self.assertEqual(kwargs["revision"], "tok-rev")

    def test_build_masked_lm_model_supports_local_udlm_loader(self) -> None:
        sentinel = object()
        with patch(
            "src.utils.transformers.UDLMForMaskedLMCompat.from_pretrained",
            return_value=sentinel,
        ) as mocked_from_pretrained:
            loaded = build_masked_lm_model(
                "kuleshov-group/udlm-lm1b",
                model_class_name="UDLMForMaskedLMCompat",
                trust_remote_code=False,
                revision="rev-123",
            )

        self.assertIs(loaded, sentinel)
        mocked_from_pretrained.assert_called_once()
        _, kwargs = mocked_from_pretrained.call_args
        self.assertFalse(bool(kwargs["trust_remote_code"]))
        self.assertEqual(kwargs["revision"], "rev-123")


if __name__ == "__main__":
    unittest.main()
