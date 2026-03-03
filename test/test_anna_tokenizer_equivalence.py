"""Tests for ANNA tokenizer: in-repo AnnaTokenizer and HF export loading."""

import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from script.preprocess.anna.anna_tokenizer import AnnaTokenizer
from src.utils.transformers import build_tokenizer

_EDGE_TEXTS: tuple[str, ...] = (
    "",
    " ",
    "anna conversion validation",
    "Hello, world!",
    "lower UPPER 123",
    "punctuation: period. comma, semi; colon:",
)


def _minimal_vocab_path() -> Path:
    """Write a minimal BERT-style vocab for tests."""
    vocab_lines = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "hello",
        "world",
        "!",
        ",",
        ".",
        "##lo",
        "##rld",
        "anna",
        "conversion",
        "validation",
        "lower",
        "upper",
        "123",
        "punctuation",
        ":",
        "period",
        "comma",
        "semi",
        ";",
        "colon",
    ]
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="vocab_")
    with open(fd, "w", encoding="utf-8") as f:
        for line in vocab_lines:
            f.write(line + "\n")
    return Path(path)


def _minimal_hf_anna_dir(vocab_path: Path) -> Path:
    hf_dir = Path(tempfile.mkdtemp(prefix="anna_hf_"))
    (hf_dir / "vocab.txt").write_text(vocab_path.read_text(encoding="utf-8"), encoding="utf-8")
    tokenizer_config: dict[str, object] = {
        "auto_map": {
            "AutoTokenizer": [
                "anna_tokenizer.AnnaTokenizer",
                "anna_tokenizer.AnnaTokenizerFast",
            ]
        },
        "tokenizer_class": "AnnaTokenizer",
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "model_max_length": 512,
    }
    (hf_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2),
        encoding="utf-8",
    )
    special_tokens_map: dict[str, str] = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
    }
    (hf_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2),
        encoding="utf-8",
    )
    source_tokenizer_module = Path("script/preprocess/anna/anna_tokenizer.py")
    if not source_tokenizer_module.is_file():
        raise FileNotFoundError(f"Missing ANNA tokenizer module: {source_tokenizer_module}")
    (hf_dir / "anna_tokenizer.py").write_text(
        source_tokenizer_module.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return hf_dir


class AnnaTokenizerStandaloneTest(unittest.TestCase):
    """Test in-repo AnnaTokenizer (no TensorFlow, no HF export)."""

    def setUp(self) -> None:
        self._vocab_path: Path = _minimal_vocab_path()

    def tearDown(self) -> None:
        if self._vocab_path.is_file():
            self._vocab_path.unlink()

    def test_tokenize_edge_cases(self) -> None:
        tokenizer = AnnaTokenizer(vocab_file=str(self._vocab_path), do_lower_case=True)
        for text in _EDGE_TEXTS:
            tokens = tokenizer.tokenize(text)
            self.assertIsInstance(tokens, list)
            self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_convert_tokens_to_ids_roundtrip(self) -> None:
        tokenizer = AnnaTokenizer(vocab_file=str(self._vocab_path), do_lower_case=True)
        text = "hello world"
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertIsInstance(ids, list)
        self.assertEqual(len(ids), len(tokens))
        back = tokenizer.convert_ids_to_tokens(ids)
        self.assertEqual(back, tokens)

    def test_vocab_size_and_special_ids(self) -> None:
        tokenizer = AnnaTokenizer(vocab_file=str(self._vocab_path), do_lower_case=True)
        self.assertGreater(tokenizer.vocab_size, 0)
        self.assertIn(tokenizer.pad_token_id, tokenizer.all_special_ids)
        self.assertIn(tokenizer.cls_token_id, tokenizer.all_special_ids)
        self.assertIn(tokenizer.sep_token_id, tokenizer.all_special_ids)


class AnnaHfExportLoadTest(unittest.TestCase):
    """Test loading exported ANNA HF dir with trust_remote_code (when dir exists)."""

    def setUp(self) -> None:
        self._vocab_path: Path = _minimal_vocab_path()
        self._anna_hf_dir: Path = _minimal_hf_anna_dir(self._vocab_path)

    def tearDown(self) -> None:
        if self._vocab_path.is_file():
            self._vocab_path.unlink()
        if self._anna_hf_dir.is_dir():
            shutil.rmtree(self._anna_hf_dir)

    def test_build_tokenizer_trust_remote_code_loads_anna_hf(self) -> None:
        tokenizer = build_tokenizer(
            str(self._anna_hf_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        tokens = tokenizer.tokenize("anna conversion validation")
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(t, str) for t in tokens))
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertEqual(len(ids), len(tokens))

    def test_auto_tokenizer_trust_remote_code_loads_anna_hf(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(self._anna_hf_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
        encoded = tokenizer(
            "hello world",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=16,
        )
        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)

    def test_auto_tokenizer_use_fast_true_loads_fast_backend(self) -> None:
        tokenizer_fast = AutoTokenizer.from_pretrained(
            str(self._anna_hf_dir),
            local_files_only=True,
            use_fast=True,
            trust_remote_code=True,
        )
        self.assertTrue(bool(tokenizer_fast.is_fast))
        tokenizer_slow_reference = AnnaTokenizer(
            vocab_file=str(self._vocab_path),
            do_lower_case=True,
        )
        for text in _EDGE_TEXTS:
            self.assertEqual(
                tokenizer_slow_reference.tokenize(text),
                tokenizer_fast.tokenize(text),
            )

    def test_build_tokenizer_can_require_fast_backend(self) -> None:
        tokenizer = build_tokenizer(
            str(self._anna_hf_dir),
            use_fast_tokenizer=True,
            trust_remote_code=True,
            require_fast_tokenizer=True,
            local_files_only=True,
        )
        self.assertTrue(bool(tokenizer.is_fast))

    def test_fast_tokenizer_pickle_roundtrip(self) -> None:
        tokenizer_fast = AutoTokenizer.from_pretrained(
            str(self._anna_hf_dir),
            local_files_only=True,
            use_fast=True,
            trust_remote_code=True,
        )
        payload: bytes = pickle.dumps(tokenizer_fast)
        restored = pickle.loads(payload)
        text = "anna conversion validation"
        self.assertEqual(tokenizer_fast.tokenize(text), restored.tokenize(text))


if __name__ == "__main__":
    unittest.main()
