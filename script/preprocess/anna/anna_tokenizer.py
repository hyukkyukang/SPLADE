# Port of ANNA tokenization (anna_final_tokenization3.py) as a Hugging Face
# PreTrainedTokenizerBase. No TensorFlow; vocab loaded with open().

from __future__ import annotations

import collections
import unicodedata
from pathlib import Path
from typing import Any

from transformers import AddedToken, PreTrainedTokenizerBase
from transformers.tokenization_python import PreTrainedTokenizer

try:
    import anna_fast_rs  # type: ignore[import-not-found]
except Exception:
    anna_fast_rs = None


def _convert_to_unicode(text: str | bytes) -> str:
    """Converts text to Unicode (Python 3), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    raise ValueError(f"Unsupported string type: {type(text)}")


def _load_vocab(vocab_file: str | Path) -> collections.OrderedDict[str, int]:
    """Loads a vocabulary file into an ordered dict (token -> index)."""
    vocab: collections.OrderedDict[str, int] = collections.OrderedDict()
    index = 0
    path = Path(vocab_file)
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            token = _convert_to_unicode(line).strip()
            if not token:
                continue
            vocab[token] = index
            index += 1
    return vocab


def _resolve_vocab_file(
    vocab_file: str | Path | None,
    *,
    fallback_vocab_file: str | Path | None = None,
    name_or_path: str | None = None,
) -> str:
    if vocab_file is not None:
        return str(vocab_file)
    if fallback_vocab_file is not None:
        return str(fallback_vocab_file)
    if name_or_path is not None:
        candidate = Path(name_or_path) / "vocab.txt"
        if candidate.is_file():
            return str(candidate)
    raise ValueError(
        "Could not resolve vocab_file for ANNA tokenizer. "
        "Expected vocab.txt in the tokenizer directory."
    )


def _whitespace_tokenize(text: str) -> list[str]:
    """Runs basic whitespace cleaning and splitting."""
    text = text.strip()
    if not text:
        return []
    return text.split()


def _is_whitespace(char: str) -> bool:
    if char in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(char) == "Zs"


def _is_control(char: str) -> bool:
    if char in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(char) in ("Cc", "Cf")


def _is_punctuation(char: str) -> bool:
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(char).startswith("P")


class _BasicTokenizer:
    """Basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(
        self, vocab: collections.OrderedDict[str, int], do_lower_case: bool = True
    ) -> None:
        self.vocab = vocab
        self.do_lower_case = do_lower_case

    def tokenize(self, text: str) -> list[str]:
        text = _convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = _whitespace_tokenize(text)
        split_tokens: list[str] = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            if token in self.vocab:
                split_tokens.append(token)
            else:
                split_tokens.extend(
                    self._find_punc_vocab(self._run_split_on_punc(token))
                )
        return _whitespace_tokenize(" ".join(split_tokens))

    def _run_strip_accents(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        output = [c for c in text if unicodedata.category(c) != "Mn"]
        return "".join(output)

    def _run_split_on_punc(self, text: str) -> list[str]:
        chars = list(text)
        i = 0
        start_new_word = True
        output: list[list[str]] = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _find_punc_vocab(self, split_punc: list[str]) -> list[str]:
        re_output: list[str] = []
        i = 0
        n = len(split_punc)
        while i < n:
            for ii in range(n):
                chunk = "".join(split_punc[i : n - ii])
                if chunk in self.vocab or chunk == split_punc[i]:
                    re_output.append(chunk)
                    i = n - ii
                    break
        return re_output

    def _tokenize_chinese_chars(self, text: str) -> str:
        output: list[str] = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.extend([" ", char, " "])
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp: int) -> bool:
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF):
            return True
        if (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F):
            return True
        if (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF):
            return True
        if (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F):
            return True
        return False

    def _clean_text(self, text: str) -> str:
        output: list[str] = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class _WordpieceTokenizer:
    """WordPiece tokenization (longest-match)."""

    def __init__(
        self,
        vocab: collections.OrderedDict[str, int],
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 200,
    ) -> None:
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text: str) -> list[str]:
        text = _convert_to_unicode(text)
        output_tokens: list[str] = []
        for token in _whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens: list[str] = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class AnnaTokenizer(PreTrainedTokenizer):
    """
    ANNA tokenizer: BERT-style WordPiece with custom BasicTokenizer logic
    (punctuation chunking by vocab, strip accents, Chinese char handling).
    HF-compatible; no TensorFlow. Load with trust_remote_code=True when
    loading from a directory that contains this module and tokenizer_config
    with auto_map.
    """

    vocab_files_names: dict[str, str] = {"vocab_file": "vocab.txt"}
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def __init__(
        self,
        vocab_file: str | Path,
        do_lower_case: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        max_input_chars_per_word: int = 200,
        **kwargs: Any,
    ) -> None:
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case
        self.max_input_chars_per_word = max_input_chars_per_word
        self._vocab = _load_vocab(vocab_file)
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        self._basic_tokenizer = _BasicTokenizer(
            vocab=self._vocab, do_lower_case=do_lower_case
        )
        self._wordpiece_tokenizer = _WordpieceTokenizer(
            vocab=self._vocab,
            unk_token=unk_token,
            max_input_chars_per_word=max_input_chars_per_word,
        )
        self._added_tokens_decoder: dict[int, AddedToken] = {}
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    def _tokenize(self, text: str) -> list[str]:
        split_tokens: list[str] = []
        for token in self._basic_tokenizer.tokenize(text):
            for sub in self._wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub)
        return split_tokens

    def tokenize(
        self,
        text: str,
        pair: str | None = None,
        add_special_tokens: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        _ = pair, add_special_tokens, kwargs
        return self._tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        unk_id = self._vocab.get(self.unk_token, 0)
        return self._vocab.get(token, unk_id)

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        return self._convert_token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self._inv_vocab.get(index, self.unk_token)

    def convert_ids_to_tokens(
        self,
        ids: int | list[int],
        skip_special_tokens: bool = False,
    ) -> str | list[str]:
        _ = skip_special_tokens
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(i) for i in ids]

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return (
            [self.cls_token_id]
            + token_ids_0
            + [self.sep_token_id]
            + token_ids_1
            + [self.sep_token_id]
        )

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        sep = self.sep_token_id
        cls = self.cls_token_id
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            return [
                1 if tid in self.all_special_ids else 0
                for tid in token_ids_0 + (token_ids_1 or [])
            ]
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str, ...]:
        import os

        if filename_prefix is not None:
            vocab_file = os.path.join(
                save_directory,
                f"{filename_prefix}-{self.vocab_files_names['vocab_file']}",
            )
        else:
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, _ in sorted(self._vocab.items(), key=lambda x: x[1]):
                writer.write(token + "\n")
        return (vocab_file,)


class AnnaTokenizerFast(AnnaTokenizer):
    """
    ANNA tokenizer fast variant backed by the optional `anna_fast_rs` extension.
    Falls back to the slow Python implementation when the extension is unavailable.
    """

    def __init__(
        self,
        vocab_file: str | Path | None,
        do_lower_case: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        max_input_chars_per_word: int = 200,
        **kwargs: Any,
    ) -> None:
        init_kwargs: dict[str, Any] = dict(kwargs)
        fallback_vocab_file: str | None = None
        if "vocab_file" in init_kwargs and init_kwargs["vocab_file"] is not None:
            fallback_vocab_file = str(init_kwargs["vocab_file"])
            del init_kwargs["vocab_file"]
        name_or_path: str | None = None
        if "name_or_path" in init_kwargs and init_kwargs["name_or_path"] is not None:
            name_or_path = str(init_kwargs["name_or_path"])
        resolved_vocab_file = _resolve_vocab_file(
            vocab_file,
            fallback_vocab_file=fallback_vocab_file,
            name_or_path=name_or_path,
        )
        super().__init__(
            vocab_file=resolved_vocab_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            max_input_chars_per_word=max_input_chars_per_word,
            **init_kwargs,
        )
        self._backend: Any | None = None
        self._backend_error: Exception | None = None
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        self._backend = None
        self._backend_error = None
        if anna_fast_rs is None:
            self._backend_error = ImportError(
                "anna_fast_rs is not available. Build and install the extension "
                "to enable the fast tokenizer."
            )
            return
        try:
            self._backend = anna_fast_rs.AnnaFastBackend(
                vocab_file=str(self.vocab_file),
                do_lower_case=bool(self.do_lower_case),
                max_input_chars_per_word=int(self.max_input_chars_per_word),
                unk_token=str(self.unk_token),
            )
        except Exception as exc:
            self._backend = None
            self._backend_error = exc

    def __getstate__(self) -> dict[str, Any]:
        state: dict[str, Any] = dict(self.__dict__)
        state["_backend"] = None
        state["_backend_error"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._initialize_backend()

    @property
    def is_fast(self) -> bool:
        return self._backend is not None

    def _tokenize(self, text: str) -> list[str]:
        if self._backend is None:
            return super()._tokenize(text)
        tokens: list[str] = self._backend.tokenize(str(text))
        return tokens


__all__ = ["AnnaTokenizer", "AnnaTokenizerFast"]
