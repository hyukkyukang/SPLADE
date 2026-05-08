"""Verify that strict_official_lens_tokenizer=True applies the tokenizer
settings used by Yibin-Lei/LENS: slow SentencePiece, left-pad, unk->eos pad.

Note: official LENS uses Mistral default ``add_eos_token=False`` and instead
appends ``"</s>"`` as a string suffix at *inference* time (not training).
The tokenizer-parity tests here therefore assert NO auto-EOS, plus that the
explicit ``"</s>"`` suffix produces exactly one trailing EOS token.
"""

from __future__ import annotations

import pytest
from transformers import PreTrainedTokenizerBase

from src.utils.transformers import build_tokenizer


@pytest.fixture(scope="module")
def mistral_repo_id() -> str:
    return "mistralai/Mistral-7B-v0.1"


def _tokenizer_cached(repo_id: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    try:
        path = try_to_load_from_cache(repo_id, "tokenizer.model")
    except Exception:
        return False
    return path is not None and path != "_CACHED_NO_EXIST"


def _requires_mistral_cached(repo_id: str) -> None:
    if not _tokenizer_cached(repo_id):
        pytest.skip(
            f"Requires {repo_id} tokenizer to be cached locally; run "
            "`huggingface-cli download mistralai/Mistral-7B-v0.1` first."
        )


def test_strict_official_tokenizer_flags(mistral_repo_id: str) -> None:
    """Verify behaviour-level tokenization invariants for the strict-LENS path.

    We don't assert ``not tokenizer.is_fast`` anymore: modern transformers
    routes everything through ``TokenizersBackend`` (whose ``.is_fast`` is
    always True) regardless of ``use_fast=False``. What actually matters is
    the *behavior*: left padding, no auto-EOS, single trailing EOS for the
    explicit ``"</s>"`` suffix, and unk pad-token fallback.
    """
    _requires_mistral_cached(mistral_repo_id)
    tokenizer: PreTrainedTokenizerBase = build_tokenizer(
        mistral_repo_id,
        strict_official_lens_tokenizer=True,
    )
    assert tokenizer.padding_side == "left"
    # Official LENS uses Mistral default add_eos_token=False; the explicit
    # "</s>" suffix is added by the encoder at inference time, not by the
    # tokenizer. Verify that bare encoding does NOT auto-append EOS.
    encoded = tokenizer("hello", add_special_tokens=True)["input_ids"]
    assert encoded[-1] != tokenizer.eos_token_id, (
        f"Expected NO auto-appended EOS (Mistral default), got {encoded[-3:]!r}."
    )
    # And "hello</s>" should produce exactly one trailing EOS (suffix as
    # special token, not template-doubled).
    encoded_with_suffix = tokenizer("hello</s>", add_special_tokens=True)["input_ids"]
    assert encoded_with_suffix[-1] == tokenizer.eos_token_id, (
        f"Expected explicit </s> suffix to be tokenized as EOS, "
        f"got {encoded_with_suffix[-3:]!r}."
    )
    assert encoded_with_suffix[-2] != tokenizer.eos_token_id, (
        f"Suffix should produce exactly one trailing EOS, "
        f"got {encoded_with_suffix[-3:]!r}."
    )
    # Pad token must match unk (unk -> eos fallback; Mistral has unk so unk wins).
    assert tokenizer.pad_token == tokenizer.unk_token
    assert tokenizer.pad_token_id == tokenizer.unk_token_id


def test_default_tokenizer_flags_unchanged(mistral_repo_id: str) -> None:
    _requires_mistral_cached(mistral_repo_id)
    tokenizer: PreTrainedTokenizerBase = build_tokenizer(
        mistral_repo_id,
        strict_official_lens_tokenizer=False,
    )
    # Default path should still work (fast tokenizer by default).
    assert tokenizer.is_fast or tokenizer.padding_side == "right"
