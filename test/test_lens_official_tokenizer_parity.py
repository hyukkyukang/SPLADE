"""Verify that the SAME tokenizer object is used for both training (via
:func:`build_model_tokenizer`) and inference (via :func:`load_official_lens`)
when ``strict_official_lens_tokenizer=True``.

If these two paths drift, training and inference would produce different
token IDs for the same text, silently degrading retrieval scores in ways
that are hard to diagnose. This test pins the invariant.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase


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
            f"Requires {repo_id} tokenizer cached locally; run "
            "`huggingface-cli download mistralai/Mistral-7B-v0.1` first."
        )


def _representative_lens_inputs() -> list[str]:
    """Cover queries (with full LENS prompt template + suffix), short docs,
    and long docs."""
    return [
        # Retrieval query — full instruct/query/response template + EOS suffix.
        "<instruct>Given a question, retrieve Wikipedia passages that answer "
        "the question.\n<query>What is the capital of France?</s>",
        # Retrieval doc — title + body + EOS suffix.
        "Eiffel Tower The Eiffel Tower is a wrought-iron lattice tower on the "
        "Champ de Mars in Paris.</s>",
        # Short doc.
        "Hello world.</s>",
        # No suffix (raw query before encoder applies template).
        "What is AI?",
    ]


def test_inference_and_training_tokenizers_match(mistral_repo_id: str) -> None:
    """The strict-LENS tokenizer used by ``build_official_lens_tokenizer``
    must produce byte-identical IDs to the one returned by
    ``build_model_tokenizer`` with ``strict_official_lens_tokenizer=True``."""
    _requires_mistral_cached(mistral_repo_id)

    from src.data.pl_module.common import build_model_tokenizer
    from src.utils.lens_official_loader import build_official_lens_tokenizer

    inference_tokenizer: PreTrainedTokenizerBase = build_official_lens_tokenizer(
        local_files_only=True,
    )

    train_cfg = OmegaConf.create({
        "huggingface_name": mistral_repo_id,
        "use_fast_tokenizer": False,
        "require_fast_tokenizer": False,
        "trust_remote_code": False,
        "local_files_only": True,
        "strict_official_lens_tokenizer": True,
    })
    training_tokenizer: PreTrainedTokenizerBase = build_model_tokenizer(train_cfg)

    for text in _representative_lens_inputs():
        a = inference_tokenizer(text, add_special_tokens=True)["input_ids"]
        b = training_tokenizer(text, add_special_tokens=True)["input_ids"]
        assert a == b, (
            "Strict-LENS tokenizers diverged between training and inference "
            f"paths for input {text!r}:\n  inference={a!r}\n  training={b!r}"
        )

    # Also: shared properties.
    assert inference_tokenizer.padding_side == training_tokenizer.padding_side
    assert inference_tokenizer.pad_token == training_tokenizer.pad_token
    assert inference_tokenizer.pad_token_id == training_tokenizer.pad_token_id
    for special in ("<instruct>", "<query>", "<response>"):
        a_id = inference_tokenizer.convert_tokens_to_ids(special)
        b_id = training_tokenizer.convert_tokens_to_ids(special)
        assert a_id == b_id, (
            f"{special} id differs: inference={a_id}, training={b_id}"
        )
