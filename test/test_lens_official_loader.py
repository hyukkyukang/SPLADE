"""Smoke test for :func:`src.utils.lens_official_loader.load_official_lens`.

Gated on the LENS-d4000 repo being cached locally to keep unit tests offline.
"""

from __future__ import annotations

import pytest
import torch

from src.utils.lens_official_loader import load_official_lens


def _repo_cached(repo_id: str, filename: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    try:
        path = try_to_load_from_cache(repo_id, filename)
    except Exception:
        return False
    return path is not None and path != "_CACHED_NO_EXIST"


def _cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 1


@pytest.mark.parametrize("repo_id,expected_dim", [
    ("yibinlei/LENS-d4000", 4000),
])
def test_load_official_lens_smoke(repo_id: str, expected_dim: int) -> None:
    if not _repo_cached(repo_id, "lm_head.pth"):
        pytest.skip(
            f"Requires {repo_id} (lm_head.pth) cached locally; run "
            f"`huggingface-cli download {repo_id}` first."
        )
    if not _cuda_available():
        pytest.skip("Requires CUDA to load Mistral-7B in a reasonable time.")

    device = torch.device("cuda:0")
    model, tokenizer = load_official_lens(
        repo_id,
        dtype=torch.float16,
        device=device,
        attn_implementation="sdpa",  # FA2 may not be installed
    )
    try:
        assert model.lm_head.out_features == expected_dim
        assert tokenizer.pad_token is not None
        # Smoke: one query forward.
        inp = tokenizer(
            "<instruct>Test instruction.\n<query>hello world",
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)
        with torch.no_grad():
            out = model(**inp, return_dict=True)
        assert out.logits.shape[-1] == expected_dim
    finally:
        del model
        torch.cuda.empty_cache()
