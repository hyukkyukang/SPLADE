"""Reference implementation of the official LENS inference path.

The public function :func:`load_official_lens` mirrors the two-step model
initialization used by the official Yibin-Lei/LENS `eval/model.py`:

1. ``MistralBiForCausalLM.from_pretrained(repo_id, ignore_mismatched_sizes=True)``
2. ``model.lm_head = torch.load('lm_head.pth')``

It returns the bare Mistral model + tokenizer tuple, for cases where we want
ground-truth embeddings independent of our SpladeEncoder wrapper (e.g. parity
tests). Production inference should instead go through the SpladeEncoder
config (`model.load_official_lens=true`), which picks up the same artifacts
via :func:`src.model.retriever.sparse.neural.splade._resolve_compact_head_path`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.model.retriever.sparse.neural.bidirectional_mistral import (
    MistralBiForCausalLM,
)
from src.utils.compact_head import OFFICIAL_LENS_HEAD_FILENAME
from src.utils.transformers import build_tokenizer


_OFFICIAL_LENS_REPOS: frozenset[str] = frozenset(
    {"yibinlei/LENS-d4000", "yibinlei/LENS-d8000"}
)
_LENS_BASE_TOKENIZER: str = "mistralai/Mistral-7B-v0.1"


def build_official_lens_tokenizer(
    *, local_files_only: bool = True
) -> PreTrainedTokenizerBase:
    """Single source of truth for the LENS tokenizer.

    Mirrors official Yibin-Lei/LENS ``eval/model.py:62-72``: base Mistral
    tokenizer, slow SentencePiece, left-padding, ``unk -> eos`` pad fallback,
    with the 3 LENS specials (``<instruct>``, ``<query>``, ``<response>``)
    added. Used by both inference (via :func:`load_official_lens`) and
    training (via :func:`src.data.pl_module.common.build_model_tokenizer`)
    so they cannot drift.

    Why not use the per-model HF tokenizer? The ``yibinlei/LENS-d{4000,8000}``
    HF repos ship a ``tokenizer.json`` whose ``TemplateProcessing``
    post-processor auto-appends ``</s>``. Combined with the encoder-side
    ``"</s>"`` string suffix, that produces TWO trailing EOS tokens — and the
    pool window ``mask[<query>:-2]`` then keeps the last real query token
    instead of dropping it, shifting embeddings vs. what the model was
    trained to produce.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        _LENS_BASE_TOKENIZER,
        use_fast=False,
        local_files_only=local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<instruct>", "<query>", "<response>"]}
    )
    return tokenizer


def _resolve_lm_head_path(source: str, local_files_only: bool = True) -> Path:
    """Return a local path to the `lm_head.pth` artifact.

    Accepts either a local directory containing `lm_head.pth` or a
    Hugging Face repo id, in which case the file is downloaded via
    ``hf_hub_download`` into the active HF cache (or fetched from cache
    when ``local_files_only`` is True).
    """
    local_path: Path = Path(source).expanduser()
    if local_path.is_dir():
        candidate: Path = local_path / OFFICIAL_LENS_HEAD_FILENAME
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"Official LENS lm_head.pth not found under local model dir: {local_path}"
        )
    from huggingface_hub import hf_hub_download

    downloaded: str = hf_hub_download(
        repo_id=str(source).strip(),
        filename=OFFICIAL_LENS_HEAD_FILENAME,
        local_files_only=bool(local_files_only),
    )
    return Path(downloaded)


def load_official_lens(
    source: str,
    *,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None,
    attn_implementation: str = "flash_attention_2",
    strict_official_tokenizer: bool = True,
    local_files_only: bool = True,
) -> Tuple[MistralBiForCausalLM, PreTrainedTokenizerBase]:
    """Load a LENS-d{4000,8000}-style checkpoint matching the official inference flow.

    Arguments
    ---------
    source : str
        HF repo id (``yibinlei/LENS-d4000`` / ``yibinlei/LENS-d8000``) or a local
        directory that mirrors the same layout.
    dtype : torch.dtype
        Parameter dtype for the Mistral backbone and the replaced lm_head.
    device : torch.device | str | None
        Target device. If ``None`` the caller is expected to move the model.
    attn_implementation : str
        ``"flash_attention_2"`` (paper-mandated, requires `flash-attn` installed)
        or ``"sdpa"`` as a fallback when FA2 is not available.
    strict_official_tokenizer : bool
        Forwarded to :func:`build_tokenizer` — reproduces the slow, left-pad,
        ``add_eos_token=True`` tokenizer used by the authors.

    Returns
    -------
    (model, tokenizer)
        ``model.lm_head`` is the shipped clustered head (4000 or 8000 rows).
        ``tokenizer`` has ``<instruct>``, ``<query>``, ``<response>`` registered.
    """
    model: MistralBiForCausalLM = MistralBiForCausalLM.from_pretrained(
        source,
        ignore_mismatched_sizes=True,
        dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=bool(local_files_only),
    )
    # The repo's config advertises ``vocab_size`` = 4000 / 8000 which mismatches
    # the on-disk ``lm_head`` tensor produced by ``ignore_mismatched_sizes``.
    # Replacing the head with the shipped artifact restores the clustered head.
    lm_head_path: Path = _resolve_lm_head_path(source, local_files_only=local_files_only)
    loaded: object = torch.load(
        str(lm_head_path), map_location="cpu", weights_only=False
    )
    if isinstance(loaded, nn.Linear):
        replacement: nn.Linear = loaded
    elif isinstance(loaded, torch.Tensor) and loaded.ndim == 2:
        replacement = nn.Linear(
            in_features=int(loaded.shape[1]),
            out_features=int(loaded.shape[0]),
            bias=False,
        )
        replacement.weight.data.copy_(loaded)
    else:
        raise ValueError(
            f"Unsupported lm_head artifact type at {lm_head_path}: {type(loaded)!r}"
        )
    replacement = replacement.to(dtype=dtype)
    model.lm_head = replacement
    model.config.vocab_size = int(replacement.out_features)
    if device is not None:
        model = model.to(device=torch.device(device))
    model.eval()

    # See ``build_official_lens_tokenizer`` for why this path uses the base
    # Mistral tokenizer rather than the per-model HF tokenizer.
    if bool(strict_official_tokenizer):
        tokenizer = build_official_lens_tokenizer(
            local_files_only=bool(local_files_only),
        )
    else:
        tokenizer = build_tokenizer(
            source,
            strict_official_lens_tokenizer=False,
            local_files_only=bool(local_files_only),
        )
    return model, tokenizer


def is_official_lens_repo(source: str) -> bool:
    """Return True if the source string matches a known official LENS repo."""
    return str(source).strip() in _OFFICIAL_LENS_REPOS


__all__ = [
    "load_official_lens",
    "is_official_lens_repo",
    "build_official_lens_tokenizer",
]
