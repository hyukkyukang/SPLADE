"""Tests for ``swap_clustered_head_from_artifact``.

Pins the contract relied on by both training (``script/train_lens.py``) and
export (``script/etc/lens_export_phase1_ckpt.py``):

1. Given an artifact dir with ``splade_compact_head.pt``, the helper replaces
   ``model.lm_head`` with a ``nn.Linear(hidden, cluster_count, bias=False)``
   matching the file, leaves ``config.vocab_size`` UNCHANGED (so the input
   embedding's expected size still aligns with the saved tensor), and
   preserves the original head's dtype/device.
2. Given an artifact dir with the official ``lm_head.pth`` (a pickled
   ``nn.Linear``), the helper performs the same swap.
3. Given an artifact dir with neither file, the helper is a no-op and
   returns ``False`` — caller should keep the original head.
4. A clustered-head whose ``in_features`` mismatches the model's
   ``hidden_size`` raises rather than silently mismangling shapes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.utils.compact_head import (
    COMPACT_HEAD_FILENAME,
    OFFICIAL_LENS_HEAD_FILENAME,
    build_clustered_compact_head_payload,
    swap_clustered_head_from_artifact,
)


class _StubConfig:
    def __init__(self, *, hidden_size: int, vocab_size: int) -> None:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class _StubModel(nn.Module):
    """Minimum surface a model needs for the swap helper: ``config`` with
    ``hidden_size``/``vocab_size``, and a ``lm_head`` ``nn.Linear``.
    """

    def __init__(self, *, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.config = _StubConfig(hidden_size=hidden_size, vocab_size=vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size


def test_swap_with_compact_head_dict_payload(tmp_path: Path) -> None:
    cluster_count, hidden = 16, 32
    weight = torch.randn(cluster_count, hidden, dtype=torch.float32)
    payload = build_clustered_compact_head_payload(weight=weight)
    torch.save(payload, tmp_path / COMPACT_HEAD_FILENAME)

    model = _StubModel(hidden_size=hidden, vocab_size=99)
    original_dtype = model.lm_head.weight.dtype

    assert swap_clustered_head_from_artifact(model, tmp_path) is True

    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight.shape == (cluster_count, hidden)
    assert model.lm_head.bias is None
    # config.vocab_size MUST stay aligned with the input embedding's row count
    # (which is unchanged by the swap). See the docstring on
    # swap_clustered_head_from_artifact for why mutating config.vocab_size
    # silently breaks a from_pretrained() round-trip.
    assert model.config.vocab_size == 99
    assert model.vocab_size == 99
    assert model.lm_head.weight.dtype == original_dtype
    # Weights must equal the saved tensor (cast to the original head's dtype).
    torch.testing.assert_close(
        model.lm_head.weight.detach().cpu(),
        weight.to(dtype=original_dtype),
    )


def test_swap_with_official_lm_head_pth_linear(tmp_path: Path) -> None:
    cluster_count, hidden = 12, 32
    head = nn.Linear(hidden, cluster_count, bias=False)
    torch.save(head, tmp_path / OFFICIAL_LENS_HEAD_FILENAME)

    model = _StubModel(hidden_size=hidden, vocab_size=99)

    assert swap_clustered_head_from_artifact(model, tmp_path) is True
    assert model.lm_head.weight.shape == (cluster_count, hidden)
    # vocab_size unchanged on purpose — see helper docstring.
    assert model.config.vocab_size == 99


def test_swap_prefers_compact_head_over_official_when_both_present(
    tmp_path: Path,
) -> None:
    """splade_compact_head.pt should win when both files exist (it's our
    own format and the canonical artifact for the cluster pipeline)."""
    hidden = 32
    compact = build_clustered_compact_head_payload(
        weight=torch.full((4, hidden), fill_value=2.0)
    )
    torch.save(compact, tmp_path / COMPACT_HEAD_FILENAME)
    torch.save(
        nn.Linear(hidden, 7, bias=False),
        tmp_path / OFFICIAL_LENS_HEAD_FILENAME,
    )

    model = _StubModel(hidden_size=hidden, vocab_size=99)
    assert swap_clustered_head_from_artifact(model, tmp_path) is True
    # Cluster_count = 4 from compact_head, NOT 7 from lm_head.pth.
    assert model.lm_head.weight.shape == (4, hidden)


def test_swap_returns_false_when_no_artifact(tmp_path: Path) -> None:
    model = _StubModel(hidden_size=8, vocab_size=99)
    assert swap_clustered_head_from_artifact(model, tmp_path) is False
    # Original head must be untouched.
    assert model.lm_head.weight.shape == (99, 8)
    assert model.config.vocab_size == 99


def test_swap_rejects_hidden_size_mismatch(tmp_path: Path) -> None:
    payload = build_clustered_compact_head_payload(
        weight=torch.randn(4, 16),  # hidden=16
    )
    torch.save(payload, tmp_path / COMPACT_HEAD_FILENAME)

    model = _StubModel(hidden_size=32, vocab_size=99)  # mismatched hidden=32
    with pytest.raises(ValueError, match="hidden_size"):
        swap_clustered_head_from_artifact(model, tmp_path)


def test_swap_preserves_bf16_dtype(tmp_path: Path) -> None:
    """Training builds the backbone in bf16; the swapped head must match."""
    cluster_count, hidden = 8, 16
    payload = build_clustered_compact_head_payload(
        weight=torch.randn(cluster_count, hidden, dtype=torch.float32)
    )
    torch.save(payload, tmp_path / COMPACT_HEAD_FILENAME)

    model = _StubModel(hidden_size=hidden, vocab_size=99)
    model.lm_head = model.lm_head.to(dtype=torch.bfloat16)

    assert swap_clustered_head_from_artifact(model, tmp_path) is True
    assert model.lm_head.weight.dtype == torch.bfloat16
