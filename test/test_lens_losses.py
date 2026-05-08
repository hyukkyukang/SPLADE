"""Smoke tests for the LENS-specific losses in src.model.losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model.losses import (
    lens_kl_distill_loss,
    lens_local_contrastive_loss,
    lens_stride_in_batch_contrastive_loss,
)


def test_stride_contrastive_converges_when_positives_aligned() -> None:
    torch.manual_seed(0)
    Q: int = 4
    g: int = 3
    D: int = 8
    # Build a passage matrix where passage[i*g] == query[i] exactly and
    # other passages are random — positives should dominate.
    q = torch.randn(Q, D)
    p = torch.randn(Q * g, D)
    for i in range(Q):
        p[i * g] = q[i]
    loss, scores = lens_stride_in_batch_contrastive_loss(
        q, p, temperature=0.02, train_group_size=g
    )
    assert scores.shape == (Q, Q * g)
    # Positive logit should be the max in each row.
    target = torch.arange(Q) * g
    argmax = scores.argmax(dim=-1)
    assert (argmax == target).all(), (
        f"argmax {argmax.tolist()} should equal positive targets {target.tolist()}."
    )
    assert torch.isfinite(loss)


def test_local_contrastive_target_is_zero_column() -> None:
    torch.manual_seed(0)
    Q, g, D = 3, 4, 6
    q = torch.randn(Q, D)
    p = torch.randn(Q * g, D)
    # Make passage at position 0 in each group match its query.
    for i in range(Q):
        p[i * g] = q[i]
    loss, scores = lens_local_contrastive_loss(
        q, p, temperature=0.02, train_group_size=g
    )
    assert scores.shape == (Q, g)
    argmax = scores.argmax(dim=-1)
    assert (argmax == torch.zeros(Q, dtype=torch.long)).all()
    assert torch.isfinite(loss)


def test_kl_distill_loss_matches_softmax_ce() -> None:
    torch.manual_seed(0)
    Q, g, D = 2, 3, 5
    q = torch.randn(Q, D, requires_grad=True)
    p = torch.randn(Q * g, D, requires_grad=True)
    # Full score matrix (not divided by temperature).
    full = q @ p.T
    teacher = torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]).flatten()
    kl = lens_kl_distill_loss(q, p, full, teacher, train_group_size=g)
    # Hand-compute expected by slicing local-group scores.
    rows = torch.arange(Q).unsqueeze(1)
    cols = (rows * g) + torch.arange(g).unsqueeze(0)
    local = full[rows, cols]  # (Q, g)
    teacher_matrix = teacher.view(Q, g).detach()
    expected = -(F.softmax(teacher_matrix, dim=-1) * F.log_softmax(local, dim=-1)).sum(dim=-1).mean()
    assert torch.allclose(kl, expected, atol=1e-6)
    kl.backward()
    assert q.grad is not None and p.grad is not None
