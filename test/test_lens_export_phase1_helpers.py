"""Unit tests for the inference helpers in
``script.etc.lens_export_phase1_ckpt``.

The full export flow loads a 15 GB Lightning checkpoint, so we don't
exercise it end-to-end here. Instead we feed synthetic LoRA-style
state-dicts that mimic the PEFT layout to the inference helpers and check
they extract rank / target modules correctly, fail loud on conflicts, and
produce the right ``alpha_was_inferred`` flag.
"""

from __future__ import annotations

import pytest
import torch

from script.etc.lens_export_phase1_ckpt import (
    _ALLOWED_MISSING_KEYS,
    _FALLBACK_LORA_TARGET_MODULES,
    _check_state_dict_load,
    _infer_lora_rank,
    _infer_lora_target_modules,
    _resolve_lora_hyperparams,
)


def _make_fake_lora_state_dict(
    *,
    rank: int = 32,
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj"),
    num_layers: int = 4,
    hidden_size: int = 4096,
) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for layer in range(num_layers):
        for module in target_modules:
            base = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            sd[f"{base}.base_layer.weight"] = torch.zeros(hidden_size, hidden_size)
            sd[f"{base}.lora_A.default.weight"] = torch.zeros(rank, hidden_size)
            sd[f"{base}.lora_B.default.weight"] = torch.zeros(hidden_size, rank)
    return sd


def test_infer_rank_returns_unique_value() -> None:
    sd = _make_fake_lora_state_dict(rank=32)
    assert _infer_lora_rank(sd) == 32


def test_infer_rank_rejects_inconsistent() -> None:
    sd = _make_fake_lora_state_dict(rank=32)
    # Inject a mismatched-rank tensor.
    sd["base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight"] = (
        torch.zeros(64, 4096)
    )
    with pytest.raises(ValueError, match="inconsistent ranks"):
        _infer_lora_rank(sd)


def test_infer_rank_raises_when_no_lora_tensors() -> None:
    sd = {"foo.weight": torch.zeros(3, 3)}
    with pytest.raises(ValueError, match="No `lora_A.default.weight` tensors"):
        _infer_lora_rank(sd)


def test_infer_target_modules_extracts_unique_names() -> None:
    sd = _make_fake_lora_state_dict(
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj")
    )
    targets = _infer_lora_target_modules(sd)
    assert set(targets) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"}


def test_infer_target_modules_falls_back_when_empty() -> None:
    sd = {"foo.weight": torch.zeros(3, 3)}
    assert _infer_lora_target_modules(sd) == _FALLBACK_LORA_TARGET_MODULES


def test_resolve_hyperparams_alpha_inferred() -> None:
    sd = _make_fake_lora_state_dict(rank=64)
    rank, alpha, dropout, alpha_was_inferred = _resolve_lora_hyperparams(
        sd, rank_override=None, alpha_override=None, dropout=0.1,
    )
    assert rank == 64
    assert alpha == 128  # default 2*r
    assert dropout == 0.1
    assert alpha_was_inferred is True


def test_resolve_hyperparams_alpha_explicit() -> None:
    sd = _make_fake_lora_state_dict(rank=32)
    rank, alpha, dropout, alpha_was_inferred = _resolve_lora_hyperparams(
        sd, rank_override=None, alpha_override=128, dropout=0.05,
    )
    assert rank == 32
    assert alpha == 128
    assert dropout == 0.05
    assert alpha_was_inferred is False


def test_resolve_hyperparams_rank_override() -> None:
    sd = _make_fake_lora_state_dict(rank=32)
    rank, _, _, _ = _resolve_lora_hyperparams(
        sd, rank_override=16, alpha_override=None, dropout=0.1,
    )
    assert rank == 16


def test_check_state_dict_load_allows_lm_head_missing() -> None:
    # No real_missing if only the allowed lm_head.weight is missing.
    _check_state_dict_load(missing=list(_ALLOWED_MISSING_KEYS), unexpected=[])


def test_check_state_dict_load_rejects_unexpected_missing() -> None:
    with pytest.raises(RuntimeError, match="Unexpected missing keys"):
        _check_state_dict_load(
            missing=["base_model.model.model.layers.0.self_attn.q_proj.weight"],
            unexpected=[],
        )


def test_check_state_dict_load_rejects_unexpected_keys() -> None:
    with pytest.raises(RuntimeError, match="Unexpected keys"):
        _check_state_dict_load(missing=[], unexpected=["extra.weight"])
