"""Regression tests for the bidirectional Mistral subclass.

Pins two invariants that, if broken, silently revert the model to causal
attention (the bug we hunted in April 2026 — ~50% NDCG@10 loss on BEIR):

1. ``config.is_causal`` is ``False`` on the bidirectional model's config so
   that transformers ≥4.50's top-level ``create_causal_mask`` returns a
   bidirectional mask.
2. Mutating ``MistralBiModel(config)`` does NOT propagate back to the
   caller's ``config`` object — sibling models that share the same config
   (e.g. a separate causal LM) must remain causal.
"""

from __future__ import annotations

import copy

from transformers import MistralConfig

from src.model.retriever.sparse.neural.bidirectional_mistral import (
    MistralBiForCausalLM,
    MistralBiModel,
)


def _tiny_mistral_config() -> MistralConfig:
    """Smallest legitimate Mistral config we can instantiate cheaply.

    Keeps construction <1s on CPU; we only inspect attributes, not run
    forward passes.
    """
    return MistralConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        sliding_window=None,
    )


def test_bimodel_sets_is_causal_false_on_inner_config() -> None:
    config = _tiny_mistral_config()
    model = MistralBiModel(config)
    # The model's *own* config (the deepcopy) must say bidirectional.
    assert model.config.is_causal is False
    # Every attention module must also have its instance flag flipped, since
    # `MistralAttention.__init__` hardcodes ``self.is_causal = True``.
    for layer in model.layers:
        assert layer.self_attn.is_causal is False, (
            "Every attention module must have is_causal=False after "
            "MistralBiModel.__init__"
        )


def test_bimodel_does_not_mutate_caller_config() -> None:
    config = _tiny_mistral_config()
    snapshot = copy.deepcopy(config)
    _ = MistralBiModel(config)
    # Caller's config object must be untouched — otherwise sibling causal
    # models built from the same config would silently flip to bidirectional.
    assert not hasattr(config, "is_causal") or config.is_causal is True, (
        "MistralBiModel.__init__ must not mutate the caller's config; "
        "deepcopy isolation regressed."
    )
    # And the relevant attributes the original had should be unchanged.
    for attr in (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
    ):
        assert getattr(config, attr) == getattr(snapshot, attr), (
            f"caller config.{attr} mutated: was {getattr(snapshot, attr)}, "
            f"is {getattr(config, attr)}"
        )


def test_biforcausallm_outer_does_not_inherit_inner_is_causal() -> None:
    """Confirm that the outer LM has its own config (the original, untouched)
    and only ``model.model.config`` carries ``is_causal=False``."""
    config = _tiny_mistral_config()
    bi = MistralBiForCausalLM(config)
    inner_cfg = bi.model.config
    assert inner_cfg.is_causal is False
    # Outer config either lacks the attribute (we never set it) or it's True.
    outer_cfg = bi.config
    assert (
        not hasattr(outer_cfg, "is_causal") or outer_cfg.is_causal is True
    ), "Outer LM config should not be flipped to bidirectional in-place."
