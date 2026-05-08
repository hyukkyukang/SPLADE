"""Bidirectional Mistral backbone for LENS-style sparse encoders.

In transformers >= 4.50, ``MistralModel.forward`` no longer calls the instance
``_update_causal_mask`` method. Instead it dispatches to the top-level
``create_causal_mask`` / ``create_sliding_window_causal_mask`` helpers in
``transformers.masking_utils``, which inspect ``config.is_causal``: setting
it to ``False`` makes those functions return a bidirectional mask
(padding-aware, non-triangular). We additionally clear each attention
module's ``self.is_causal`` flag (hardcoded to True in
``MistralAttention.__init__``) so SDPA's ``is_causal`` short-circuit and FA2
both observe the bidirectional intent.

See ``feedback_bidirectional_mistral_fix.md`` in auto-memory for the
diagnostic story and the dead-code purge that this file represents.
"""

import copy
from typing import Any

from torch import nn
from transformers import MistralForCausalLM, MistralModel, MistralPreTrainedModel


class MistralBiModel(MistralModel):
    """Mistral decoder backbone with bidirectional attention for LENS."""

    def __init__(self, config: Any) -> None:
        # Defensive copy — `config` is sometimes shared across sibling models
        # that should remain causal; mutating it in place would silently flip
        # them to bidirectional too.
        config = copy.deepcopy(config)
        config.is_causal = False
        super().__init__(config)
        for layer in self.layers:
            self_attn: Any = getattr(layer, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "is_causal"):
                self_attn.is_causal = False


class MistralBiForCausalLM(MistralForCausalLM):
    """Bidirectional MistralForCausalLM used by LENS-style sparse encoders."""

    def __init__(self, config: Any) -> None:
        MistralPreTrainedModel.__init__(self, config)
        self.model = MistralBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
