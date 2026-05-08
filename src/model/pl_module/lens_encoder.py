"""LENS training-time bi-encoder.

Differs from the inference-only :class:`src.utils.lens_mteb_encoder.LENSMTEBEncoder`
in three ways that matter for training:

1. Accepts the LENS collator's *list of sub-batch dicts* directly, running the
   Mistral forward once per sub-batch and concatenating reps inside the graph
   so autograd flows through.
2. Owns the PEFT (LoRA) wrapping so training can update adapter weights.
3. Keeps the normalize / temperature / pooling knobs as nn.Module state for
   Lightning compatibility.

The ``encode`` method never chains a no-grad guard: gradient flow is required
on every sub-batch so the effective-batch size from cross-device
``all_gather`` remains trainable end-to-end.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.retriever.sparse.neural.bidirectional_mistral import (
    MistralBiForCausalLM,
)


def _pooling_log1p_relu_max(logits: Tensor, pooling_mask: Tensor) -> Tensor:
    activated: Tensor = torch.log1p(torch.relu(logits))
    masked: Tensor = activated * pooling_mask.unsqueeze(-1).to(dtype=activated.dtype)
    return torch.max(masked, dim=1).values


class LENSBiEncoder(nn.Module):
    """Bi-encoder for LENS training.

    Parameters
    ----------
    model
        A :class:`MistralBiForCausalLM` (or any causal-LM with a ``.lm_head``
        attribute) whose output space is the clustered sparse head.
    normalize : bool
        L2-normalize each representation before returning.
    """

    def __init__(
        self,
        model: MistralBiForCausalLM | nn.Module,
        *,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalize: bool = bool(normalize)

    def _encode_sub_batch(self, features: Mapping[str, Tensor]) -> Tensor:
        input_ids: Tensor = features["input_ids"]
        attention_mask: Tensor = features["attention_mask"]
        pooling_mask: Tensor = features["pooling_mask"]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits: Tensor = outputs.logits
        rep: Tensor = _pooling_log1p_relu_max(logits, pooling_mask)
        return rep

    def encode(
        self,
        features: Iterable[Mapping[str, Tensor]] | Mapping[str, Tensor],
    ) -> Tensor:
        """Run the model on a list of sub-batches and concatenate reps.

        ``features`` may be one sub-batch dict or an iterable of them (the
        output shape matches :class:`src.data.lens_collator.LENSCollator`).
        """
        if isinstance(features, Mapping):
            rep: Tensor = self._encode_sub_batch(features)
        else:
            reps: List[Tensor] = [self._encode_sub_batch(sub) for sub in features]
            if not reps:
                raise ValueError("LENSBiEncoder.encode received no sub-batches.")
            rep = torch.cat(reps, dim=0)
        if self.normalize:
            rep = F.normalize(rep, p=2, dim=-1)
        return rep

    # --- Convenience for eval / debugging --------------------------------------

    @torch.no_grad()
    def encode_inference(
        self,
        features: Iterable[Mapping[str, Tensor]] | Mapping[str, Tensor],
    ) -> Tensor:
        return self.encode(features)


__all__ = ["LENSBiEncoder"]
