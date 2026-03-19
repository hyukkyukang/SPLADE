from typing import Any

import torch
from torch import nn
from transformers import MistralForCausalLM, MistralModel, MistralPreTrainedModel
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def _resolve_attn_implementation(config: Any) -> str:
    attn_implementation: Any = getattr(config, "_attn_implementation", None)
    if attn_implementation is None:
        attn_implementation = getattr(config, "attn_implementation", None)
    if attn_implementation is None:
        return "eager"
    normalized: str = str(attn_implementation).strip()
    return normalized or "eager"


def _resolve_target_length(
    *,
    attention_mask: torch.Tensor | None,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor | None,
    past_key_values: Any,
) -> int:
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim >= 2:
        return int(attention_mask.shape[-1])
    if isinstance(cache_position, torch.Tensor) and cache_position.numel() > 0:
        return int(cache_position.max().item()) + 1
    if past_key_values is not None:
        get_max_length: Any = getattr(past_key_values, "get_max_length", None)
        if callable(get_max_length):
            try:
                max_length: Any = get_max_length()
                if max_length is not None:
                    return int(max_length)
            except Exception:
                pass
        get_seq_length: Any = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                seq_length: Any = get_seq_length()
                if seq_length is not None:
                    return int(seq_length) + int(input_tensor.shape[1])
            except Exception:
                pass
    return int(input_tensor.shape[1])


def _maybe_unmask_unattended(
    *,
    attn_implementation: str,
    causal_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    min_dtype: float,
    output_attentions: bool,
) -> torch.Tensor:
    if attn_implementation != "sdpa":
        return causal_mask
    if attention_mask is None or attention_mask.device.type != "cuda":
        return causal_mask
    if output_attentions:
        return causal_mask
    unmask_fn: Any = getattr(AttentionMaskConverter, "_unmask_unattended", None)
    if not callable(unmask_fn):
        return causal_mask
    return unmask_fn(causal_mask, min_dtype)


class MistralBiModel(MistralModel):
    """Mistral decoder backbone with bidirectional attention for LENS."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        layer: nn.Module
        for layer in self.layers:
            self_attn: Any = getattr(layer, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "is_causal"):
                self_attn.is_causal = False

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        del args, kwargs
        attn_implementation: str = _resolve_attn_implementation(self.config)
        if attn_implementation == "flash_attention_2":
            if attention_mask is not None and use_cache:
                if bool((attention_mask[:, -1] == 0).any().item()):
                    raise ValueError(
                        "Batched generation with right padding is unsupported for "
                        "Flash Attention 2 Mistral. Use left padding instead."
                    )
            if attention_mask is not None and bool((attention_mask == 0).any().item()):
                return attention_mask
            return None

        if attention_mask is not None and attention_mask.ndim == 4:
            if attention_mask.max().item() != 0:
                raise ValueError(
                    "Custom 4D attention masks must already be inverted with max==0."
                )
            return attention_mask

        dtype: torch.dtype = input_tensor.dtype
        device: torch.device = input_tensor.device
        min_dtype: float = torch.finfo(dtype).min
        batch_size: int = int(input_tensor.shape[0])
        sequence_length: int = int(input_tensor.shape[1])
        target_length: int = _resolve_target_length(
            attention_mask=attention_mask,
            input_tensor=input_tensor,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        causal_mask: torch.Tensor = torch.zeros(
            (batch_size, 1, sequence_length, target_length),
            dtype=dtype,
            device=device,
        )
        if attention_mask is not None and attention_mask.ndim == 2:
            mask_length: int = min(int(attention_mask.shape[-1]), target_length)
            padding_mask: torch.Tensor = attention_mask[:, None, None, :mask_length] == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask,
                min_dtype,
            )

        return _maybe_unmask_unattended(
            attn_implementation=attn_implementation,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            min_dtype=min_dtype,
            output_attentions=output_attentions,
        )


class MistralBiForCausalLM(MistralForCausalLM):
    """Bidirectional MistralForCausalLM used by LENS-style sparse encoders."""

    def __init__(self, config: Any) -> None:
        MistralPreTrainedModel.__init__(self, config)
        self.model = MistralBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
