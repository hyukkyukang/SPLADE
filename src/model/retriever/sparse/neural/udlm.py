from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig, modeling_outputs


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    residual: torch.Tensor | None,
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out: torch.Tensor = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    residual: torch.Tensor | None,
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    residual: torch.Tensor | None,
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return modulate(x, shift, scale)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb_qkv(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply GPT-NeoX-style rotary embedding to Q/K for packed [B, S, 3, H, D]."""
    if qkv.ndim != 5:
        raise ValueError("UDLM rotary helper expects qkv shape [B, S, 3, H, D].")
    rotary_dim: int = int(cos.shape[-1]) * 2
    if rotary_dim <= 0:
        return qkv
    if rotary_dim > int(qkv.shape[-1]):
        raise ValueError(
            f"Rotary dim {rotary_dim} exceeds head dim {int(qkv.shape[-1])}."
        )
    cos_full: torch.Tensor = cos.repeat_interleave(2, dim=-1).to(
        device=qkv.device,
        dtype=qkv.dtype,
    )[None, :, None, None, :]
    sin_full: torch.Tensor = sin.repeat_interleave(2, dim=-1).to(
        device=qkv.device,
        dtype=qkv.dtype,
    )[None, :, None, None, :]
    qkv_rotated: torch.Tensor = qkv.clone()
    qk_prefix: torch.Tensor = qkv[:, :, :2, :, :rotary_dim]
    qkv_rotated[:, :, :2, :, :rotary_dim] = (
        qk_prefix * cos_full + _rotate_half(qk_prefix) * sin_full
    )
    return qkv_rotated


class UDLMConfig(PretrainedConfig):
    """Local HF config for loading the published UDLM checkpoint without remote code."""

    model_type = "udlm"

    def __init__(
        self,
        vocab_size: int = 30522,
        model_length: int = 128,
        hidden_dim: int = 768,
        cond_dim: int = 128,
        n_blocks: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        time_conditioning: bool = True,
        cfg: bool = False,
        cfg_num_classes: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.model_length = int(model_length)
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.n_blocks = int(n_blocks)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.time_conditioning = bool(time_conditioning)
        self.cfg = bool(cfg)
        self.cfg_num_classes = int(cfg_num_classes)


class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized: torch.Tensor = F.layer_norm(x.float(), [self.dim])
        return normalized.to(dtype=x.dtype) * self.weight[None, None, :]


class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10_000) -> None:
        super().__init__()
        inv_freq: torch.Tensor = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached: int | None = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        seq_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len: int = int(x.shape[seq_dim])
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t: torch.Tensor = torch.arange(seq_len, device=x.device).type_as(
                self.inv_freq
            )
            freqs: torch.Tensor = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb: torch.Tensor = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)
        if self.cos_cached is None or self.sin_cached is None:
            raise RuntimeError("Rotary cache was not initialized.")
        return self.cos_cached, self.sin_cached


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = int(frequency_embedding_size)

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        half: int = dim // 2
        freqs: torch.Tensor = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / max(half, 1)
        )
        args: torch.Tensor = t[:, None].float() * freqs[None]
        embedding: torch.Tensor = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq: torch.Tensor = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, cond_size: int) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, cond_size)
        self.num_classes = int(num_classes)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


def regular_attention_multi_headed(
    qkv: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, seq_len, _, num_heads, head_dim = qkv.shape
    q: torch.Tensor = qkv[:, :, 0, :, :].transpose(1, 2)
    k: torch.Tensor = qkv[:, :, 1, :, :].transpose(1, 2)
    v: torch.Tensor = qkv[:, :, 2, :, :].transpose(1, 2)
    key_padding_mask: torch.Tensor | None = None
    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError("UDLM attention_mask must have shape [B, S].")
        key_padding_mask = attention_mask.to(device=qkv.device, dtype=torch.bool)

    attention_output: torch.Tensor = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=(
            None
            if key_padding_mask is None
            else key_padding_mask[:, None, None, :]
        ),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)
    if key_padding_mask is not None:
        attention_output = attention_output * key_padding_mask[:, :, None, None].to(
            dtype=attention_output.dtype
        )
    return attention_output.reshape(batch_size, seq_len, num_heads * head_dim)


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout = float(dropout)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self) -> Any:
        if self.training:
            return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos_sin: tuple[torch.Tensor, torch.Tensor],
        c: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size: int = int(x.shape[0])
        seq_len: int = int(x.shape[1])
        del batch_size, seq_len
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        x_skip: torch.Tensor = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        qkv: torch.Tensor = self.attn_qkv(x)
        qkv = qkv.view(x.shape[0], x.shape[1], 3, self.n_heads, -1)
        cos, sin = rotary_cos_sin
        qkv = _apply_rotary_emb_qkv(
            qkv,
            cos=cos[0, :, 0, 0, : cos.shape[-1] // 2].to(dtype=qkv.dtype),
            sin=sin[0, :, 0, 0, : sin.shape[-1] // 2].to(dtype=qkv.dtype),
        )
        x = regular_attention_multi_headed(qkv, attention_mask=attention_mask)
        x = bias_dropout_scale_fn(
            self.attn_out(x),
            None,
            gate_msa,
            x_skip,
            self.dropout,
        )
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        return self.linear(x)


class DITBackbone(nn.Module):
    def __init__(self, config: UDLMConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.vocab_embed = EmbeddingLayer(config.hidden_dim, config.vocab_size)
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.cond_map: LabelEmbedder | None
        if config.cfg:
            self.cond_map = LabelEmbedder(config.cfg_num_classes + 1, config.cond_dim)
        else:
            self.cond_map = None
        self.rotary_emb = Rotary(config.hidden_dim // config.n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    config.hidden_dim,
                    config.n_heads,
                    config.cond_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.output_layer = DDitFinalLayer(
            config.hidden_dim,
            config.vocab_size,
            config.cond_dim,
        )

    def forward(
        self,
        indices: torch.Tensor,
        sigma: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        x_emb: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if sigma is None:
            sigma = torch.zeros(
                indices.shape[0],
                device=indices.device,
                dtype=torch.float32,
            )
        if not self.config.time_conditioning:
            sigma = torch.zeros_like(sigma)
        all_hidden_states: list[torch.Tensor] = []

        c: torch.Tensor = F.silu(self.sigma_map(sigma))
        if cond is not None:
            if self.cond_map is None:
                raise ValueError(
                    "Conditioning variable provided, but Model was not initialized "
                    "with condition embedding layer."
                )
            c = c + F.silu(self.cond_map(cond))

        if x_emb is None:
            x: torch.Tensor = self.vocab_embed(indices)
            if output_hidden_states:
                all_hidden_states.append(x)
            rotary_cos_sin = self.rotary_emb(x)
            for block in self.blocks:
                x = block(
                    x,
                    rotary_cos_sin,
                    c,
                    attention_mask=attention_mask,
                )
                if output_hidden_states:
                    all_hidden_states.append(x)
        else:
            x = x_emb
        logits: torch.Tensor = self.output_layer(x, c)
        return logits, all_hidden_states


class UDLMForMaskedLMCompat(PreTrainedModel):
    """Local UDLM loader that removes flash-attn and einops dependencies."""

    config_class = UDLMConfig
    base_model_prefix = "backbone"
    main_input_name = "input_ids"
    all_tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: UDLMConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(config)
        del args, kwargs
        self.backbone = DITBackbone(config)
        self.vocab_size = int(config.vocab_size)

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.vocab_embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.vocab_embed = value  # type: ignore[assignment]

    def get_output_embeddings(self) -> nn.Module:
        return self.backbone.output_layer.linear

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.backbone.output_layer.linear = value  # type: ignore[assignment]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        timesteps: torch.FloatTensor | None = None,
        cond: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> modeling_outputs.MaskedLMOutput | tuple[torch.Tensor, list[torch.Tensor]] | torch.Tensor:
        del use_cache, kwargs
        if input_ids is None:
            raise ValueError("UDLMForMaskedLMCompat requires input_ids.")
        resolved_output_hidden_states: bool = bool(
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        # SPLADE expects .logits by default even if the checkpoint config disables
        # return_dict, so prefer HF-style structured output unless explicitly
        # overridden by the caller.
        resolved_return_dict: bool = True if return_dict is None else bool(return_dict)
        if timesteps is None:
            timesteps = torch.zeros(
                input_ids.shape[0],
                device=input_ids.device,
                dtype=torch.float32,
            )

        logits, all_hidden_states = self.backbone(
            indices=input_ids,
            sigma=timesteps,
            attention_mask=attention_mask,
            cond=cond,
            output_hidden_states=resolved_output_hidden_states,
        )
        if resolved_return_dict:
            return modeling_outputs.MaskedLMOutput(
                logits=logits,
                hidden_states=tuple(all_hidden_states)
                if resolved_output_hidden_states
                else None,
                loss=None,
            )
        if resolved_output_hidden_states:
            return logits, all_hidden_states
        return logits


__all__ = ["UDLMConfig", "UDLMForMaskedLMCompat"]
