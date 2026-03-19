from typing import Any, Mapping

import torch

COMPACT_HEAD_FILENAME: str = "splade_compact_head.pt"
_RESERVED_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "alignment",
        "weight",
        "bias",
        "token_ids",
    }
)


def _copy_weight_tensor(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("Compact-head weight must be a rank-2 tensor.")
    return weight.detach().cpu().contiguous()


def _copy_bias_tensor(
    bias: torch.Tensor | None,
    *,
    out_features: int,
) -> torch.Tensor | None:
    if bias is None:
        return None
    if bias.ndim != 1:
        raise ValueError("Compact-head bias must be a rank-1 tensor.")
    if int(bias.shape[0]) != int(out_features):
        raise ValueError(
            "Compact-head bias length must match the number of output rows."
        )
    return bias.detach().cpu().contiguous()


def _normalize_extra_metadata(
    extra_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if extra_metadata is None:
        return {}
    normalized: dict[str, Any] = {}
    key: str
    value: Any
    for key, value in extra_metadata.items():
        normalized_key: str = str(key)
        if normalized_key in _RESERVED_PAYLOAD_KEYS:
            raise ValueError(
                "extra_metadata may not override reserved compact-head key "
                f"{normalized_key!r}."
            )
        normalized[normalized_key] = value
    return normalized


def build_token_aligned_compact_head_payload(
    *,
    weight: torch.Tensor,
    token_ids: list[int] | torch.Tensor,
    bias: torch.Tensor | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    copied_weight: torch.Tensor = _copy_weight_tensor(weight)
    out_features: int = int(copied_weight.shape[0])
    copied_bias: torch.Tensor | None = _copy_bias_tensor(
        bias,
        out_features=out_features,
    )
    if isinstance(token_ids, torch.Tensor):
        normalized_token_ids: list[int] = [
            int(token_id) for token_id in token_ids.detach().cpu().flatten().tolist()
        ]
    else:
        normalized_token_ids = [int(token_id) for token_id in token_ids]
    if len(normalized_token_ids) != out_features:
        raise ValueError(
            "Token-aligned compact heads must provide one token id per output row."
        )

    payload: dict[str, Any] = {
        "alignment": "token_ids",
        "weight": copied_weight,
        "token_ids": normalized_token_ids,
    }
    if copied_bias is not None:
        payload["bias"] = copied_bias
    payload.update(_normalize_extra_metadata(extra_metadata))
    return payload


def build_clustered_compact_head_payload(
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    copied_weight: torch.Tensor = _copy_weight_tensor(weight)
    copied_bias: torch.Tensor | None = _copy_bias_tensor(
        bias,
        out_features=int(copied_weight.shape[0]),
    )
    payload: dict[str, Any] = {
        "alignment": "latent_cluster",
        "weight": copied_weight,
    }
    if copied_bias is not None:
        payload["bias"] = copied_bias
    payload.update(_normalize_extra_metadata(extra_metadata))
    return payload
