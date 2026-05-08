from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

COMPACT_HEAD_FILENAME: str = "splade_compact_head.pt"
OFFICIAL_LENS_HEAD_FILENAME: str = "lm_head.pth"
# Filenames probed (in order) when swapping a backbone's ``lm_head`` with a
# clustered head shipped alongside an artifact directory. Keep
# ``splade_compact_head.pt`` first: that's the format produced by our own
# cluster-artifact build, and the official ``lm_head.pth`` is only present in
# Yibin-Lei/LENS-d{4000,8000} repos.
_CLUSTERED_HEAD_CANDIDATE_FILENAMES: tuple[str, ...] = (
    COMPACT_HEAD_FILENAME,
    OFFICIAL_LENS_HEAD_FILENAME,
)
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


def _normalize_token_ids(raw_token_ids: Any) -> list[int] | None:
    if raw_token_ids is None:
        return None
    if isinstance(raw_token_ids, torch.Tensor):
        return [
            int(token_id)
            for token_id in raw_token_ids.detach().cpu().flatten().tolist()
        ]
    if isinstance(raw_token_ids, list):
        return [int(token_id) for token_id in raw_token_ids]
    return None


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


def _normalize_payload_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_weight: Any = payload.get("weight")
    if not isinstance(raw_weight, torch.Tensor):
        raise ValueError("Compact-head payload is missing a tensor `weight`.")
    raw_bias: Any = payload.get("bias")
    bias_tensor: torch.Tensor | None = raw_bias if isinstance(raw_bias, torch.Tensor) else None
    token_ids: list[int] | None = _normalize_token_ids(payload.get("token_ids"))
    raw_alignment: Any = payload.get("alignment")
    normalized_alignment: str = (
        str(raw_alignment).strip()
        if raw_alignment is not None and str(raw_alignment).strip()
        else ("token_ids" if token_ids is not None else "latent_cluster")
    )
    extra_metadata: dict[str, Any] = {
        str(key): value
        for key, value in payload.items()
        if str(key) not in _RESERVED_PAYLOAD_KEYS
    }
    if normalized_alignment == "token_ids":
        if token_ids is None:
            raise ValueError(
                "Token-aligned compact-head payloads must include `token_ids`."
            )
        return build_token_aligned_compact_head_payload(
            weight=raw_weight,
            bias=bias_tensor,
            token_ids=token_ids,
            extra_metadata=extra_metadata,
        )
    return build_clustered_compact_head_payload(
        weight=raw_weight,
        bias=bias_tensor,
        extra_metadata=extra_metadata,
    )


def load_compact_head_payload(raw_payload: Any) -> dict[str, Any]:
    """Normalize supported compact-head formats into the repo payload schema."""
    if isinstance(raw_payload, nn.Linear):
        return build_clustered_compact_head_payload(
            weight=raw_payload.weight,
            bias=raw_payload.bias,
            extra_metadata={"source_format": "official_lens_linear"},
        )
    if isinstance(raw_payload, Mapping):
        return _normalize_payload_mapping(raw_payload)
    raise ValueError(
        "Unsupported compact-head artifact payload. Expected a mapping or nn.Linear."
    )


def torch_load_compact_head_artifact(path: str | Path) -> Any:
    artifact_path: str = str(path)
    try:
        return torch.load(artifact_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(artifact_path, map_location="cpu")


def find_clustered_head_artifact(source: str | Path) -> Path | None:
    """Return the first clustered-head artifact found under ``source``.

    Probes ``splade_compact_head.pt`` then ``lm_head.pth``; returns ``None``
    if ``source`` is not a directory or contains neither filename.
    """
    src_dir: Path = Path(source).expanduser()
    if not src_dir.is_dir():
        return None
    for name in _CLUSTERED_HEAD_CANDIDATE_FILENAMES:
        candidate: Path = src_dir / name
        if candidate.is_file():
            return candidate
    return None


def swap_clustered_head_from_artifact(
    model: nn.Module,
    source: str | Path,
) -> bool:
    """Replace ``model.lm_head`` with the clustered head shipped at ``source``.

    Mirrors the official LENS finetune flow (``finetune/load_model.py:82-95``):
    swap the standard-vocab ``lm_head`` for a much smaller ``nn.Linear`` whose
    rows are cluster centroids, BEFORE LoRA wrapping. After this call:

      - ``model.lm_head`` is a ``nn.Linear(hidden_size, cluster_count, bias=...)``
        whose dtype/device match the original head.

    NOTE: ``model.config.vocab_size`` is intentionally NOT updated. The model's
    input embedding (``model.embed_tokens``) still has ``[orig_vocab_size, hidden_size]``,
    and ``config.vocab_size`` must remain consistent with that shape so a subsequent
    ``from_pretrained()`` round-trip can rebuild the correct embedding. Updating
    ``config.vocab_size`` would cause ``from_pretrained()`` to build ``embed_tokens``
    of size ``cluster_count`` and silently truncate input IDs, triggering CUDA
    out-of-bounds asserts at inference (any tokenizer ID >= cluster_count).

    Returns ``True`` if a swap occurred, ``False`` if no clustered-head
    artifact was found under ``source``.

    The function is idempotent in the sense that it only reads from ``source``
    and writes to ``model``; calling it twice replaces the head twice.
    """
    artifact_path: Path | None = find_clustered_head_artifact(source)
    if artifact_path is None:
        return False
    payload: dict[str, Any] = load_compact_head_payload(
        torch_load_compact_head_artifact(artifact_path)
    )
    weight: torch.Tensor = payload["weight"]
    if weight.ndim != 2:
        raise ValueError(
            f"Clustered head weight at {artifact_path} must be rank-2; "
            f"got shape {tuple(weight.shape)!r}."
        )
    bias_raw: Any = payload.get("bias")
    bias: torch.Tensor | None = bias_raw if isinstance(bias_raw, torch.Tensor) else None
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])

    old_head: Any = getattr(model, "lm_head", None)
    if old_head is None or not hasattr(old_head, "weight"):
        raise AttributeError(
            "Model has no `lm_head.weight`; cannot determine target dtype/device."
        )
    expected_in: int = int(getattr(model.config, "hidden_size", in_features))
    if in_features != expected_in:
        raise ValueError(
            f"Clustered head expects in_features={in_features} but model "
            f"hidden_size={expected_in}. Artifact at {artifact_path} is for a "
            "different backbone."
        )
    target_dtype: torch.dtype = old_head.weight.dtype
    target_device: torch.device = old_head.weight.device

    new_head: nn.Linear = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias is not None,
    )
    new_head.weight.data.copy_(weight.to(dtype=target_dtype))
    if bias is not None and new_head.bias is not None:
        new_head.bias.data.copy_(bias.to(dtype=target_dtype))
    new_head = new_head.to(device=target_device, dtype=target_dtype)
    model.lm_head = new_head
    return True


__all__: list[str] = [
    "COMPACT_HEAD_FILENAME",
    "OFFICIAL_LENS_HEAD_FILENAME",
    "build_clustered_compact_head_payload",
    "build_token_aligned_compact_head_payload",
    "find_clustered_head_artifact",
    "load_compact_head_payload",
    "swap_clustered_head_from_artifact",
    "torch_load_compact_head_artifact",
]
