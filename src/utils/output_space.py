from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.utils.normalize import normalize_optional_str

_TOKEN_ALIGNED_VALUES: set[str] = {
    "token",
    "token_id",
    "token_ids",
    "term_token_id",
    "term_token_ids",
}
_CLUSTERED_VALUES: set[str] = {
    "cluster",
    "clusters",
    "latent_cluster",
    "latent_clusters",
    "cluster_centroid",
    "cluster_centroids",
    "unstructured",
}


def normalize_compact_head_alignment(value: Any | None) -> str | None:
    """Normalize compact-head alignment metadata when present."""
    normalized: str | None = normalize_optional_str(value)
    if normalized is None:
        return None
    normalized = normalized.lower().replace("-", "_")
    if normalized in _TOKEN_ALIGNED_VALUES:
        return "token_ids"
    if normalized in _CLUSTERED_VALUES:
        return "latent_cluster"
    raise ValueError(f"Unsupported compact-head alignment: {value!r}")


@dataclass(frozen=True)
class OutputSpaceSpec:
    """Canonical output-space semantics for sparse encoders and artifacts."""

    vocab_size: int
    compact_head_alignment: str
    output_token_aligned: bool

    @classmethod
    def from_alignment(
        cls,
        *,
        vocab_size: int,
        compact_head_alignment: Any | None,
        output_token_aligned: Any | None = None,
        default_alignment: str = "token_ids",
    ) -> "OutputSpaceSpec":
        normalized_alignment: str | None = normalize_compact_head_alignment(
            compact_head_alignment
        )
        if normalized_alignment is None:
            normalized_alignment = (
                default_alignment
                if output_token_aligned is None
                else ("token_ids" if bool(output_token_aligned) else "latent_cluster")
            )
        token_aligned: bool = normalized_alignment == "token_ids"
        if output_token_aligned is not None and bool(output_token_aligned) != token_aligned:
            raise ValueError(
                "compact_head_alignment and output_token_aligned disagree: "
                f"{compact_head_alignment!r} vs {output_token_aligned!r}"
            )
        return cls(
            vocab_size=int(vocab_size),
            compact_head_alignment=normalized_alignment,
            output_token_aligned=token_aligned,
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        vocab_size: int | None = None,
    ) -> "OutputSpaceSpec":
        resolved_vocab_size: Any = metadata.get("vocab_size", vocab_size)
        if resolved_vocab_size is None:
            raise ValueError("Output-space metadata is missing vocab_size.")
        return cls.from_alignment(
            vocab_size=int(resolved_vocab_size),
            compact_head_alignment=metadata.get("compact_head_alignment"),
            output_token_aligned=metadata.get("output_token_aligned"),
            default_alignment="token_ids",
        )

    @classmethod
    def from_encoder(cls, encoder: Any) -> "OutputSpaceSpec":
        return cls.from_alignment(
            vocab_size=int(getattr(encoder, "vocab_size")),
            compact_head_alignment=getattr(encoder, "compact_head_alignment", None),
            output_token_aligned=getattr(encoder, "output_token_aligned", None),
            default_alignment="token_ids",
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "compact_head_alignment": self.compact_head_alignment,
            "output_token_aligned": self.output_token_aligned,
        }

    def resolve_exclude_token_ids(
        self,
        exclude_token_ids: torch.Tensor | Sequence[int] | None,
        *,
        token_id_to_output_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map tokenizer token ids onto output-dimension ids when supported."""
        if exclude_token_ids is None:
            return torch.empty((0,), dtype=torch.long)
        if isinstance(exclude_token_ids, torch.Tensor):
            exclude_ids: torch.Tensor = exclude_token_ids.to(dtype=torch.long).flatten()
        else:
            if not exclude_token_ids:
                return torch.empty((0,), dtype=torch.long)
            exclude_ids = torch.tensor(
                [int(token_id) for token_id in exclude_token_ids],
                dtype=torch.long,
            )
        if int(exclude_ids.numel()) == 0 or not self.output_token_aligned:
            return torch.empty((0,), dtype=torch.long)

        if (
            token_id_to_output_index is not None
            and isinstance(token_id_to_output_index, torch.Tensor)
            and int(token_id_to_output_index.numel()) > 0
        ):
            mapping_device: torch.device = token_id_to_output_index.device
            exclude_ids = exclude_ids.to(device=mapping_device)
            max_token_id: int = int(token_id_to_output_index.shape[0]) - 1
            valid_source_ids: torch.Tensor = exclude_ids[
                (exclude_ids >= 0) & (exclude_ids <= max_token_id)
            ]
            if int(valid_source_ids.numel()) == 0:
                return torch.empty((0,), dtype=torch.long)
            mapped_ids: torch.Tensor = token_id_to_output_index[valid_source_ids]
            mapped_ids = mapped_ids[mapped_ids >= 0]
            if int(mapped_ids.numel()) == 0:
                return torch.empty((0,), dtype=torch.long)
            return torch.unique(mapped_ids.to(dtype=torch.long), sorted=True).cpu()

        valid_output_ids: torch.Tensor = exclude_ids[
            (exclude_ids >= 0) & (exclude_ids < self.vocab_size)
        ]
        if int(valid_output_ids.numel()) == 0:
            return torch.empty((0,), dtype=torch.long)
        return torch.unique(valid_output_ids.to(dtype=torch.long), sorted=True).cpu()


def resolve_model_output_space(model: Any) -> OutputSpaceSpec | None:
    encoder: Any = getattr(model, "encoder", None)
    if encoder is None:
        return None
    output_space: Any = getattr(encoder, "output_space", None)
    if isinstance(output_space, OutputSpaceSpec):
        return output_space
    if not hasattr(encoder, "vocab_size"):
        return None
    return OutputSpaceSpec.from_encoder(encoder)


def resolve_model_output_exclude_ids(
    model: Any,
    exclude_token_ids: Sequence[int],
) -> list[int]:
    if not exclude_token_ids:
        return []
    encoder: Any = getattr(model, "encoder", None)
    resolver: Any = (
        None if encoder is None else getattr(encoder, "resolve_output_exclude_ids", None)
    )
    if callable(resolver):
        resolved_output_ids: torch.Tensor = resolver(
            torch.tensor(list(exclude_token_ids), dtype=torch.long)
        )
        return [int(output_id) for output_id in resolved_output_ids.tolist()]
    output_space: OutputSpaceSpec | None = resolve_model_output_space(model)
    if output_space is None:
        return [int(token_id) for token_id in exclude_token_ids]
    token_id_to_output_index: Any = (
        None if encoder is None else getattr(encoder, "token_id_to_output_index", None)
    )
    resolved_output_ids = output_space.resolve_exclude_token_ids(
        list(exclude_token_ids),
        token_id_to_output_index=(
            token_id_to_output_index
            if isinstance(token_id_to_output_index, torch.Tensor)
            else None
        ),
    )
    return [int(output_id) for output_id in resolved_output_ids.tolist()]


__all__ = [
    "OutputSpaceSpec",
    "normalize_compact_head_alignment",
    "resolve_model_output_exclude_ids",
    "resolve_model_output_space",
]
