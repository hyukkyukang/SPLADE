import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizerBase


def resolve_boundary_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Return known boundary token ids to exclude during term embedding extraction."""
    candidate_ids: list[int | None] = [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
    ]
    token_ids: set[int] = {
        int(token_id)
        for token_id in candidate_ids
        if token_id is not None and int(token_id) >= 0
    }
    return sorted(token_ids)


class EmbeddingGemmaLSRModel(nn.Module):
    """Sparse retriever with a frozen or trainable dense backbone and a target-term head."""

    def __init__(
        self,
        *,
        backbone: PreTrainedModel,
        target_vocab: list[str],
        boundary_token_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        if not target_vocab:
            raise ValueError("target_vocab must not be empty.")

        self.backbone: PreTrainedModel = backbone
        self.target_vocab: list[str] = list(target_vocab)
        self.boundary_token_ids: list[int] = (
            [] if boundary_token_ids is None else sorted({int(x) for x in boundary_token_ids})
        )

        hidden_size: int = self._resolve_hidden_size(self.backbone)
        self.projection: nn.Linear = nn.Linear(hidden_size, len(self.target_vocab), bias=True)

    @staticmethod
    def _resolve_hidden_size(backbone: PreTrainedModel) -> int:
        config: Any = backbone.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "d_model"):
            return int(config.d_model)
        raise ValueError("Unable to resolve hidden size from backbone config.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs: Any = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if not hasattr(outputs, "last_hidden_state"):
            raise ValueError("Backbone outputs must include last_hidden_state.")
        hidden: torch.Tensor = outputs.last_hidden_state
        logits: torch.Tensor = self.projection(hidden)
        token_scores: torch.Tensor = torch.log1p(torch.relu(logits))

        token_mask: torch.Tensor = attention_mask.unsqueeze(-1).to(dtype=torch.bool)
        neg_inf: torch.Tensor = torch.tensor(
            float("-inf"), dtype=token_scores.dtype, device=token_scores.device
        )
        masked: torch.Tensor = token_scores.masked_fill(~token_mask, neg_inf)
        pooled_max: torch.Tensor = masked.max(dim=1).values
        pooled_max = torch.clamp(pooled_max, min=0.0)
        return pooled_max

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids=input_ids, attention_mask=attention_mask)

    @classmethod
    def from_backbone_name(
        cls,
        *,
        backbone_name_or_path: str,
        target_vocab: list[str],
        boundary_token_ids: list[int] | None = None,
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool | None = None,
    ) -> "EmbeddingGemmaLSRModel":
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(trust_remote_code),
        }
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype
        if local_files_only is not None:
            load_kwargs["local_files_only"] = bool(local_files_only)
        backbone: PreTrainedModel = AutoModel.from_pretrained(
            backbone_name_or_path,
            **load_kwargs,
        )
        return cls(
            backbone=backbone,
            target_vocab=target_vocab,
            boundary_token_ids=boundary_token_ids,
        )

    def save_pretrained(
        self,
        output_dir: str | Path,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        destination: Path = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        self.backbone.save_pretrained(destination)
        if tokenizer is not None:
            tokenizer.save_pretrained(destination)

        projection_payload: dict[str, torch.Tensor] = {
            "weight": self.projection.weight.detach().cpu(),
            "bias": self.projection.bias.detach().cpu(),
        }
        torch.save(projection_payload, destination / "lsr_projection.pt")

        payload: dict[str, Any] = {
            "target_vocab": self.target_vocab,
            "boundary_token_ids": self.boundary_token_ids,
            "head_dim": len(self.target_vocab),
            "hidden_size": int(self.projection.in_features),
        }
        if extra_metadata:
            payload["metadata"] = extra_metadata
        (destination / "lsr_config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "EmbeddingGemmaLSRModel":
        model_path: Path = Path(model_dir)
        config_path: Path = model_path / "lsr_config.json"
        projection_path: Path = model_path / "lsr_projection.pt"

        if not config_path.is_file():
            raise FileNotFoundError(f"Missing LSR config file: {config_path}")
        if not projection_path.is_file():
            raise FileNotFoundError(f"Missing LSR projection file: {projection_path}")

        payload: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        target_vocab: list[str] = [str(term) for term in payload["target_vocab"]]
        boundary_token_ids: list[int] = [
            int(token_id) for token_id in payload.get("boundary_token_ids", [])
        ]

        model: EmbeddingGemmaLSRModel = cls.from_backbone_name(
            backbone_name_or_path=str(model_path),
            target_vocab=target_vocab,
            boundary_token_ids=boundary_token_ids,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

        projection_state: dict[str, torch.Tensor] = torch.load(
            projection_path,
            map_location=map_location,
        )
        model.projection.weight.data.copy_(projection_state["weight"].to(model.projection.weight.dtype))
        model.projection.bias.data.copy_(projection_state["bias"].to(model.projection.bias.dtype))
        return model


def discover_fragmented_terms(
    tokenizer: PreTrainedTokenizerBase,
    terms: Iterable[str],
    *,
    threshold: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    fragmented: list[str] = []
    report: list[dict[str, Any]] = []
    term: str
    for term in terms:
        token_ids: list[int] = list(
            tokenizer(term, add_special_tokens=False)["input_ids"]
        )
        subword_count: int = len(token_ids)
        if subword_count >= int(threshold):
            fragmented.append(term)
        report.append(
            {
                "term": term,
                "subword_count": subword_count,
                "token_ids": token_ids,
                "fragmented": bool(subword_count >= int(threshold)),
            }
        )
    return fragmented, report


def build_semantic_projection_initialization(
    *,
    model: EmbeddingGemmaLSRModel,
    tokenizer: PreTrainedTokenizerBase,
    target_vocab: list[str],
    df_map: dict[str, int],
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Create W_proj and b_proj from token-level hidden states.

    Returns:
        weights: [N, d_model]
        biases: [N]
        metadata: per-term extraction diagnostics
    """
    model.eval()
    boundary_ids: list[int] = list(model.boundary_token_ids)
    boundary_tensor: torch.Tensor | None = None
    if boundary_ids:
        boundary_tensor = torch.tensor(boundary_ids, dtype=torch.long, device=device)

    rows: list[torch.Tensor] = []
    biases: list[float] = []
    metadata: list[dict[str, Any]] = []

    with torch.no_grad():
        term: str
        for term in target_vocab:
            encoded: dict[str, torch.Tensor] = tokenizer(
                term,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids: torch.Tensor = encoded["input_ids"].to(device)
            attention_mask: torch.Tensor = encoded["attention_mask"].to(device)

            outputs: Any = model.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden: torch.Tensor = outputs.last_hidden_state[0]

            valid_mask: torch.Tensor = attention_mask[0].to(dtype=torch.bool)
            if boundary_tensor is not None and int(boundary_tensor.numel()) > 0:
                boundary_mask: torch.Tensor = torch.isin(input_ids[0], boundary_tensor)
                valid_mask = valid_mask & ~boundary_mask
            if not bool(valid_mask.any()):
                valid_mask = attention_mask[0].to(dtype=torch.bool)

            selected_vectors: torch.Tensor = hidden[valid_mask]
            term_vector: torch.Tensor = selected_vectors.mean(dim=0)
            norm_before: float = float(
                torch.linalg.vector_norm(term_vector, ord=2).detach().cpu().item()
            )
            safe_norm: torch.Tensor = torch.clamp(
                torch.linalg.vector_norm(term_vector, ord=2),
                min=1e-12,
            )
            normalized: torch.Tensor = term_vector / safe_norm

            rows.append(normalized.detach().cpu())
            df_value: int = int(df_map.get(term, 0))
            bias_value: float = -float(alpha) * math.log(float(df_value) + 1.0)
            biases.append(bias_value)

            token_ids: list[int] = (
                tokenizer(term, add_special_tokens=False)["input_ids"]
            )
            metadata.append(
                {
                    "term": term,
                    "token_ids": [int(token_id) for token_id in token_ids],
                    "subword_count": int(len(token_ids)),
                    "weight_norm_before_l2": norm_before,
                    "bias": bias_value,
                    "df": df_value,
                }
            )

    weight_tensor: torch.Tensor = torch.stack(rows, dim=0)
    bias_tensor: torch.Tensor = torch.tensor(biases, dtype=weight_tensor.dtype)
    return weight_tensor, bias_tensor, metadata


def apply_projection_initialization(
    model: EmbeddingGemmaLSRModel,
    *,
    weights: torch.Tensor,
    biases: torch.Tensor,
) -> None:
    if tuple(weights.shape) != tuple(model.projection.weight.shape):
        raise ValueError(
            "Projection weight shape mismatch: "
            f"expected={tuple(model.projection.weight.shape)}, "
            f"got={tuple(weights.shape)}"
        )
    if tuple(biases.shape) != tuple(model.projection.bias.shape):
        raise ValueError(
            "Projection bias shape mismatch: "
            f"expected={tuple(model.projection.bias.shape)}, "
            f"got={tuple(biases.shape)}"
        )

    model.projection.weight.data.copy_(
        weights.to(device=model.projection.weight.device, dtype=model.projection.weight.dtype)
    )
    model.projection.bias.data.copy_(
        biases.to(device=model.projection.bias.device, dtype=model.projection.bias.dtype)
    )
