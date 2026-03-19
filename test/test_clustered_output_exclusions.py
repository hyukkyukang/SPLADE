import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from src.model.retriever.sparse.neural.splade import SpladeEncoder


class _DummyHiddenOutput:
    def __init__(self, hidden_states: torch.Tensor) -> None:
        self.last_hidden_state: torch.Tensor = hidden_states


class _DummyHiddenModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size: int = int(hidden_size)
        self.config = types.SimpleNamespace(use_cache=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
        use_cache: bool = False,
    ) -> _DummyHiddenOutput:
        _ = attention_mask, return_dict, use_cache
        batch_size: int = int(input_ids.shape[0])
        seq_len: int = int(input_ids.shape[1])
        hidden = torch.ones((batch_size, seq_len, self.hidden_size), dtype=torch.float32)
        return _DummyHiddenOutput(hidden)


class _DummyModelOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits: torch.Tensor = logits


class _DummyCausalLM(nn.Module):
    def __init__(self, *, vocab_size: int = 32, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            vocab_size=int(vocab_size),
            use_cache=False,
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            bos_token_id=3,
            eos_token_id=4,
        )
        self.model = _DummyHiddenModel(hidden_size=hidden_size)
        self.lm_head = nn.Linear(hidden_size, int(vocab_size), bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
    ) -> _DummyModelOutput:
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=use_cache,
        ).last_hidden_state
        return _DummyModelOutput(self.lm_head(hidden))

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, module: nn.Module | None) -> None:
        self.lm_head = nn.Identity() if module is None else module


def _build_compact_payload(
    *,
    out_features: int,
    hidden_size: int,
    alignment: str,
    token_ids: list[int] | None = None,
) -> dict[str, torch.Tensor | list[int] | str]:
    payload: dict[str, torch.Tensor | list[int] | str] = {
        "alignment": alignment,
        "weight": torch.ones((out_features, hidden_size), dtype=torch.float32),
        "bias": torch.zeros((out_features,), dtype=torch.float32),
    }
    if token_ids is not None:
        payload["token_ids"] = list(token_ids)
    return payload


class ClusteredOutputExclusionsTest(unittest.TestCase):
    def test_token_aligned_compact_head_filters_invalid_token_ids(self) -> None:
        dummy_model = _DummyCausalLM(vocab_size=64, hidden_size=8)
        with tempfile.TemporaryDirectory(prefix="clustered_exclude_token_aligned_") as tmp:
            model_dir = Path(tmp)
            torch.save(
                _build_compact_payload(
                    out_features=3,
                    hidden_size=8,
                    alignment="token_ids",
                    token_ids=[10, 20, 30],
                ),
                model_dir / "splade_compact_head.pt",
            )

            with patch(
                "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
                return_value=dummy_model,
            ), patch(
                "src.model.retriever.sparse.neural.splade.resolve_model_name_or_path",
                return_value=str(model_dir),
            ):
                encoder = SpladeEncoder(
                    model_name=str(model_dir),
                    sparse_activation="log1p_relu",
                    huggingface_model_class="AutoModelForCausalLM",
                )

        resolved_output_ids = encoder.resolve_output_exclude_ids(
            torch.tensor([20, -1, 999, 15], dtype=torch.long)
        )
        self.assertEqual(tuple(resolved_output_ids.tolist()), (1,))

    def test_latent_cluster_head_returns_no_output_exclusions(self) -> None:
        dummy_model = _DummyCausalLM(vocab_size=64, hidden_size=8)
        with tempfile.TemporaryDirectory(prefix="clustered_exclude_latent_") as tmp:
            model_dir = Path(tmp)
            torch.save(
                _build_compact_payload(
                    out_features=4,
                    hidden_size=8,
                    alignment="latent_cluster",
                ),
                model_dir / "splade_compact_head.pt",
            )

            with patch(
                "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
                return_value=dummy_model,
            ), patch(
                "src.model.retriever.sparse.neural.splade.resolve_model_name_or_path",
                return_value=str(model_dir),
            ):
                encoder = SpladeEncoder(
                    model_name=str(model_dir),
                    sparse_activation="log1p_relu",
                    huggingface_model_class="AutoModelForCausalLM",
                )

        resolved_output_ids = encoder.resolve_output_exclude_ids(
            torch.tensor([0, 1, 2], dtype=torch.long)
        )
        self.assertEqual(int(resolved_output_ids.numel()), 0)


if __name__ == "__main__":
    unittest.main()
