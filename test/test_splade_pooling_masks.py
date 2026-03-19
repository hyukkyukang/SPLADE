import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.model.retriever.sparse.neural.splade import SpladeModel


class _DummyHiddenOutput:
    def __init__(self, hidden_states: torch.Tensor) -> None:
        self.last_hidden_state: torch.Tensor = hidden_states


class _DummyHiddenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(use_cache=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
        use_cache: bool = False,
    ) -> _DummyHiddenOutput:
        _ = attention_mask, return_dict, use_cache
        hidden_states: torch.Tensor = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        return _DummyHiddenOutput(hidden_states)


class _DummyModelOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits: torch.Tensor = logits


class _DummyCausalLM(nn.Module):
    def __init__(self, *, vocab_size: int = 16) -> None:
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
        self.model = _DummyHiddenModel()
        self.lm_head = nn.Linear(1, 1, bias=False)
        self.lm_head.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
    ) -> _DummyModelOutput:
        hidden_states: torch.Tensor = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=use_cache,
        ).last_hidden_state
        return _DummyModelOutput(self.lm_head(hidden_states))

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, module: nn.Module | None) -> None:
        self.lm_head = nn.Identity() if module is None else module


class SpladePoolingMaskTest(unittest.TestCase):
    def test_query_pooling_mask_changes_mlm_pooling(self) -> None:
        dummy_model = _DummyCausalLM()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=dummy_model,
        ):
            model = SpladeModel(
                family="lens",
                model_name="dummy",
                huggingface_model_class="AutoModelForCausalLM",
                query_pooling="max",
                doc_pooling="max",
                sparse_activation="log1p_relu",
            )

        input_ids = torch.tensor([[1, 5, 2]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        pooling_mask = torch.tensor([[1, 0, 1]], dtype=torch.long)

        full = model.encode_queries(input_ids, attention_mask)
        masked = model.encode_queries(
            input_ids,
            attention_mask,
            pooling_mask=pooling_mask,
        )

        self.assertEqual(tuple(full.shape), (1, 1))
        self.assertTrue(torch.allclose(full, torch.log1p(torch.tensor([[5.0]]))))
        self.assertTrue(torch.allclose(masked, torch.log1p(torch.tensor([[2.0]]))))

    def test_query_pooling_mask_changes_doc_only_bow(self) -> None:
        dummy_model = _DummyCausalLM(vocab_size=16)
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=dummy_model,
        ):
            model = SpladeModel(
                family="splade",
                model_name="dummy",
                huggingface_model_class="AutoModelForCausalLM",
                query_pooling="max",
                doc_pooling="max",
                sparse_activation="log1p_relu",
                doc_only=True,
            )

        input_ids = torch.tensor([[1, 5, 2]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        pooling_mask = torch.tensor([[1, 0, 1]], dtype=torch.long)

        reps = model.encode_queries(
            input_ids,
            attention_mask,
            pooling_mask=pooling_mask,
        )

        self.assertEqual(float(reps[0, 1].item()), 0.0)
        self.assertEqual(float(reps[0, 2].item()), 0.0)
        self.assertEqual(float(reps[0, 5].item()), 0.0)
        self.assertEqual(float(reps.sum().item()), 0.0)

        no_special_ids = torch.tensor([[6, 5, 7]], dtype=torch.long)
        bow = model.encode_queries(
            no_special_ids,
            attention_mask,
            pooling_mask=pooling_mask,
        )
        self.assertEqual(float(bow[0, 6].item()), 1.0)
        self.assertEqual(float(bow[0, 5].item()), 0.0)
        self.assertEqual(float(bow[0, 7].item()), 1.0)


if __name__ == "__main__":
    unittest.main()
