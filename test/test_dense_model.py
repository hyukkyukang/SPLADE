import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.model.retriever.dense.neural.hf_dense import DenseModel
from src.utils.model_utils import load_dense_checkpoint


class _DummyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(hidden_size=2, pad_token_id=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        _ = attention_mask
        hidden_states = torch.stack(
            [
                input_ids.float() * self.scale,
                (input_ids.float() + 1.0) * self.scale,
            ],
            dim=-1,
        )
        return SimpleNamespace(last_hidden_state=hidden_states)


class DenseModelTest(unittest.TestCase):
    def _build_model(self, **kwargs: object) -> DenseModel:
        similarity: str = str(kwargs.pop("similarity", "dot"))
        normalize: bool = bool(kwargs.pop("normalize", False))
        with patch(
            "src.model.retriever.dense.neural.hf_dense.build_pretrained_model",
            return_value=_DummyBackbone(),
        ):
            return DenseModel(
                family="dense",
                model_name="dummy",
                huggingface_model_class="AutoModel",
                query_pooling="mean",
                doc_pooling="cls",
                query_window_pooling="mean",
                doc_window_pooling="mean",
                similarity=similarity,
                normalize=normalize,
                **kwargs,
            )

    def test_dense_model_applies_mean_and_cls_pooling(self) -> None:
        model = self._build_model()
        query_input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
        query_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        query_pooling_mask = torch.tensor([[0, 1, 1], [1, 0, 0]])
        doc_input_ids = torch.tensor([[9, 8, 7]])
        doc_attention_mask = torch.tensor([[1, 1, 1]])

        query_embeddings = model.encode_queries(
            query_input_ids,
            query_attention_mask,
            pooling_mask=query_pooling_mask,
        )
        doc_embeddings = model.encode_docs(doc_input_ids, doc_attention_mask)

        expected_query = torch.tensor([[2.5, 3.5], [4.0, 5.0]])
        expected_doc = torch.tensor([[9.0, 10.0]])
        self.assertTrue(torch.allclose(query_embeddings, expected_query))
        self.assertTrue(torch.allclose(doc_embeddings, expected_doc))

    def test_cosine_similarity_forces_normalization(self) -> None:
        model = self._build_model(similarity="cosine")
        self.assertTrue(model.normalize)
        embeddings = model.encode_queries(
            torch.tensor([[3, 0]]),
            torch.tensor([[1, 0]]),
        )
        norms = torch.linalg.norm(embeddings, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms)))

    def test_load_dense_checkpoint_strips_lightning_prefix(self) -> None:
        model = self._build_model()
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = f"{tmp_dir}/dense.ckpt"
            state_dict = {
                f"model.{name}": tensor.clone()
                for name, tensor in model.state_dict().items()
            }
            torch.save({"state_dict": state_dict}, checkpoint_path)
            missing, unexpected = load_dense_checkpoint(model, checkpoint_path)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
