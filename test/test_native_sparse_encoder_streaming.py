import types
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch import nn

from import_stubs import install_fake_sentence_transformers

install_fake_sentence_transformers()

from src.utils.sparse_encoder import NativeSparseEncoderAdapter


class _DummyTokenizer:
    def __init__(self) -> None:
        self.all_special_ids: list[int] = [0]
        self._encoded: dict[str, list[int]] = {
            "doc-a": [1, 2],
            "doc-b": [3],
            "doc-c": [4, 5, 6],
        }

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        _ = padding, truncation, return_tensors
        encoded: list[list[int]] = [self._encoded[text][:max_length] for text in texts]
        max_batch_length: int = max(len(item) for item in encoded)
        input_ids = torch.zeros((len(encoded), max_batch_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        row_idx: int
        for row_idx, token_ids in enumerate(encoded):
            token_tensor = torch.tensor(token_ids, dtype=torch.long)
            input_ids[row_idx, : len(token_ids)] = token_tensor
            attention_mask[row_idx, : len(token_ids)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class _DummySparseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = types.SimpleNamespace(
            vocab_size=4,
            dtype=torch.float32,
        )
        self._lookup: dict[int, torch.Tensor] = {
            1: torch.tensor([0.1, 0.9, 0.0, 0.2], dtype=torch.float32),
            3: torch.tensor([0.4, 0.0, 0.3, 0.8], dtype=torch.float32),
            4: torch.tensor([0.0, 0.7, 0.6, 0.0], dtype=torch.float32),
        }

    def eval(self) -> "_DummySparseModel":
        super().eval()
        return self

    def encode_docs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = attention_mask, pooling_mask
        rows: list[torch.Tensor] = []
        first_token_id: torch.Tensor
        for first_token_id in input_ids[:, 0]:
            rows.append(self._lookup[int(first_token_id.item())].to(input_ids.device))
        return torch.stack(rows, dim=0)

    encode_queries = encode_docs


class NativeSparseEncoderStreamingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _DummySparseModel()
        self.adapter = NativeSparseEncoderAdapter(
            model=self.model,
            tokenizer=_DummyTokenizer(),
            model_cfg=OmegaConf.create(
                {
                    "family": "splade",
                    "doc_trim_last_tokens": 0,
                }
            ),
            device=torch.device("cpu"),
            batch_size=2,
            max_query_length=16,
            max_doc_length=16,
        )
        self.expected_dense = torch.tensor(
            [
                [0.0, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.8],
                [0.0, 0.7, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

    def test_dense_batches_are_pruned_before_concat(self) -> None:
        real_cat = torch.cat

        def _checking_cat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
            self.assertEqual(len(tensors), 2)
            tensor: torch.Tensor
            for tensor in tensors:
                self.assertFalse(tensor.is_sparse)
                self.assertEqual(tensor.device.type, "cpu")
                active_counts = (tensor != 0).sum(dim=1)
                self.assertTrue(torch.all(active_counts <= 1))
            return real_cat(tensors, dim=dim)

        with patch("src.utils.sparse_encoder.torch.cat", side_effect=_checking_cat):
            embeddings = self.adapter.encode_document(
                ["doc-a", "doc-b", "doc-c"],
                batch_size=2,
                convert_to_sparse_tensor=False,
                save_to_cpu=True,
                max_active_dims=1,
            )

        self.assertFalse(embeddings.is_sparse)
        self.assertEqual(embeddings.device.type, "cpu")
        self.assertTrue(torch.allclose(embeddings, self.expected_dense))

    def test_sparse_batches_are_pruned_and_sparsified_before_concat(self) -> None:
        real_cat = torch.cat

        def _checking_cat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
            self.assertEqual(len(tensors), 2)
            tensor: torch.Tensor
            for tensor in tensors:
                self.assertTrue(tensor.is_sparse)
                self.assertEqual(tensor.device.type, "cpu")
                active_counts = (tensor.to_dense() != 0).sum(dim=1)
                self.assertTrue(torch.all(active_counts <= 1))
            return real_cat(tensors, dim=dim)

        with patch("src.utils.sparse_encoder.torch.cat", side_effect=_checking_cat):
            embeddings = self.adapter.encode_document(
                ["doc-a", "doc-b", "doc-c"],
                batch_size=2,
                convert_to_sparse_tensor=True,
                save_to_cpu=True,
                max_active_dims=1,
            )

        self.assertTrue(embeddings.is_sparse)
        self.assertEqual(embeddings.device.type, "cpu")
        self.assertTrue(torch.allclose(embeddings.to_dense(), self.expected_dense))


if __name__ == "__main__":
    unittest.main()
