import unittest
from types import SimpleNamespace

import torch

from src.model.retriever.sparse.neural.splade import SpladeModel


def _build_fake_model(
    *, vocab_size: int, token_id_to_output_index: torch.Tensor, exclude_ids: torch.Tensor
) -> SimpleNamespace:
    encoder = SimpleNamespace(
        vocab_size=vocab_size,
        mlm=SimpleNamespace(dtype=torch.float32),
        token_id_to_output_index=token_id_to_output_index,
    )
    return SimpleNamespace(
        encoder=encoder,
        _query_exclude_token_ids=exclude_ids,
    )


class DocOnlyQueryEncodingTest(unittest.TestCase):
    def test_compact_query_encoding_ignores_excluded_and_out_of_range_ids(self) -> None:
        model = _build_fake_model(
            vocab_size=3,
            token_id_to_output_index=torch.tensor([-1, 2, -1, 0, 1], dtype=torch.long),
            exclude_ids=torch.tensor([0], dtype=torch.long),
        )
        input_ids = torch.tensor([[0, 3, 4, 9], [1, 4, 7, 3]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 1]], dtype=torch.long)

        bow = SpladeModel._encode_query_terms(model, input_ids, attention_mask)

        expected = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(bow, expected))


if __name__ == "__main__":
    unittest.main()
