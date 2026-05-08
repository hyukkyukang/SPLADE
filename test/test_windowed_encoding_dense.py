import unittest

import torch

from src.utils.windowed_encoding import encode_and_aggregate_windows


class DenseWindowedEncodingTest(unittest.TestCase):
    def test_mean_pooling_aggregates_per_entity(self) -> None:
        input_ids = torch.tensor([[1], [2], [3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        indptr = torch.tensor([0, 2, 3], dtype=torch.long)

        encoded = encode_and_aggregate_windows(
            input_ids,
            attention_mask,
            None,
            indptr=indptr,
            encode_fn=lambda chunk_input_ids, chunk_attention_mask, chunk_pooling_mask: (
                torch.cat(
                    [
                        chunk_input_ids.float(),
                        (chunk_input_ids.float() + 10.0),
                    ],
                    dim=1,
                )
            ),
            pooling_mode="mean",
            output_dim=2,
            output_dtype=torch.float32,
            pad_token_id=0,
            chunk_size=2,
            use_fixed_size_chunks=False,
            entity_name="document",
        )

        expected = torch.tensor([[1.5, 11.5], [3.0, 13.0]])
        self.assertTrue(torch.allclose(encoded, expected))
