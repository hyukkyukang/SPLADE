import unittest

import torch

from src.utils.windowed_encoding import (
    encode_and_aggregate_windows,
    encode_in_chunks,
)


class WindowedAggregationTest(unittest.TestCase):
    def test_encode_in_chunks_matches_eager_and_marks_steps(self) -> None:
        input_ids = torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        pooling_mask = torch.tensor(
            [
                [1, 0],
                [1, 1],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=torch.long,
        )

        def encode_fn(
            chunk_input_ids: torch.Tensor,
            chunk_attention_mask: torch.Tensor,
            chunk_pooling_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            pooled_mask = (
                torch.zeros_like(chunk_attention_mask)
                if chunk_pooling_mask is None
                else chunk_pooling_mask
            )
            return torch.stack(
                [
                    (chunk_input_ids.float() * chunk_attention_mask.float()).sum(dim=1),
                    pooled_mask.float().sum(dim=1),
                ],
                dim=1,
            )

        expected = encode_fn(input_ids, attention_mask, pooling_mask)
        mark_steps: list[str] = []
        actual = encode_in_chunks(
            input_ids,
            attention_mask,
            pooling_mask,
            encode_fn=encode_fn,
            chunk_size=2,
            mark_step=lambda: mark_steps.append("step"),
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(mark_steps), 3)

    def test_encode_and_aggregate_windows_sums_per_entity(self) -> None:
        input_ids = torch.tensor([[1], [2], [3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        def encode_fn(
            chunk_input_ids: torch.Tensor,
            chunk_attention_mask: torch.Tensor,
            chunk_pooling_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            _ = chunk_attention_mask, chunk_pooling_mask
            values = chunk_input_ids[:, 0].float()
            return torch.stack([values, values.square()], dim=1)

        aggregated = encode_and_aggregate_windows(
            input_ids,
            attention_mask,
            None,
            indptr=torch.tensor([0, 2, 3], dtype=torch.long),
            encode_fn=encode_fn,
            pooling_mode="sum",
            output_dim=2,
            output_dtype=torch.float32,
            pad_token_id=0,
            chunk_size=2,
            use_fixed_size_chunks=True,
            entity_name="document",
        )

        expected = torch.tensor([[3.0, 5.0], [3.0, 9.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(aggregated, expected))

    def test_encode_and_aggregate_windows_zeroes_empty_entities_for_max(self) -> None:
        input_ids = torch.tensor([[1], [4]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        def encode_fn(
            chunk_input_ids: torch.Tensor,
            chunk_attention_mask: torch.Tensor,
            chunk_pooling_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            _ = chunk_attention_mask, chunk_pooling_mask
            values = chunk_input_ids[:, 0].float()
            return torch.stack([values, values * 2.0], dim=1)

        aggregated = encode_and_aggregate_windows(
            input_ids,
            attention_mask,
            None,
            indptr=[0, 1, 1, 2],
            encode_fn=encode_fn,
            pooling_mode="max",
            output_dim=2,
            output_dtype=torch.float32,
            pad_token_id=0,
            chunk_size=2,
            use_fixed_size_chunks=True,
            entity_name="query",
        )

        expected = torch.tensor(
            [
                [1.0, 2.0],
                [0.0, 0.0],
                [4.0, 8.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(aggregated, expected))


if __name__ == "__main__":
    unittest.main()
