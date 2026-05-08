import unittest
from dataclasses import dataclass
from unittest.mock import patch

from src.data.pl_module.common import (
    ContiguousDistributedSampler,
    RowGroupInterleavedDistributedSampler,
    StridedDistributedSampler,
)


@dataclass(frozen=True)
class _Entry:
    start_idx: int
    num_rows: int


class _Dataset:
    def __len__(self) -> int:
        return 40


class EncodeSamplerTest(unittest.TestCase):
    def test_strided_sampler_interleaves_rows(self) -> None:
        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=1),
            patch("torch.distributed.get_world_size", return_value=4),
        ):
            sampler = StridedDistributedSampler(_Dataset())

        self.assertEqual(list(sampler), [1, 5, 9, 13, 17, 21, 25, 29, 33, 37])

    def test_row_group_interleaved_sampler_preserves_group_locality(self) -> None:
        entries = [
            _Entry(start_idx=0, num_rows=4),
            _Entry(start_idx=4, num_rows=4),
            _Entry(start_idx=8, num_rows=4),
            _Entry(start_idx=12, num_rows=4),
        ]
        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=1),
            patch("torch.distributed.get_world_size", return_value=2),
        ):
            sampler = RowGroupInterleavedDistributedSampler(
                _Dataset(),
                row_group_entries=entries,
            )

        self.assertEqual(list(sampler), [4, 5, 6, 7, 12, 13, 14, 15])

    def test_contiguous_sampler_keeps_single_range(self) -> None:
        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=2),
            patch("torch.distributed.get_world_size", return_value=4),
        ):
            sampler = ContiguousDistributedSampler(_Dataset())

        self.assertEqual(list(sampler), list(range(20, 30)))


if __name__ == "__main__":
    unittest.main()
