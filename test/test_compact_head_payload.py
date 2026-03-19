import unittest

import torch

from src.utils.compact_head import (
    build_clustered_compact_head_payload,
    build_token_aligned_compact_head_payload,
)


class CompactHeadPayloadTest(unittest.TestCase):
    def test_build_token_aligned_payload(self) -> None:
        payload = build_token_aligned_compact_head_payload(
            weight=torch.ones((2, 3)),
            bias=torch.zeros((2,)),
            token_ids=[5, 9],
            extra_metadata={"terms": ["foo", "bar"]},
        )

        self.assertEqual(payload["alignment"], "token_ids")
        self.assertEqual(payload["token_ids"], [5, 9])
        self.assertEqual(payload["terms"], ["foo", "bar"])
        self.assertEqual(tuple(payload["weight"].shape), (2, 3))

    def test_build_clustered_payload(self) -> None:
        payload = build_clustered_compact_head_payload(
            weight=torch.ones((4, 3)),
            extra_metadata={"cluster_count": 4},
        )

        self.assertEqual(payload["alignment"], "latent_cluster")
        self.assertEqual(payload["cluster_count"], 4)
        self.assertNotIn("token_ids", payload)
        self.assertEqual(tuple(payload["weight"].shape), (4, 3))

    def test_rejects_reserved_extra_metadata_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved compact-head key"):
            _ = build_clustered_compact_head_payload(
                weight=torch.ones((2, 2)),
                extra_metadata={"alignment": "token_ids"},
            )


if __name__ == "__main__":
    unittest.main()
