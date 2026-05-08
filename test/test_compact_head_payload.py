import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.utils.compact_head import (
    OFFICIAL_LENS_HEAD_FILENAME,
    build_clustered_compact_head_payload,
    build_token_aligned_compact_head_payload,
    load_compact_head_payload,
    torch_load_compact_head_artifact,
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

    def test_load_compact_head_payload_accepts_official_linear_module(self) -> None:
        linear = nn.Linear(5, 7, bias=True)

        payload = load_compact_head_payload(linear)

        self.assertEqual(payload["alignment"], "latent_cluster")
        self.assertEqual(payload["source_format"], "official_lens_linear")
        self.assertEqual(tuple(payload["weight"].shape), (7, 5))
        self.assertEqual(tuple(payload["bias"].shape), (7,))

    def test_load_compact_head_payload_accepts_state_dict_like_mapping(self) -> None:
        payload = load_compact_head_payload(
            {
                "weight": torch.ones((3, 4), dtype=torch.float32),
                "bias": torch.zeros((3,), dtype=torch.float32),
            }
        )

        self.assertEqual(payload["alignment"], "latent_cluster")
        self.assertEqual(tuple(payload["weight"].shape), (3, 4))
        self.assertNotIn("token_ids", payload)

    def test_torch_load_compact_head_artifact_supports_pickled_linear(self) -> None:
        with tempfile.TemporaryDirectory(prefix="official_lens_head_") as tmp:
            artifact_path = Path(tmp) / OFFICIAL_LENS_HEAD_FILENAME
            linear = nn.Linear(4, 6, bias=False)
            torch.save(linear, artifact_path)

            loaded = torch_load_compact_head_artifact(artifact_path)

        self.assertIsInstance(loaded, nn.Linear)
        self.assertEqual(tuple(loaded.weight.shape), (6, 4))


if __name__ == "__main__":
    unittest.main()
