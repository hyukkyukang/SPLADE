import types
import unittest

import torch

from src.utils.output_space import (
    OutputSpaceSpec,
    normalize_compact_head_alignment,
    resolve_model_output_exclude_ids,
)


class OutputSpaceSpecTest(unittest.TestCase):
    def test_normalize_compact_head_alignment_canonicalizes_variants(self) -> None:
        self.assertEqual(normalize_compact_head_alignment("token"), "token_ids")
        self.assertEqual(
            normalize_compact_head_alignment("cluster-centroid"),
            "latent_cluster",
        )
        self.assertIsNone(normalize_compact_head_alignment(None))

    def test_from_metadata_recovers_legacy_boolean_alignment(self) -> None:
        spec = OutputSpaceSpec.from_metadata(
            {
                "vocab_size": 8,
                "output_token_aligned": False,
            }
        )

        self.assertEqual(spec.compact_head_alignment, "latent_cluster")
        self.assertFalse(spec.output_token_aligned)

    def test_token_aligned_output_space_maps_token_ids_to_output_ids(self) -> None:
        spec = OutputSpaceSpec.from_alignment(
            vocab_size=4,
            compact_head_alignment="token_ids",
        )
        resolved = spec.resolve_exclude_token_ids(
            [20, 999, 10],
            token_id_to_output_index=torch.tensor(
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
                dtype=torch.long,
            ),
        )

        self.assertEqual(tuple(resolved.tolist()), (0, 1))

    def test_latent_cluster_output_space_ignores_token_id_exclusions(self) -> None:
        spec = OutputSpaceSpec.from_alignment(
            vocab_size=8,
            compact_head_alignment="latent_cluster",
        )
        resolved = spec.resolve_exclude_token_ids([10, 11, 12])
        self.assertEqual(int(resolved.numel()), 0)

    def test_model_output_exclude_ids_falls_back_to_output_space_spec(self) -> None:
        encoder = types.SimpleNamespace(
            output_space=OutputSpaceSpec.from_alignment(
                vocab_size=4,
                compact_head_alignment="token_ids",
            ),
            token_id_to_output_index=torch.tensor(
                [2, -1, -1, 1, -1, 0], dtype=torch.long
            ),
        )
        model = types.SimpleNamespace(encoder=encoder)

        resolved = resolve_model_output_exclude_ids(model, [5, 3, 100])

        self.assertEqual(resolved, [0, 1])


if __name__ == "__main__":
    unittest.main()
