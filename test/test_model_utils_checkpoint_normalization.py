import unittest

from src.utils.model_utils import _normalize_checkpoint_state_dict


class ModelUtilsCheckpointNormalizationTest(unittest.TestCase):
    def test_collapses_compiled_encoder_fn_prefixes_to_encoder(self) -> None:
        normalized = _normalize_checkpoint_state_dict(
            {
                "model._query_encoder_fn._encoder_fn._orig_mod.mlm.weight": 1,
                "model._doc_encoder_fn._encoder_fn._orig_mod.mlm.weight": 2,
            }
        )

        self.assertEqual(normalized["encoder.mlm.weight"], 1)
        self.assertNotIn("_query_encoder_fn._encoder_fn.mlm.weight", normalized)
        self.assertNotIn("_doc_encoder_fn._encoder_fn.mlm.weight", normalized)
        self.assertEqual(normalized["_query_encoder_wrapper.encoder.mlm.weight"], 1)
        self.assertEqual(normalized["_doc_encoder_wrapper.encoder.mlm.weight"], 1)

    def test_prefers_canonical_encoder_key_over_wrapper_duplicate(self) -> None:
        normalized = _normalize_checkpoint_state_dict(
            {
                "model._query_encoder_fn._encoder_fn._orig_mod.mlm.weight": 1,
                "model.encoder.mlm.weight": 9,
            }
        )

        self.assertEqual(normalized["encoder.mlm.weight"], 9)
        self.assertEqual(normalized["_query_encoder_wrapper.encoder.mlm.weight"], 9)

    def test_normalizes_peft_lora_keys_under_compiled_encoder_wrappers(self) -> None:
        normalized = _normalize_checkpoint_state_dict(
            {
                (
                    "model._query_encoder_fn._encoder_fn._orig_mod.mlm.base_model."
                    "model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
                ): 3,
            }
        )

        canonical_key = (
            "encoder.mlm.base_model.model.model.layers.0.self_attn.q_proj."
            "lora_A.default.weight"
        )
        self.assertEqual(normalized[canonical_key], 3)
        self.assertEqual(
            normalized[
                (
                    "_query_encoder_wrapper.encoder.mlm.base_model.model.model."
                    "layers.0.self_attn.q_proj.lora_A.default.weight"
                )
            ],
            3,
        )


if __name__ == "__main__":
    unittest.main()
