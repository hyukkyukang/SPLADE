import unittest

from src.utils.checkpoint_compat import add_state_dict_prefix_aliases


class CheckpointCompatTest(unittest.TestCase):
    def test_adds_aliases_without_dropping_original_keys(self) -> None:
        state_dict = {
            "model._orig_mod.encoder.weight": 1,
            "model.encoder.bias": 2,
        }
        remapped = add_state_dict_prefix_aliases(
            state_dict,
            aliases=(("model._orig_mod.", "model."),),
        )

        self.assertIn("model._orig_mod.encoder.weight", remapped)
        self.assertIn("model.encoder.weight", remapped)
        self.assertEqual(remapped["model.encoder.weight"], 1)
        self.assertEqual(remapped["model.encoder.bias"], 2)

    def test_does_not_override_existing_destination_key(self) -> None:
        state_dict = {
            "model._orig_mod.encoder.weight": 1,
            "model.encoder.weight": 9,
        }
        remapped = add_state_dict_prefix_aliases(
            state_dict,
            aliases=(("model._orig_mod.", "model."),),
        )
        self.assertEqual(remapped["model.encoder.weight"], 9)


if __name__ == "__main__":
    unittest.main()
