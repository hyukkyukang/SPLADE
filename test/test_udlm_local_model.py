import unittest

import torch
from transformers.modeling_outputs import MaskedLMOutput

from src.model.retriever.sparse.neural.udlm import UDLMConfig, UDLMForMaskedLMCompat


class UDLMForMaskedLMCompatTest(unittest.TestCase):
    def _build_model(self) -> UDLMForMaskedLMCompat:
        model = UDLMForMaskedLMCompat(
            UDLMConfig(
                vocab_size=32,
                model_length=8,
                hidden_dim=8,
                cond_dim=4,
                n_blocks=1,
                n_heads=2,
                dropout=0.0,
                time_conditioning=True,
                return_dict=False,
            )
        )
        model.eval()
        return model

    def test_forward_defaults_to_masked_lm_output_with_zero_timesteps(self) -> None:
        model = self._build_model()
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        implicit = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        explicit = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            timesteps=torch.zeros((1,), dtype=torch.float32),
            return_dict=True,
        )

        self.assertIsInstance(implicit, MaskedLMOutput)
        self.assertTrue(torch.allclose(implicit.logits, explicit.logits))

    def test_attention_mask_prevents_padding_leakage(self) -> None:
        model = self._build_model()
        short_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
        short_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        padded_ids = torch.tensor([[5, 6, 7, 0, 0]], dtype=torch.long)
        padded_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
        timesteps = torch.zeros((1,), dtype=torch.float32)

        short_output = model(
            input_ids=short_ids,
            attention_mask=short_mask,
            timesteps=timesteps,
            return_dict=True,
        )
        padded_output = model(
            input_ids=padded_ids,
            attention_mask=padded_mask,
            timesteps=timesteps,
            return_dict=True,
        )

        self.assertTrue(
            torch.allclose(
                short_output.logits[:, :3, :],
                padded_output.logits[:, :3, :],
                atol=1e-5,
                rtol=1e-4,
            )
        )

    def test_forward_accepts_use_cache_and_hidden_states(self) -> None:
        model = self._build_model()
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            timesteps=torch.zeros((1,), dtype=torch.float32),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        self.assertIsInstance(output, MaskedLMOutput)
        self.assertIsNotNone(output.hidden_states)
        self.assertGreater(len(output.hidden_states), 0)


if __name__ == "__main__":
    unittest.main()
