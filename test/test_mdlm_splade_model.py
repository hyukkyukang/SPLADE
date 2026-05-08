import logging
import types
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from src.model.pl_module.compile_policy import TrainingCompilePolicyManager
from src.model.retriever.sparse.neural.mdlm_splade import MDLMSpladeModel
from src.utils.model_utils import build_splade_model


class _DummyModelOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits: torch.Tensor = logits


class _DummyMaskedLM(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 12) -> None:
        super().__init__()
        self.dummy_weight = torch.nn.Parameter(torch.ones((), dtype=torch.float32))
        self.config = types.SimpleNamespace(
            vocab_size=int(vocab_size),
            use_cache=False,
            pad_token_id=0,
            unk_token_id=1,
            cls_token_id=2,
            sep_token_id=3,
            mask_token_id=4,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
    ) -> _DummyModelOutput:
        _ = attention_mask, use_cache
        logits: torch.Tensor = torch.nn.functional.one_hot(
            input_ids,
            num_classes=int(self.config.vocab_size),
        ).to(dtype=torch.float32)
        logits = logits * 10.0
        return _DummyModelOutput(logits)


class _TimedDummyMaskedLM(_DummyMaskedLM):
    def __init__(self, *, vocab_size: int = 12) -> None:
        super().__init__(vocab_size=vocab_size)
        self.forward_calls: list[dict[str, torch.Tensor | bool | None]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
        timesteps: torch.Tensor | None = None,
    ) -> _DummyModelOutput:
        stored_timesteps: torch.Tensor | None = None
        if timesteps is not None:
            stored_timesteps = timesteps.detach().clone()
        self.forward_calls.append(
            {
                "timesteps": stored_timesteps,
                "use_cache": bool(use_cache),
            }
        )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )


def _build_cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "mdlm_splade_distilbert",
                "family": "mdlm_splade",
                "huggingface_name": "distilbert-base-uncased",
                "huggingface_model_class": "AutoModelForMaskedLM",
                "query_pooling": "max",
                "doc_pooling": "max",
                "sparse_activation": "log1p_relu",
                "attn_implementation": "sdpa",
                "dtype": "float32",
                "normalize": False,
                "doc_only": False,
                "tie_word_embeddings": True,
                "freeze_backbone": False,
                "exclude_token_ids": [0, 1, 2, 3, 4],
                "mask_token_id": 4,
                "peft": {"enabled": False},
            }
        }
    )


class MDLMSpladeModelTest(unittest.TestCase):
    def test_build_splade_model_returns_mdlm_model_for_family(self) -> None:
        cfg = _build_cfg()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=_DummyMaskedLM(),
        ):
            model = build_splade_model(cfg, use_cpu=True)
        self.assertIsInstance(model, MDLMSpladeModel)
        self.assertTrue(bool(model.supports_mdlm_aux_loss))

    def test_encode_queries_zeroes_special_token_dimensions(self) -> None:
        cfg = _build_cfg()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=_DummyMaskedLM(),
        ):
            model = build_splade_model(cfg, use_cpu=True)

        input_ids = torch.tensor([[2, 5, 4]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

        reps = model.encode_queries(input_ids, attention_mask)

        self.assertEqual(float(reps[0, 2].item()), 0.0)
        self.assertEqual(float(reps[0, 4].item()), 0.0)
        self.assertGreater(float(reps[0, 5].item()), 0.0)

    def test_fused_encode_queries_and_docs_applies_output_mask(self) -> None:
        cfg = _build_cfg()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=_DummyMaskedLM(),
        ):
            model = build_splade_model(cfg, use_cpu=True)

        compile_policy = TrainingCompilePolicyManager(
            model=model,
            logger=logging.getLogger("test_mdlm_splade_model"),
        )
        query_input_ids = torch.tensor([[2, 5, 4]], dtype=torch.long)
        query_attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        doc_input_ids = torch.tensor([[3, 6, 4]], dtype=torch.long)
        doc_attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

        query_reps, doc_reps = compile_policy.encode_queries_and_docs(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
        )

        self.assertEqual(float(query_reps[0, 2].item()), 0.0)
        self.assertEqual(float(query_reps[0, 4].item()), 0.0)
        self.assertGreater(float(query_reps[0, 5].item()), 0.0)
        self.assertEqual(float(doc_reps[0, 3].item()), 0.0)
        self.assertEqual(float(doc_reps[0, 4].item()), 0.0)
        self.assertGreater(float(doc_reps[0, 6].item()), 0.0)

    def test_subs_log_probs_copy_unmasked_tokens_and_forbid_mask_prediction(self) -> None:
        cfg = _build_cfg()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=_DummyMaskedLM(),
        ):
            model = build_splade_model(cfg, use_cpu=True)

        raw_logits = torch.zeros((1, 2, 12), dtype=torch.float32)
        xt = torch.tensor([[5, 4]], dtype=torch.long)

        log_probs = model.subs_log_probs(raw_logits, xt)

        self.assertEqual(float(log_probs[0, 0, 5].item()), 0.0)
        self.assertLess(float(log_probs[0, 0, 4].item()), -1e8)
        self.assertLess(float(log_probs[0, 1, 4].item()), -1e8)
        self.assertTrue(torch.isfinite(log_probs[0, 1, 5]))

    def test_masked_target_nll_matches_subs_log_probs_for_masked_positions(self) -> None:
        cfg = _build_cfg()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=_DummyMaskedLM(),
        ):
            model = build_splade_model(cfg, use_cpu=True)

        raw_logits = torch.randn((2, 3, 12), dtype=torch.float32)
        input_ids = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.long)
        xt = torch.tensor([[4, 6, 4], [8, 4, 10]], dtype=torch.long)
        masked = xt.eq(model.mask_token_id)

        log_probs = model.subs_log_probs(raw_logits, xt)
        dense_nll = -torch.gather(
            log_probs,
            dim=-1,
            index=input_ids.unsqueeze(-1),
        ).squeeze(-1)
        efficient_nll = model._masked_target_nll(raw_logits, input_ids)

        self.assertTrue(
            torch.allclose(
                efficient_nll[masked],
                dense_nll[masked],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_sample_noisy_view_forces_one_lexical_mask_when_needed(self) -> None:
        cfg = _build_cfg()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=_DummyMaskedLM(),
        ):
            model = build_splade_model(cfg, use_cpu=True)

        input_ids = torch.tensor([[2, 5, 3]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

        first_rand = torch.tensor([0.2], dtype=torch.float32)
        second_rand = torch.full((1, 3), 0.9, dtype=torch.float32)
        third_rand = torch.tensor([[0.1, 0.8, 0.2]], dtype=torch.float32)

        with patch(
            "src.model.retriever.sparse.neural.mdlm_splade.torch.rand",
            side_effect=[first_rand, second_rand, third_rand],
        ):
            xt, masked, t = model.sample_noisy_view(
                input_ids,
                attention_mask,
                mask_probability_eps=1e-3,
                force_mask_at_least_one=True,
            )

        self.assertAlmostEqual(float(t[0].item()), 0.2, places=6)
        self.assertFalse(bool(masked[0, 0].item()))
        self.assertTrue(bool(masked[0, 1].item()))
        self.assertFalse(bool(masked[0, 2].item()))
        self.assertEqual(int(xt[0, 1].item()), 4)
        self.assertEqual(int(xt[0, 0].item()), 2)
        self.assertEqual(int(xt[0, 2].item()), 3)

    def test_compute_mdlm_aux_loss_passes_sampled_timesteps_when_supported(self) -> None:
        cfg = _build_cfg()
        timed_mlm = _TimedDummyMaskedLM()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=timed_mlm,
        ):
            model = build_splade_model(cfg, use_cpu=True)

        input_ids = torch.tensor([[2, 5, 3]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        sampled_t = torch.tensor([0.35], dtype=torch.float32)
        masked = torch.tensor([[False, True, False]], dtype=torch.bool)

        with patch.object(
            model,
            "sample_noisy_view",
            return_value=(input_ids.clone(), masked, sampled_t),
        ):
            _ = model.compute_mdlm_aux_loss(input_ids, attention_mask)

        self.assertGreaterEqual(len(timed_mlm.forward_calls), 1)
        recorded_t = timed_mlm.forward_calls[-1]["timesteps"]
        self.assertIsInstance(recorded_t, torch.Tensor)
        assert isinstance(recorded_t, torch.Tensor)
        self.assertTrue(torch.allclose(recorded_t, sampled_t))

    def test_compute_grouped_mdlm_aux_losses_uses_one_forward_for_same_shape_groups(self) -> None:
        cfg = _build_cfg()
        timed_mlm = _TimedDummyMaskedLM()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=timed_mlm,
        ):
            model = build_splade_model(cfg, use_cpu=True)

        input_ids_groups = (
            torch.tensor([[2, 5, 3]], dtype=torch.long),
            torch.tensor([[2, 6, 3], [2, 7, 3]], dtype=torch.long),
        )
        attention_mask_groups = tuple(torch.ones_like(group) for group in input_ids_groups)

        losses = model.compute_grouped_mdlm_aux_losses(
            input_id_groups=input_ids_groups,
            attention_mask_groups=attention_mask_groups,
        )

        self.assertEqual(len(losses), 2)
        self.assertEqual(len(timed_mlm.forward_calls), 1)

    def test_compute_grouped_mdlm_aux_losses_respects_reduction_masks(self) -> None:
        cfg = _build_cfg()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=_DummyMaskedLM(),
        ):
            model = build_splade_model(cfg, use_cpu=True)

        input_ids_groups = (
            torch.tensor([[2, 5, 3]], dtype=torch.long),
            torch.tensor([[2, 6, 3], [2, 7, 3]], dtype=torch.long),
        )
        attention_mask_groups = tuple(torch.ones_like(group) for group in input_ids_groups)
        reduction_mask_groups = (
            None,
            torch.tensor([True, False], dtype=torch.bool),
        )

        with patch.object(
            model,
            "_compute_mdlm_aux_loss_per_example",
            return_value=torch.tensor([1.0, 3.0, 9.0], dtype=torch.float32),
        ):
            losses = model.compute_grouped_mdlm_aux_losses(
                input_id_groups=input_ids_groups,
                attention_mask_groups=attention_mask_groups,
                reduction_mask_groups=reduction_mask_groups,
            )

        self.assertEqual(len(losses), 2)
        self.assertAlmostEqual(float(losses[0].item()), 1.0)
        self.assertAlmostEqual(float(losses[1].item()), 3.0)

    def test_compute_grouped_mdlm_aux_losses_rejects_mismatched_sequence_lengths(self) -> None:
        cfg = _build_cfg()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=_DummyMaskedLM(),
        ):
            model = build_splade_model(cfg, use_cpu=True)

        with self.assertRaisesRegex(ValueError, "share sequence length"):
            model.compute_grouped_mdlm_aux_losses(
                input_id_groups=(
                    torch.tensor([[2, 5, 3]], dtype=torch.long),
                    torch.tensor([[2, 6, 3, 4]], dtype=torch.long),
                ),
                attention_mask_groups=(
                    torch.ones((1, 3), dtype=torch.long),
                    torch.ones((1, 4), dtype=torch.long),
                ),
            )


if __name__ == "__main__":
    unittest.main()
