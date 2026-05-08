import types
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from import_stubs import install_fake_sentence_transformers

install_fake_sentence_transformers()

from src.model.retriever.sparse.neural.ordered_mask_slot_splade import (
    OrderedMaskSlotSpladeModel,
)
from src.model.retriever.sparse.neural.pretrained_diffusion_ordered_mask_slot_splade import (
    PretrainedDiffusionOrderedMaskSlotSpladeModel,
)
from src.utils.model_utils import build_splade_model
from src.utils.sparse_encoder import resolve_nanobeir_backend


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
        return _DummyModelOutput(logits * 10.0)


class _DummyTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab: dict[str, int] = dict(vocab)
        self.is_fast: bool = True
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.cls_token = "[CLS]"

    def __len__(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def _build_distil_cfg() -> object:
    return OmegaConf.create(
        {
            "model": {
                "name": "ordered_mask_slot_splade_distilbert",
                "family": "ordered_mask_slot_splade",
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
                "num_mask_slots": 2,
                "benchmark_adapter": "auto",
                "peft": {"enabled": False},
            }
        }
    )


def _build_diffusion_cfg() -> object:
    return OmegaConf.create(
        {
            "model": {
                "name": "pretrained_diffusion_ordered_mask_slot_splade_udlm_lm1b",
                "family": "pretrained_diffusion_ordered_mask_slot_splade",
                "backbone_pretraining_type": "diffusion",
                "huggingface_name": "kuleshov-group/udlm-lm1b",
                "tokenizer_name": "bert-base-uncased",
                "baseline_tokenizer_name": "distilbert-base-uncased",
                "enforce_same_tokenizer_as_baseline": True,
                "huggingface_model_class": "UDLMForMaskedLMCompat",
                "query_pooling": "max",
                "doc_pooling": "max",
                "sparse_activation": "log1p_relu",
                "attn_implementation": "sdpa",
                "dtype": "float32",
                "normalize": False,
                "doc_only": False,
                "tie_word_embeddings": True,
                "freeze_backbone": False,
                "trust_remote_code": False,
                "model_revision": "00dfee2a0578719ea93739884173d4393906a8fd",
                "tokenizer_revision": "main",
                "local_files_only": None,
                "use_fast_tokenizer": True,
                "require_fast_tokenizer": False,
                "benchmark_adapter": "auto",
                "exclude_token_ids": [0, 1, 2, 3, 4],
                "mask_token_id": 4,
                "num_mask_slots": 2,
                "peft": {"enabled": False},
            }
        }
    )


class OrderedMaskSlotSpladeModelTest(unittest.TestCase):
    def test_build_splade_model_returns_ordered_mask_slot_model(self) -> None:
        cfg = _build_distil_cfg()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=_DummyMaskedLM(),
        ), patch(
            "src.model.retriever.sparse.neural.splade._resolve_compact_head_path",
            return_value=None,
        ):
            model = build_splade_model(cfg, use_cpu=True)

        self.assertIsInstance(model, OrderedMaskSlotSpladeModel)
        self.assertTrue(bool(model.supports_ordered_mask_slot_loss))
        self.assertEqual(int(model.num_mask_slots), 2)

    def test_encode_queries_with_slot_logits_masks_special_outputs_and_extracts_slots(self) -> None:
        cfg = _build_distil_cfg()
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=_DummyMaskedLM(),
        ), patch(
            "src.model.retriever.sparse.neural.splade._resolve_compact_head_path",
            return_value=None,
        ):
            model = build_splade_model(cfg, use_cpu=True)

        input_ids = torch.tensor([[2, 5, 4, 6, 4]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
        pooling_mask = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.long)

        reps, slot_logits = model.encode_queries_with_slot_logits(
            input_ids,
            attention_mask,
            pooling_mask=pooling_mask,
        )

        self.assertEqual(tuple(slot_logits.shape), (1, 2, 12))
        self.assertEqual(float(reps[0, 4].item()), 0.0)
        self.assertGreater(float(reps[0, 6].item()), 0.0)
        self.assertEqual(int(slot_logits[0, 0].argmax().item()), 6)
        self.assertEqual(int(slot_logits[0, 1].argmax().item()), 4)

    def test_build_splade_model_returns_pretrained_diffusion_ordered_mask_slot_model(self) -> None:
        cfg = _build_diffusion_cfg()
        vocab = {f"tok-{idx}": idx for idx in range(12)}
        tokenizer = _DummyTokenizer(vocab)
        with patch(
            "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
            return_value=_DummyMaskedLM(),
        ), patch(
            "src.model.retriever.sparse.neural.splade._resolve_compact_head_path",
            return_value=None,
        ), patch(
            "src.model.retriever.sparse.neural.pretrained_diffusion_ordered_mask_slot_splade.build_tokenizer",
            side_effect=[tokenizer, tokenizer],
        ):
            model = build_splade_model(cfg, use_cpu=True)

        self.assertIsInstance(model, PretrainedDiffusionOrderedMaskSlotSpladeModel)
        self.assertTrue(bool(model.supports_ordered_mask_slot_loss))

    def test_ordered_mask_slot_families_use_native_nanobeir_backend(self) -> None:
        distil_cfg = OmegaConf.create(
            {
                "model": {
                    "family": "ordered_mask_slot_splade",
                    "benchmark_adapter": "auto",
                    "doc_only": False,
                    "peft": {"enabled": False},
                }
            }
        )
        diffusion_cfg = OmegaConf.create(
            {
                "model": {
                    "family": "pretrained_diffusion_ordered_mask_slot_splade",
                    "benchmark_adapter": "auto",
                    "doc_only": False,
                    "peft": {"enabled": False},
                }
            }
        )

        distil_backend, _ = resolve_nanobeir_backend(distil_cfg)
        diffusion_backend, _ = resolve_nanobeir_backend(diffusion_cfg)

        self.assertEqual(distil_backend, "native")
        self.assertEqual(diffusion_backend, "native")


if __name__ == "__main__":
    unittest.main()
