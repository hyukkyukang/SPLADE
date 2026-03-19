import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import BertConfig, BertForMaskedLM

from src.utils.transformers import (
    _resolve_checkpoint_path,
    build_masked_lm_model,
    build_tokenizer,
)


class _DummyTokenizer:
    def __init__(self) -> None:
        self.is_fast: bool = True
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.cls_token = "[CLS]"


def _build_tiny_bert() -> BertForMaskedLM:
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=64,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    )
    config.tie_word_embeddings = True
    model = BertForMaskedLM(config)
    model.eval()
    return model


def _save_condenser_style_checkpoint(
    checkpoint_path: Path,
    *,
    source_model: BertForMaskedLM,
    tokenizer_path: str,
) -> None:
    model_state = source_model.state_dict()
    checkpoint_state: dict[str, torch.Tensor] = {}

    key: str
    value: torch.Tensor
    for key, value in model_state.items():
        if key.startswith("bert.embeddings."):
            suffix = key[len("bert.") :]
            checkpoint_state[f"model._orig_mod.{suffix}"] = value.detach().clone()
            continue
        if key.startswith("bert.encoder.layer.0."):
            suffix = key[len("bert.encoder.layer.0.") :]
            checkpoint_state[
                f"model._orig_mod.early_layers.0.{suffix}"
            ] = value.detach().clone()
            continue
        if key.startswith("bert.encoder.layer.1."):
            suffix = key[len("bert.encoder.layer.1.") :]
            checkpoint_state[
                f"model._orig_mod.late_layers.0.{suffix}"
            ] = value.detach().clone()
            continue
        if key.startswith("cls."):
            suffix = key[len("cls.") :]
            checkpoint_state[f"model._orig_mod.mlm_head.{suffix}"] = (
                value.detach().clone()
            )
            continue

    checkpoint = {
        "state_dict": checkpoint_state,
        "hyper_parameters": {
            "root_dir_path": ".",
            "model": {
                "name": "condenser_base",
                "vocab_size": 32,
                "hidden_size": 16,
                "num_early_layers": 1,
                "num_late_layers": 1,
                "num_head_layers": 1,
                "num_attention_heads": 4,
                "intermediate_size": 32,
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 64,
                "type_vocab_size": 2,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "tie_word_embeddings": True,
                "tokenizer_path": tokenizer_path,
            },
        },
    }
    torch.save(checkpoint, checkpoint_path)


class TransformersCheckpointFallbackTest(unittest.TestCase):
    def test_resolve_checkpoint_path_prefers_priority_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="checkpoint_priority_") as tmp:
            model_dir = Path(tmp)
            checkpoint_dir = model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            preferred = checkpoint_dir / "last.ckpt"
            fallback = checkpoint_dir / "epoch=09.ckpt"
            fallback.write_bytes(b"fallback")
            preferred.write_bytes(b"preferred")
            os.utime(fallback, (2_000_000_000, 2_000_000_000))
            os.utime(preferred, (1_000_000_000, 1_000_000_000))

            resolved = _resolve_checkpoint_path(str(model_dir))

            self.assertEqual(resolved, preferred)

    def test_resolve_checkpoint_path_uses_latest_mtime_without_priority(self) -> None:
        with tempfile.TemporaryDirectory(prefix="checkpoint_latest_") as tmp:
            model_dir = Path(tmp)
            checkpoint_dir = model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            older = checkpoint_dir / "z_old.ckpt"
            newer = checkpoint_dir / "a_new.ckpt"
            older.write_bytes(b"older")
            newer.write_bytes(b"newer")
            os.utime(older, (1_000_000_000, 1_000_000_000))
            os.utime(newer, (2_000_000_000, 2_000_000_000))

            resolved = _resolve_checkpoint_path(str(model_dir))

            self.assertEqual(resolved, newer)

    def test_build_masked_lm_model_falls_back_to_condenser_checkpoint(self) -> None:
        source_model = _build_tiny_bert()
        with tempfile.TemporaryDirectory(prefix="trained_anna_base_hf_") as tmp:
            model_dir = Path(tmp)
            checkpoint_dir = model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "last.ckpt"
            _save_condenser_style_checkpoint(
                checkpoint_path,
                source_model=source_model,
                tokenizer_path="./model/anna_base_hf",
            )

            loaded_model = build_masked_lm_model(str(model_dir))

            self.assertIsInstance(loaded_model, BertForMaskedLM)
            source_state = source_model.state_dict()
            loaded_state = loaded_model.state_dict()
            self.assertTrue(
                torch.equal(
                    source_state["bert.encoder.layer.1.output.dense.weight"],
                    loaded_state["bert.encoder.layer.1.output.dense.weight"],
                )
            )
            self.assertTrue(
                torch.equal(
                    source_state["cls.predictions.transform.dense.weight"],
                    loaded_state["cls.predictions.transform.dense.weight"],
                )
            )

    def test_build_tokenizer_uses_checkpoint_fallback_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tokenizer_fallback_") as tmp:
            tmp_path = Path(tmp)
            model_root = tmp_path / "data" / "model"
            trained_dir = model_root / "trained_anna_base_hf"
            checkpoint_dir = trained_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            fallback_tokenizer_dir = model_root / "anna_base_hf"
            fallback_tokenizer_dir.mkdir(parents=True, exist_ok=True)
            (fallback_tokenizer_dir / "vocab.txt").write_text(
                "[PAD]\n[UNK]\n", encoding="utf-8"
            )

            source_model = _build_tiny_bert()
            _save_condenser_style_checkpoint(
                checkpoint_dir / "last.ckpt",
                source_model=source_model,
                tokenizer_path="./model/anna_base_hf",
            )

            dummy_tokenizer = _DummyTokenizer()

            def fake_from_pretrained(source: str, **_: object) -> _DummyTokenizer:
                if source == str(trained_dir):
                    raise ValueError("no tokenizer artifacts")
                if source == str(fallback_tokenizer_dir):
                    return dummy_tokenizer
                raise ValueError(f"unexpected source: {source}")

            with patch(
                "src.utils.transformers.AutoTokenizer.from_pretrained",
                side_effect=fake_from_pretrained,
            ), patch(
                "src.utils.transformers.torch.load",
                side_effect=AssertionError(
                    "build_tokenizer should not deserialize checkpoints"
                ),
            ):
                tokenizer = build_tokenizer(
                    str(trained_dir),
                    use_fast_tokenizer=True,
                    trust_remote_code=True,
                    require_fast_tokenizer=True,
                )

            self.assertIs(tokenizer, dummy_tokenizer)

    def test_build_masked_lm_model_supports_causal_lm_loader(self) -> None:
        dummy_model = _build_tiny_bert()
        with patch(
            "src.utils.transformers.AutoModelForMaskedLM.from_pretrained"
        ) as mocked_masked_loader, patch(
            "src.utils.transformers.AutoModelForCausalLM.from_pretrained",
            return_value=dummy_model,
        ) as mocked_causal_loader:
            loaded_model = build_masked_lm_model(
                "google/embeddinggemma-300m",
                model_class_name="AutoModelForCausalLM",
            )
        self.assertIs(loaded_model, dummy_model)
        mocked_causal_loader.assert_called_once()
        mocked_masked_loader.assert_not_called()

    def test_build_masked_lm_model_supports_bidirectional_mistral_loader(self) -> None:
        dummy_model = _build_tiny_bert()
        with patch(
            "src.utils.transformers.MistralBiForCausalLM.from_pretrained",
            return_value=dummy_model,
        ) as mocked_bi_loader, patch(
            "src.utils.transformers.AutoModelForMaskedLM.from_pretrained"
        ) as mocked_masked_loader:
            loaded_model = build_masked_lm_model(
                "outputs/model_creation/lens/mistral_cluster4k",
                model_class_name="MistralBiForCausalLM",
            )
        self.assertIs(loaded_model, dummy_model)
        mocked_bi_loader.assert_called_once()
        mocked_masked_loader.assert_not_called()

    def test_build_masked_lm_model_rejects_unknown_model_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported huggingface_model_class"):
            _ = build_masked_lm_model(
                "distilbert-base-uncased",
                model_class_name="AutoModelForSequenceClassification",
            )


if __name__ == "__main__":
    unittest.main()
