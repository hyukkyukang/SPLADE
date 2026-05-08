import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import BertConfig, BertModel

from src.model.retriever.dense.neural.dpr_biencoder import (
    build_dpr_biencoder_model,
    infer_dpr_bert_config_from_checkpoint,
    infer_dpr_bert_config_from_state_dict,
    load_dpr_biencoder_checkpoint,
)


class DPRBiEncoderModelTest(unittest.TestCase):
    def _make_tower(self, seed: int) -> BertModel:
        torch.manual_seed(seed)
        config = BertConfig(
            vocab_size=17,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=16,
            type_vocab_size=2,
            pad_token_id=0,
        )
        return BertModel(config)

    def _build_checkpoint(self, checkpoint_path: Path) -> None:
        question_model = self._make_tower(seed=123)
        ctx_model = self._make_tower(seed=456)
        state_dict: dict[str, torch.Tensor] = {}
        for key, value in question_model.state_dict().items():
            state_dict[f"question_model.{key}"] = value.clone()
        for key, value in ctx_model.state_dict().items():
            state_dict[f"ctx_model.{key}"] = value.clone()
        torch.save(
            {
                "model_dict": state_dict,
                "encoder_params": {
                    "encoder_model_type": "bilingual_encoder",
                    "projection_dim": 0,
                    "sequence_length": 16,
                },
            },
            checkpoint_path,
        )

    def test_infer_dpr_config_from_state_dict(self) -> None:
        question_model = self._make_tower(seed=123)
        state_dict = {
            f"question_model.{key}": value.clone()
            for key, value in question_model.state_dict().items()
        }
        config = infer_dpr_bert_config_from_state_dict(state_dict)
        self.assertEqual(config.vocab_size, 17)
        self.assertEqual(config.hidden_size, 8)
        self.assertEqual(config.num_hidden_layers, 2)
        self.assertEqual(config.num_attention_heads, 2)
        self.assertEqual(config.intermediate_size, 16)
        self.assertEqual(config.max_position_embeddings, 16)

    def test_build_and_load_dpr_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "dpr.ckpt"
            self._build_checkpoint(checkpoint_path)
            config = infer_dpr_bert_config_from_checkpoint(checkpoint_path)
            self.assertEqual(config.hidden_size, 8)
            cfg = OmegaConf.create(
                {
                    "family": "dense",
                    "dense_architecture": "dpr_biencoder",
                    "huggingface_name": str(checkpoint_path),
                    "query_pooling": "cls",
                    "doc_pooling": "cls",
                    "query_window_pooling": "mean",
                    "doc_window_pooling": "mean",
                    "similarity": "dot",
                    "normalize": False,
                    "bert_config": {
                        "vocab_size": 17,
                        "hidden_size": 8,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "intermediate_size": 16,
                        "max_position_embeddings": 16,
                        "type_vocab_size": 2,
                        "pad_token_id": 0,
                    },
                }
            )
            model = build_dpr_biencoder_model(
                model_cfg=cfg,
                dtype=None,
                checkpoint_path=str(checkpoint_path),
            )
            missing, unexpected = load_dpr_biencoder_checkpoint(
                model,
                str(checkpoint_path),
            )
            self.assertEqual(missing, [])
            self.assertEqual(unexpected, [])
            model.eval()
            input_ids = torch.tensor([[1, 2, 3, 4]])
            attention_mask = torch.tensor([[1, 1, 1, 1]])
            query_emb = model.encode_queries(input_ids, attention_mask)
            doc_emb = model.encode_docs(input_ids, attention_mask)
            self.assertEqual(tuple(query_emb.shape), (1, 8))
            self.assertEqual(tuple(doc_emb.shape), (1, 8))
            self.assertFalse(torch.allclose(query_emb, doc_emb))

    def test_build_dpr_model_defaults_to_cls_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "dpr.ckpt"
            self._build_checkpoint(checkpoint_path)
            cfg = OmegaConf.create(
                {
                    "family": "dense",
                    "dense_architecture": "dpr_biencoder",
                    "huggingface_name": str(checkpoint_path),
                    "similarity": "dot",
                    "normalize": False,
                    "bert_config": {
                        "vocab_size": 17,
                        "hidden_size": 8,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "intermediate_size": 16,
                        "max_position_embeddings": 16,
                        "type_vocab_size": 2,
                        "pad_token_id": 0,
                    },
                }
            )
            model = build_dpr_biencoder_model(
                model_cfg=cfg,
                dtype=None,
                checkpoint_path=str(checkpoint_path),
            )
            self.assertEqual(model.query_pooling, "cls")
            self.assertEqual(model.doc_pooling, "cls")
            self.assertEqual(model.query_window_pooling, "cls")
            self.assertEqual(model.doc_window_pooling, "cls")
