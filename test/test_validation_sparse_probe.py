import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from omegaconf import OmegaConf

from src.data.collator import UniversalCollator
from src.data.dataclass import TrainingDataItem
from src.model.pl_module.validation_sparse_probe import ValidationSparseProbeLogger


class _FakeTokenizer:
    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"tok{int(token_id)}" for token_id in token_ids]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        _ = (skip_special_tokens, clean_up_tokenization_spaces)
        return " ".join(f"tok{int(token_id)}" for token_id in token_ids)


class _FakeOrderedModel:
    supports_ordered_mask_slot_loss = True

    def encode_queries_with_slot_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (input_ids, attention_mask, pooling_mask)
        query_reps = torch.tensor(
            [
                [0.0, 0.8, 0.2, 0.0, 0.5, 0.0],
                [0.0, 0.1, 0.9, 0.0, 0.4, 0.0],
            ],
            dtype=torch.float32,
        )
        slot_logits = torch.tensor(
            [
                [[0.1, 1.0, 0.3, 0.0, 0.2, 0.0], [0.0, 0.2, 0.1, 0.9, 0.4, 0.0]],
                [[0.0, 0.2, 1.1, 0.0, 0.5, 0.1], [0.0, 0.3, 0.2, 0.4, 1.2, 0.0]],
            ],
            dtype=torch.float32,
        )
        return query_reps[: input_ids.shape[0]], slot_logits[: input_ids.shape[0]]

    def encode_docs_with_slot_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (input_ids, attention_mask, pooling_mask)
        doc_reps = torch.tensor(
            [
                [0.0, 0.2, 0.0, 0.7, 0.6, 0.0],
                [0.0, 0.4, 0.0, 0.6, 0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        slot_logits = torch.tensor(
            [
                [[0.0, 0.1, 0.2, 1.3, 0.4, 0.0], [0.0, 0.2, 0.1, 0.3, 1.1, 0.0]],
                [[0.0, 0.3, 0.2, 1.0, 0.6, 0.0], [0.0, 0.4, 0.2, 0.5, 1.4, 0.0]],
            ],
            dtype=torch.float32,
        )
        return doc_reps[: input_ids.shape[0]], slot_logits[: input_ids.shape[0]]


class _SimpleDataset:
    def __init__(self, items: list[TrainingDataItem]) -> None:
        self._items = items
        self.collator = UniversalCollator(pad_token_id=0)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> TrainingDataItem:
        return self._items[idx]


def _build_item(index: int) -> TrainingDataItem:
    query_ids = torch.tensor([101, 11 + index, 12 + index, 0], dtype=torch.long)
    query_attention = torch.tensor([1, 1, 1, 0], dtype=torch.long)
    query_pooling = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    doc_ids = torch.tensor(
        [
            [201, 21 + index, 22 + index, 0],
            [201, 31 + index, 32 + index, 0],
        ],
        dtype=torch.long,
    )
    doc_attention = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    doc_pooling = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    return TrainingDataItem(
        data_idx=index,
        qid=f"q{index}",
        pos_ids=[f"p{index}"],
        neg_ids=[f"n{index}"],
        query_text=f"query text {index}",
        doc_texts=[f"positive doc {index}", f"negative doc {index}"],
        query_input_ids=query_ids,
        query_attention_mask=query_attention,
        query_pooling_mask=query_pooling,
        doc_input_ids=doc_ids,
        doc_attention_mask=doc_attention,
        doc_pooling_mask=doc_pooling,
        doc_mask=torch.tensor([True, True]),
        pos_mask=torch.tensor([True, False]),
        teacher_scores=torch.tensor([1.0, 0.0], dtype=torch.float32),
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        pos_scores=None,
        neg_scores=None,
    )


class ValidationSparseProbeLoggerTest(unittest.TestCase):
    def _build_module(
        self,
        *,
        log_dir: str,
        dataset: _SimpleDataset,
        probe_indices: list[int] | None = None,
        selection_seed: int = 13,
    ) -> tuple[SimpleNamespace, Mock]:
        mlflow_experiment: Mock = Mock()
        mlflow_logger = SimpleNamespace(
            run_id="unit-run-id",
            experiment=mlflow_experiment,
        )
        cfg = OmegaConf.create(
            {
                "seed": selection_seed,
                "log_dir": log_dir,
                "training": {
                    "validation_sparse_probe": {
                        "enabled": True,
                        "num_pairs": 2,
                        "top_k_sparse": 3,
                        "top_k_slot": 2,
                        "include_slot_logits": True,
                        "log_every_n_val": 1,
                        "selection_seed": selection_seed,
                        "probe_indices": probe_indices,
                        "artifact_dir": "validation_sparse_probe",
                        "persist_selection_filename": "selection.json",
                        "write_json": True,
                        "write_markdown": True,
                    }
                },
            }
        )
        trainer = SimpleNamespace(
            is_global_zero=True,
            sanity_checking=False,
            datamodule=SimpleNamespace(
                val_dataset=dataset,
                tokenizer=_FakeTokenizer(),
            ),
            loggers=[mlflow_logger],
        )
        module = SimpleNamespace(
            cfg=cfg,
            device=torch.device("cpu"),
            global_step=123,
            current_epoch=4,
            trainer=trainer,
            model=_FakeOrderedModel(),
        )
        return module, mlflow_experiment

    def test_logs_human_readable_sparse_probe_and_persists_selection(self) -> None:
        items = [_build_item(0), _build_item(1), _build_item(2)]
        dataset = _SimpleDataset(items)
        with tempfile.TemporaryDirectory() as tmpdir:
            module, mlflow_experiment = self._build_module(
                log_dir=tmpdir,
                dataset=dataset,
                probe_indices=None,
                selection_seed=7,
            )
            logger = ValidationSparseProbeLogger(
                module=module,
                cfg=module.cfg,
                logger=Mock(),
            )
            logger.run_validation_epoch_end()

            selection_path = (
                Path(tmpdir) / "validation_sparse_probe" / "selection.json"
            )
            self.assertTrue(selection_path.is_file())
            selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
            self.assertEqual(len(selection_payload["indices"]), 2)

            json_path = (
                Path(tmpdir) / "validation_sparse_probe" / "step_00000123.json"
            )
            markdown_path = (
                Path(tmpdir) / "validation_sparse_probe" / "step_00000123.md"
            )
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["global_step"], 123)
            self.assertEqual(len(payload["samples"]), 2)
            self.assertEqual(payload["samples"][0]["query_text"][:10], "query text")
            self.assertEqual(payload["samples"][0]["doc_id"][0], "p")
            self.assertEqual(
                payload["samples"][0]["query_top_sparse"][0]["token"],
                "tok1",
            )
            self.assertIn("query_slot_topk", payload["samples"][0])
            self.assertIn("doc_slot_topk", payload["samples"][0])

            self.assertEqual(mlflow_experiment.log_artifact.call_count, 2)

            second_module, _ = self._build_module(
                log_dir=tmpdir,
                dataset=dataset,
                probe_indices=None,
                selection_seed=999,
            )
            second_logger = ValidationSparseProbeLogger(
                module=second_module,
                cfg=second_module.cfg,
                logger=Mock(),
            )
            self.assertEqual(
                second_logger._resolve_probe_indices(dataset),
                selection_payload["indices"],
            )

    def test_uses_explicit_probe_indices(self) -> None:
        items = [_build_item(0), _build_item(1), _build_item(2)]
        dataset = _SimpleDataset(items)
        with tempfile.TemporaryDirectory() as tmpdir:
            module, _ = self._build_module(
                log_dir=tmpdir,
                dataset=dataset,
                probe_indices=[2, 0],
            )
            logger = ValidationSparseProbeLogger(
                module=module,
                cfg=module.cfg,
                logger=Mock(),
            )
            self.assertEqual(logger._resolve_probe_indices(dataset), [2, 0])


if __name__ == "__main__":
    unittest.main()
