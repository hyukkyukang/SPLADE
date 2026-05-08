from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.preprocess.patent_splade_terms import (
    COMBINED_TRUNCATE_DOCUMENT_ENCODING_MODE,
    LocalParquetPatentSource,
    PatentDocument,
    PatentWindowBatchCollator,
    build_source_token_key,
    build_document_windows,
    build_prefixed_windows,
    build_runtime_model_and_tokenizer,
    collect_patent_ids,
    dense_vector_to_source_token_term_weights,
    dense_vector_to_term_weights,
    normalize_patent_text,
    select_contiguous_shard,
    update_aggregated_term_provenance,
)
from src.utils.output_space import OutputSpaceSpec


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.all_special_ids = [self.pad_token_id, self.cls_token_id, self.sep_token_id]
        self._token_to_id: dict[str, int] = {
            "[PAD]": self.pad_token_id,
            "[CLS]": self.cls_token_id,
            "[SEP]": self.sep_token_id,
        }
        self._id_to_token: dict[int, str] = {
            self.pad_token_id: "[PAD]",
            self.cls_token_id: "[CLS]",
            self.sep_token_id: "[SEP]",
        }
        self._next_id: int = 1000

    def _normalize_tokens(self, text: str) -> list[str]:
        return text.replace("\n", " ").split()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        token: str
        for token in self._normalize_tokens(text):
            token_id: int | None = self._token_to_id.get(token)
            if token_id is None:
                token_id = self._next_id
                self._next_id += 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(token_id)
        if not add_special_tokens:
            return ids
        return [self.cls_token_id, *ids, self.sep_token_id]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        _ = pair
        return 2

    def prepare_for_model(
        self,
        token_ids: list[int],
        *,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        return_attention_mask: bool = True,
    ) -> dict[str, list[int]]:
        _ = padding, truncation, return_attention_mask
        if add_special_tokens:
            input_ids: list[int] = [self.cls_token_id, *token_ids, self.sep_token_id]
        else:
            input_ids = list(token_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

    def build_inputs_with_special_tokens(self, token_ids_0: list[int]) -> list[int]:
        return [self.cls_token_id, *token_ids_0, self.sep_token_id]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_token[int(token_id)] for token_id in ids]

    def pad(
        self,
        encoded_inputs: list[dict[str, list[int]]],
        *,
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        _ = padding
        if return_tensors != "pt":
            raise ValueError("This fake tokenizer only supports return_tensors='pt'.")
        max_length = max(len(item["input_ids"]) for item in encoded_inputs)
        padded_input_ids: list[list[int]] = []
        padded_attention_masks: list[list[int]] = []
        item: dict[str, list[int]]
        for item in encoded_inputs:
            pad_length = max_length - len(item["input_ids"])
            padded_input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_length)
            padded_attention_masks.append(item["attention_mask"] + [0] * pad_length)
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
        }


class PatentSpladeTermsTest(unittest.TestCase):
    def test_collect_patent_ids_dedupes_question_and_nested_label_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_a = root / "a.json"
            dataset_b = root / "b.json"
            dataset_a.write_text(
                json.dumps(
                    [
                        {
                            "question_id": "US1",
                            "labels": [{"label_id": "US2"}, {"label_id": "US3"}],
                        },
                        {"question_id": "US2", "labels": [{"label_id": "US4"}]},
                    ]
                ),
                encoding="utf-8",
            )
            dataset_b.write_text(
                json.dumps(
                    [
                        {
                            "question_id": "US5",
                            "labels": [{"label_id": "US3"}, {"label_id": "US6"}],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                collect_patent_ids([dataset_a, dataset_b]),
                ["US1", "US2", "US3", "US4", "US5", "US6"],
            )

    def test_normalize_patent_text_joins_claim_bodies(self) -> None:
        claims = [
            {"number": "1", "body": "First claim body"},
            {"number": "2", "body": "Second claim body"},
        ]
        self.assertEqual(
            normalize_patent_text(claims),
            "First claim body Second claim body",
        )

    def test_build_prefixed_windows_repeats_title_prefix_for_each_claim_chunk(self) -> None:
        tokenizer = _FakeTokenizer()
        title = "Alpha Beta"
        claims = "one two three four five six seven eight"
        prefix_ids = tokenizer.encode("Title: Alpha Beta Claims:", add_special_tokens=False)
        windows = build_prefixed_windows(
            tokenizer,
            title=title,
            field_label="Claims",
            field_text=claims,
            max_length=9,
            overlap_tokens=0,
        )
        self.assertGreater(len(windows), 1)
        input_ids: list[int]
        _attention_mask: list[int]
        for input_ids, _attention_mask in windows:
            self.assertEqual(input_ids[1 : 1 + len(prefix_ids)], prefix_ids)

    def test_build_document_windows_uses_title_only_fallback(self) -> None:
        tokenizer = _FakeTokenizer()
        document = PatentDocument(
            doc_id="US1",
            title="Fallback Title",
            abstract="",
            claims="",
        )
        windows = build_document_windows(
            tokenizer,
            document,
            max_length=16,
        )
        self.assertEqual(len(windows), 1)
        tokens = tokenizer.convert_ids_to_tokens(windows[0].input_ids)
        self.assertEqual(tokens[1], "Title:")
        self.assertEqual(tokens[2], "Fallback")
        self.assertEqual(tokens[3], "Title")

    def test_build_document_windows_default_mode_keeps_abstract_and_claims_separate(self) -> None:
        tokenizer = _FakeTokenizer()
        document = PatentDocument(
            doc_id="US1",
            title="Joined Title",
            abstract="Abstract body",
            claims="Claims body",
        )
        windows = build_document_windows(
            tokenizer,
            document,
            max_length=32,
        )
        self.assertEqual(len(windows), 2)
        abstract_tokens = tokenizer.convert_ids_to_tokens(windows[0].input_ids)
        claims_tokens = tokenizer.convert_ids_to_tokens(windows[1].input_ids)
        self.assertIn("Abstract:", abstract_tokens)
        self.assertNotIn("Claims:", abstract_tokens)
        self.assertIn("Claims:", claims_tokens)
        self.assertNotIn("Abstract:", claims_tokens)

    def test_build_document_windows_combined_mode_uses_single_sequence(self) -> None:
        tokenizer = _FakeTokenizer()
        document = PatentDocument(
            doc_id="US1",
            title="Joined Title",
            abstract="Abstract body",
            claims="Claims body",
        )
        windows = build_document_windows(
            tokenizer,
            document,
            max_length=32,
            document_encoding_mode=COMBINED_TRUNCATE_DOCUMENT_ENCODING_MODE,
        )
        self.assertEqual(len(windows), 1)
        tokens = tokenizer.convert_ids_to_tokens(windows[0].input_ids)
        self.assertEqual(
            tokens[1:-1],
            [
                "Title:",
                "Joined",
                "Title",
                "Abstract:",
                "Abstract",
                "body",
                "Claims:",
                "Claims",
                "body",
            ],
        )

    def test_build_document_windows_combined_mode_truncates_to_first_window_only(self) -> None:
        tokenizer = _FakeTokenizer()
        document = PatentDocument(
            doc_id="US1",
            title="Alpha",
            abstract="a1 a2 a3 a4",
            claims="c1 c2 c3 c4",
        )
        windows = build_document_windows(
            tokenizer,
            document,
            max_length=8,
            document_encoding_mode=COMBINED_TRUNCATE_DOCUMENT_ENCODING_MODE,
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(
            tokenizer.convert_ids_to_tokens(windows[0].input_ids),
            ["[CLS]", "Title:", "Alpha", "Abstract:", "a1", "a2", "a3", "[SEP]"],
        )

    def test_dense_vector_to_term_weights_returns_descending_token_weights(self) -> None:
        tokenizer = _FakeTokenizer()
        alpha_id, beta_id, gamma_id = tokenizer.encode("alpha beta gamma")
        vector = torch.zeros(2000, dtype=torch.float32)
        vector[alpha_id] = 0.7
        vector[beta_id] = 1.2
        vector[gamma_id] = 0.3
        result = dense_vector_to_term_weights(
            vector,
            output_space=OutputSpaceSpec.from_alignment(
                vocab_size=2000,
                compact_head_alignment="token_ids",
            ),
            tokenizer=tokenizer,
            exclude_output_ids=[],
            min_weight=0.5,
            top_k=2,
        )
        self.assertEqual(list(result.keys()), ["beta", "alpha"])
        self.assertAlmostEqual(result["beta"], 1.2, places=6)
        self.assertAlmostEqual(result["alpha"], 0.7, places=6)

    def test_local_parquet_patent_source_fetches_matching_documents(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            self.fail(f"pyarrow must be available for this test: {exc}")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet_path = root / "patent_us_docs_slice00of24-00000.parquet"
            table = pa.table(
                {
                    "doc_id": ["US1", "US2", "US3"],
                    "title": ["Title 1", "Title 2", "Title 3"],
                    "abstract": ["Abstract 1", None, "Abstract 3"],
                    "claims": ["Claims 1", "Claims 2", None],
                }
            )
            pq.write_table(table, parquet_path)
            source = LocalParquetPatentSource(corpus_path=root)
            result = source.fetch_documents(["US2", "US3"], batch_size=2, show_progress=False)
            self.assertEqual(sorted(result.keys()), ["US2", "US3"])
            self.assertEqual(result["US2"].title, "Title 2")
            self.assertEqual(result["US2"].abstract, "")
            self.assertEqual(result["US3"].claims, "")

    def test_select_contiguous_shard_preserves_order(self) -> None:
        values = list(range(10))
        self.assertEqual(select_contiguous_shard(values, shard_index=0, shard_count=3), [0, 1, 2])
        self.assertEqual(select_contiguous_shard(values, shard_index=1, shard_count=3), [3, 4, 5])
        self.assertEqual(select_contiguous_shard(values, shard_index=2, shard_count=3), [6, 7, 8, 9])

    def test_patent_window_batch_collator_flattens_document_windows(self) -> None:
        tokenizer = _FakeTokenizer()
        collator = PatentWindowBatchCollator(
            tokenizer=tokenizer,
            max_length=16,
            claim_overlap_tokens=0,
            document_encoding_mode="split_fields_windowed",
        )
        batch = collator(
            [
                PatentDocument(
                    doc_id="US1",
                    title="Title One",
                    abstract="Short abstract",
                    claims="Short claims",
                ),
                PatentDocument(
                    doc_id="US2",
                    title="Title Two",
                    abstract="",
                    claims="Only claims",
                ),
            ]
        )
        self.assertEqual(batch.doc_ids, ["US1", "US2"])
        self.assertGreaterEqual(int(batch.input_ids.shape[0]), 3)
        self.assertEqual(batch.window_doc_indices.tolist(), [0, 0, 1])
        self.assertEqual(batch.window_indices.tolist(), [0, 1, 0])

    def test_dense_vector_to_source_token_term_weights_groups_by_source_key(self) -> None:
        tokenizer = _FakeTokenizer()
        source_a_id, source_b_id, alpha_id, beta_id, gamma_id = tokenizer.encode(
            "sourceA sourceB alpha beta gamma"
        )
        vector = torch.zeros(2000, dtype=torch.float32)
        vector[alpha_id] = 0.7
        vector[beta_id] = 1.2
        vector[gamma_id] = 0.8
        winning_window_indices = torch.zeros(2000, dtype=torch.int32)
        winning_token_positions = torch.zeros(2000, dtype=torch.int32)
        winning_source_token_ids = torch.zeros(2000, dtype=torch.int32)
        winning_window_indices[alpha_id] = 1
        winning_window_indices[beta_id] = 1
        winning_window_indices[gamma_id] = 2
        winning_token_positions[alpha_id] = 4
        winning_token_positions[beta_id] = 4
        winning_token_positions[gamma_id] = 7
        winning_source_token_ids[alpha_id] = source_a_id
        winning_source_token_ids[beta_id] = source_a_id
        winning_source_token_ids[gamma_id] = source_b_id

        result = dense_vector_to_source_token_term_weights(
            vector,
            output_space=OutputSpaceSpec.from_alignment(
                vocab_size=2000,
                compact_head_alignment="token_ids",
            ),
            tokenizer=tokenizer,
            exclude_output_ids=[],
            min_weight=0.5,
            top_k=3,
            winning_window_indices=winning_window_indices,
            winning_token_positions=winning_token_positions,
            winning_source_token_ids=winning_source_token_ids,
        )

        source_a_key = build_source_token_key(
            window_index=1,
            token_position=4,
            source_token_id=source_a_id,
            source_token_text="sourceA",
        )
        source_b_key = build_source_token_key(
            window_index=2,
            token_position=7,
            source_token_id=source_b_id,
            source_token_text="sourceB",
        )
        self.assertEqual(set(result.keys()), {source_a_key, source_b_key})
        self.assertEqual(list(result[source_a_key].keys()), ["beta", "alpha"])
        self.assertAlmostEqual(result[source_a_key]["beta"], 1.2, places=6)
        self.assertAlmostEqual(result[source_a_key]["alpha"], 0.7, places=6)
        self.assertEqual(list(result[source_b_key].keys()), ["gamma"])
        self.assertAlmostEqual(result[source_b_key]["gamma"], 0.8, places=6)

    def test_update_aggregated_term_provenance_prefers_higher_scores_and_keeps_ties(self) -> None:
        first = update_aggregated_term_provenance(
            None,
            vector=torch.tensor([0.1, 0.6, 0.4], dtype=torch.float32),
            window_index=0,
            token_positions=torch.tensor([1, 2, 3], dtype=torch.int64),
            source_token_ids=torch.tensor([11, 12, 13], dtype=torch.int64),
        )

        updated = update_aggregated_term_provenance(
            first,
            vector=torch.tensor([0.5, 0.6, 0.2], dtype=torch.float32),
            window_index=1,
            token_positions=torch.tensor([4, 5, 6], dtype=torch.int64),
            source_token_ids=torch.tensor([21, 22, 23], dtype=torch.int64),
        )

        self.assertTrue(
            torch.equal(updated.vector, torch.tensor([0.5, 0.6, 0.4], dtype=torch.float32))
        )
        self.assertEqual(updated.winning_window_indices.tolist(), [1, 0, 0])
        self.assertEqual(updated.winning_token_positions.tolist(), [4, 2, 3])
        self.assertEqual(updated.winning_source_token_ids.tolist(), [21, 12, 13])

    def test_build_runtime_model_and_tokenizer_uses_checkpoint_loader(self) -> None:
        fake_model = object()
        fake_tokenizer = object()
        with (
            mock.patch(
                "src.preprocess.patent_splade_terms.apply_checkpoint_model_config",
                side_effect=lambda cfg, checkpoint_path, logger: cfg,
            ) as apply_cfg,
            mock.patch(
                "src.preprocess.patent_splade_terms.build_splade_model_with_checkpoint",
                return_value=fake_model,
            ) as build_model,
            mock.patch(
                "src.preprocess.patent_splade_terms.build_tokenizer",
                return_value=fake_tokenizer,
            ) as build_tokenizer_mock,
        ):
            model, tokenizer, cfg = build_runtime_model_and_tokenizer(
                model_name="local-model",
                checkpoint_path="/tmp/model.ckpt",
                use_cpu=True,
            )

        self.assertIs(model, fake_model)
        self.assertIs(tokenizer, fake_tokenizer)
        apply_cfg.assert_called_once()
        self.assertEqual(apply_cfg.call_args.kwargs["checkpoint_path"], "/tmp/model.ckpt")
        build_model.assert_called_once()
        self.assertEqual(
            build_model.call_args.kwargs["checkpoint_path"], "/tmp/model.ckpt"
        )
        build_tokenizer_mock.assert_called_once()
        self.assertEqual(
            build_tokenizer_mock.call_args.args[0], str(cfg.model.huggingface_name)
        )


if __name__ == "__main__":
    unittest.main()
