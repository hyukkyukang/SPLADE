import random
import unittest
from unittest.mock import patch

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from src.data.dataset.patent_us_in_batch import (
    PatentUsInBatchDataset,
    format_patent_title_abstract_claims,
)
from src.data.registry import resolve_dataset_builder


def build_dataset_cfg(
    *,
    hf_max_samples: int | None = None,
    hf_skip_samples: int = 0,
) -> DictConfig:
    return OmegaConf.create(
        {
            "name": "patent_us_in_batch",
            "type": "patent_us_in_batch",
            "split": "train",
            "corpus_split": "train",
            "hf_split": "train",
            "hf_name": "parquet",
            "hf_subset": None,
            "hf_cache_dir": None,
            "hf_max_samples": hf_max_samples,
            "hf_skip_samples": hf_skip_samples,
            "hf_data_files": {"train": "data/patent/train/stage1.parquet"},
            "query_corpus_hf_name": "parquet",
            "query_corpus_hf_cache_dir": None,
            "query_corpus_hf_data_files": {"train": ".cache/hf/patent-us-corpus/*.parquet"},
            "query_hf_data_files": None,
            "corpus_hf_data_files": {"train": ".cache/hf/patent-us-corpus/*.parquet"},
            "query_lookup_hf_name": None,
            "query_lookup_hf_subset": None,
            "query_lookup_hf_split": "train",
            "query_lookup_hf_cache_dir": None,
            "query_lookup_hf_data_files": None,
            "query_lookup_id_column": "query_id",
            "query_lookup_text_column": "text",
            "local_triplets_dir": None,
            "negative_sampling": {
                "strategy": "random",
                "top_k": None,
                "random_k": None,
                "random_pool": None,
            },
            "num_positives": 1,
            "num_negatives": 0,
            "query_subset_name": "train",
            "query_split_name": "train",
            "query_id_column": "query_id",
            "query_text_column": "query_text",
            "corpus_subset_name": "train",
            "corpus_split_name": "train",
            "corpus_id_column": "doc_id",
            "corpus_text_column": "abstract",
            "corpus_title_column": "title",
            "corpus_additional_text_columns": ["claims"],
        }
    )


class PatentUsInBatchDatasetTest(unittest.TestCase):
    def test_row_to_meta_item_keeps_inline_query_and_doc_ids(self) -> None:
        dataset = PatentUsInBatchDataset(cfg=build_dataset_cfg())

        meta_item = dataset._row_to_meta_item(
            {
                "query_id": "q1",
                "query_text": "How does the valve work?",
                "pos_doc_ids": ["US100", "US101"],
            },
            0,
            num_positives=1,
            num_negatives=0,
            rng=random.Random(0),
        )

        self.assertEqual(meta_item.qid, "q1")
        self.assertEqual(meta_item.query_text, "How does the valve work?")
        self.assertEqual(len(meta_item.pos_ids), 1)
        self.assertEqual(meta_item.neg_ids, [])

    def test_row_to_meta_item_requires_query_text(self) -> None:
        dataset = PatentUsInBatchDataset(cfg=build_dataset_cfg())

        with self.assertRaisesRegex(ValueError, "missing query_text"):
            dataset._row_to_meta_item(
                {
                    "query_id": "q1",
                    "query_text": "",
                    "pos_doc_ids": ["US100"],
                },
                0,
                num_positives=1,
                num_negatives=0,
                rng=random.Random(0),
            )

    def test_format_patent_title_abstract_claims_uses_requested_template(self) -> None:
        text = format_patent_title_abstract_claims(
            {
                "title": "Valve Assembly",
                "abstract": "A valve assembly with a hinged seal.",
                "claims": "1. A valve assembly...",
            }
        )

        self.assertEqual(
            text,
            "Title: Valve Assembly. Abstract: A valve assembly with a hinged seal.. "
            "Claims: 1. A valve assembly...",
        )

    def test_meta_dataset_respects_skip_and_max_window(self) -> None:
        dataset = PatentUsInBatchDataset(
            cfg=build_dataset_cfg(hf_max_samples=2, hf_skip_samples=1)
        )
        source_dataset: Dataset = Dataset.from_list(
            [
                {
                    "query_id": "q0",
                    "query_text": "query 0",
                    "pos_doc_ids": ["US000"],
                },
                {
                    "query_id": "q1",
                    "query_text": "query 1",
                    "pos_doc_ids": ["US001"],
                },
                {
                    "query_id": "q2",
                    "query_text": "query 2",
                    "pos_doc_ids": ["US002"],
                },
                {
                    "query_id": "q3",
                    "query_text": "query 3",
                    "pos_doc_ids": ["US003"],
                },
            ]
        )
        with patch.object(dataset, "_load_hf_dataset", return_value=source_dataset):
            meta_dataset: Dataset = dataset.meta_dataset

        self.assertEqual(len(meta_dataset), 2)
        self.assertEqual(
            [meta_dataset[0]["query_id"], meta_dataset[1]["query_id"]],
            ["q1", "q2"],
        )

    def test_registry_resolves_dataset_type(self) -> None:
        builder = resolve_dataset_builder(build_dataset_cfg())
        self.assertIs(builder, PatentUsInBatchDataset)


if __name__ == "__main__":
    unittest.main()
