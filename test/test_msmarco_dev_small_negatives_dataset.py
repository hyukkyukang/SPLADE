import random
import unittest
from unittest.mock import patch

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from src.data.dataset.msmarco_dev_small_negatives import (
    MSMARCODevSmallNegativesDataset,
)
from src.data.registry import resolve_dataset_builder


def build_dataset_cfg(
    *,
    hf_max_samples: int | None = None,
    hf_skip_samples: int = 0,
    query_lookup_hf_name: str | None = None,
) -> DictConfig:
    return OmegaConf.create(
        {
            "name": "msmarco_dev_small_negatives",
            "type": "msmarco_dev_small_negatives",
            "split": "colbertv2",
            "hf_name": "antoinelouis/msmarco-dev-small-negatives",
            "hf_subset": "negatives",
            "hf_cache_dir": None,
            "hf_max_samples": hf_max_samples,
            "hf_skip_samples": hf_skip_samples,
            "hf_data_files": None,
            "query_corpus_hf_name": "sentence-transformers/msmarco",
            "query_corpus_hf_cache_dir": None,
            "query_corpus_hf_data_files": None,
            "query_lookup_hf_name": query_lookup_hf_name,
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
            "query_subset_name": "queries",
            "query_split_name": "train",
            "query_id_column": "query_id",
            "query_text_column": "query",
            "corpus_subset_name": "corpus",
            "corpus_split_name": "train",
            "corpus_id_column": "passage_id",
            "corpus_text_column": "passage",
            "corpus_title_column": None,
        }
    )


class MSMARCODevSmallNegativesDatasetTest(unittest.TestCase):
    def test_row_to_meta_item_uses_ordered_top_k_negatives(self) -> None:
        dataset = MSMARCODevSmallNegativesDataset(cfg=build_dataset_cfg())
        row = {
            "qid": "q1",
            "pos": [101, 102],
            "neg": [201, 202, 203],
        }

        meta_item = dataset._row_to_meta_item(
            row,
            0,
            num_positives=1,
            num_negatives=2,
            rng=random.Random(0),
        )

        self.assertEqual(meta_item.qid, "q1")
        self.assertEqual(meta_item.pos_ids, ["101"])
        self.assertEqual(meta_item.neg_ids, ["201", "202"])

    def test_row_to_meta_item_accepts_scalar_positive(self) -> None:
        dataset = MSMARCODevSmallNegativesDataset(cfg=build_dataset_cfg())
        row = {
            "qid": "q2",
            "pos": 1001,
            "neg": [2001, 2002],
        }

        meta_item = dataset._row_to_meta_item(
            row,
            0,
            num_positives=1,
            num_negatives=1,
            rng=random.Random(0),
        )

        self.assertEqual(meta_item.pos_ids, ["1001"])
        self.assertEqual(meta_item.neg_ids, ["2001"])

    def test_row_to_meta_item_requires_positive_ids(self) -> None:
        dataset = MSMARCODevSmallNegativesDataset(cfg=build_dataset_cfg())
        row = {"qid": "q3", "pos": [], "neg": [3001]}

        with self.assertRaisesRegex(ValueError, "missing positive ids"):
            dataset._row_to_meta_item(
                row,
                0,
                num_positives=1,
                num_negatives=1,
                rng=random.Random(0),
            )

    def test_meta_dataset_respects_skip_and_max_window(self) -> None:
        dataset = MSMARCODevSmallNegativesDataset(
            cfg=build_dataset_cfg(hf_max_samples=2, hf_skip_samples=1)
        )
        source_rows = [
            {"qid": "q0", "pos": [0], "neg": [100]},
            {"qid": "q1", "pos": [1], "neg": [101]},
            {"qid": "q2", "pos": [2], "neg": [102]},
            {"qid": "q3", "pos": [3], "neg": [103]},
        ]
        source_dataset: Dataset = Dataset.from_list(source_rows)
        with patch.object(dataset, "_load_hf_dataset", return_value=source_dataset):
            meta_dataset: Dataset = dataset.meta_dataset

        self.assertEqual(len(meta_dataset), 2)
        self.assertEqual([meta_dataset[0]["qid"], meta_dataset[1]["qid"]], ["q1", "q2"])

    def test_registry_resolves_dataset_type(self) -> None:
        builder = resolve_dataset_builder(build_dataset_cfg())
        self.assertIs(builder, MSMARCODevSmallNegativesDataset)

    def test_lookup_query_texts_uses_optional_lookup_dataset(self) -> None:
        dataset = MSMARCODevSmallNegativesDataset(
            cfg=build_dataset_cfg(query_lookup_hf_name="dummy/query-lookup")
        )
        meta_rows = Dataset.from_list([{"qid": "q0", "pos": [1], "neg": [2]}])
        lookup_rows = Dataset.from_list(
            [
                {"query_id": "q1", "text": "one"},
                {"query_id": "q2", "text": "two"},
            ]
        )

        def _mock_load(
            hf_name: str,
            hf_subset: str | None,
            split: str,
            cache_dir: str | None,
            data_files: dict[str, str] | None,
        ) -> Dataset:
            _ = hf_subset
            _ = split
            _ = cache_dir
            _ = data_files
            if hf_name == "dummy/query-lookup":
                return lookup_rows
            return meta_rows

        with patch.object(dataset, "_load_hf_dataset", side_effect=_mock_load):
            resolved = dataset.lookup_query_texts(["q2", "q_missing", "q1"])

        self.assertEqual(resolved, {"q2": "two", "q1": "one"})


if __name__ == "__main__":
    unittest.main()
