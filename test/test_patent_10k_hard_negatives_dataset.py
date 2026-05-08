import random
import unittest
from unittest.mock import patch

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from src.data.dataset.patent_10k_hard_negatives import (
    Patent10KHardNegativesDataset,
)
from src.data.registry import resolve_dataset_builder


def build_dataset_cfg(
    *,
    hf_max_samples: int | None = None,
    hf_skip_samples: int = 0,
    negative_strategy: str = "topk_plus_random",
) -> DictConfig:
    return OmegaConf.create(
        {
            "name": "patent_10k_hard_negatives",
            "type": "patent_10k_hard_negatives",
            "split": "train",
            "hf_name": "Hyukkyu/patent-10k",
            "hf_subset": "default",
            "hf_cache_dir": None,
            "hf_max_samples": hf_max_samples,
            "hf_skip_samples": hf_skip_samples,
            "hf_data_files": None,
            "query_corpus_hf_name": None,
            "query_corpus_hf_cache_dir": None,
            "query_corpus_hf_data_files": None,
            "query_lookup_hf_name": None,
            "query_lookup_hf_subset": None,
            "query_lookup_hf_split": "train",
            "query_lookup_hf_cache_dir": None,
            "query_lookup_hf_data_files": None,
            "query_lookup_id_column": "query_id",
            "query_lookup_text_column": "text",
            "local_triplets_dir": None,
            "negative_sampling": {
                "strategy": negative_strategy,
                "top_k": 2,
                "random_k": 2,
                "random_pool": 4,
            },
            "num_positives": 1,
            "num_negatives": 4,
            "query_subset_name": "default",
            "query_split_name": "train",
            "query_id_column": "query_id",
            "query_text_column": "query_text",
            "corpus_subset_name": "default",
            "corpus_split_name": "train",
            "corpus_id_column": "positive_node_id",
            "corpus_text_column": "positive_text",
            "corpus_title_column": None,
        }
    )


def build_row() -> dict[str, object]:
    return {
        "query_id": "q1",
        "query_text": "claim limitation about thermal control",
        "positive_node_id": "pos-node",
        "positive_text": "positive chunk text",
        "hard_negative_node_ids": [
            "neg-1",
            "neg-2",
            "neg-3",
            "neg-4",
            "neg-5",
            "neg-2",
        ],
        "hard_negative_texts": [
            "negative text 1",
            "negative text 2",
            "negative text 3",
            "negative text 4",
            "negative text 5",
            "duplicate negative text 2",
        ],
        "hard_negative_ranks": [1, 2, 3, 4, 5, 6],
    }


class Patent10KHardNegativesDatasetTest(unittest.TestCase):
    def test_row_to_meta_item_selects_ranked_negatives(self) -> None:
        dataset = Patent10KHardNegativesDataset(
            cfg=build_dataset_cfg(negative_strategy="topk")
        )

        meta_item = dataset._row_to_meta_item(
            build_row(),
            0,
            num_positives=1,
            num_negatives=3,
            rng=random.Random(0),
        )

        self.assertEqual(meta_item.qid, "q1")
        self.assertEqual(meta_item.query_text, "claim limitation about thermal control")
        self.assertEqual(meta_item.pos_ids, ["pos-node"])
        self.assertEqual(meta_item.pos_texts, ["positive chunk text"])
        self.assertEqual(meta_item.neg_ids, ["neg-1", "neg-2", "neg-3"])
        self.assertEqual(
            meta_item.neg_texts,
            ["negative text 1", "negative text 2", "negative text 3"],
        )

    def test_row_to_meta_item_topk_plus_random_uses_ranked_pool(self) -> None:
        dataset = Patent10KHardNegativesDataset(cfg=build_dataset_cfg())

        meta_item = dataset._row_to_meta_item(
            build_row(),
            0,
            num_positives=1,
            num_negatives=4,
            rng=random.Random(0),
        )

        self.assertEqual(meta_item.neg_ids[:2], ["neg-1", "neg-2"])
        self.assertEqual(len(meta_item.neg_ids), 4)
        self.assertTrue(set(meta_item.neg_ids).issubset({"neg-1", "neg-2", "neg-3", "neg-4"}))
        self.assertEqual(len(set(meta_item.neg_ids)), 4)

    def test_row_to_meta_item_requires_inline_text_fields(self) -> None:
        dataset = Patent10KHardNegativesDataset(cfg=build_dataset_cfg())
        row = build_row()
        row["positive_text"] = ""

        with self.assertRaisesRegex(ValueError, "missing positive_text"):
            dataset._row_to_meta_item(
                row,
                0,
                num_positives=1,
                num_negatives=1,
                rng=random.Random(0),
            )

    def test_meta_dataset_respects_skip_and_max_window(self) -> None:
        dataset = Patent10KHardNegativesDataset(
            cfg=build_dataset_cfg(hf_max_samples=2, hf_skip_samples=1)
        )
        source_rows = [
            {**build_row(), "query_id": "q0"},
            {**build_row(), "query_id": "q1"},
            {**build_row(), "query_id": "q2"},
            {**build_row(), "query_id": "q3"},
        ]
        source_dataset: Dataset = Dataset.from_list(source_rows)
        with patch.object(dataset, "_load_hf_dataset", return_value=source_dataset):
            meta_dataset: Dataset = dataset.meta_dataset

        self.assertEqual(len(meta_dataset), 2)
        self.assertEqual(
            [meta_dataset[0]["query_id"], meta_dataset[1]["query_id"]],
            ["q1", "q2"],
        )

    def test_meta_dataset_uses_hf_split_override(self) -> None:
        cfg = build_dataset_cfg()
        cfg.split = "train"
        cfg.hf_split = "validation"
        dataset = Patent10KHardNegativesDataset(cfg=cfg)
        source_dataset: Dataset = Dataset.from_list([{**build_row(), "query_id": "q0"}])

        with patch.object(dataset, "_load_hf_dataset", return_value=source_dataset) as mocked_load:
            _ = dataset.meta_dataset

        self.assertEqual(mocked_load.call_args.kwargs["split"], "validation")

    def test_registry_resolves_dataset_type(self) -> None:
        builder = resolve_dataset_builder(build_dataset_cfg())
        self.assertIs(builder, Patent10KHardNegativesDataset)


if __name__ == "__main__":
    unittest.main()
