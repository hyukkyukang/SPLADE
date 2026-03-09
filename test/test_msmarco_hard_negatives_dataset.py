import random
import unittest

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from src.data.dataset.msmarco_hard_negatives import MSMARCOHardNegativesDataset
from src.data.registry import resolve_dataset_builder


def build_dataset_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "name": "msmarco_hard_negatives",
            "type": "msmarco_hard_negatives",
            "split": "train",
            "hf_name": "sentence-transformers/msmarco-hard-negatives",
            "hf_subset": None,
            "hf_cache_dir": None,
            "hf_max_samples": None,
            "hf_skip_samples": 0,
            "hf_data_files": {"train": "msmarco-hard-negatives.jsonl.gz"},
            "query_corpus_hf_name": "sentence-transformers/msmarco",
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
                "strategy": "random",
                "top_k": None,
                "random_k": None,
                "random_pool": None,
            },
            "num_positives": 1,
            "num_negatives": 4,
            "query_subset_name": "queries",
            "query_split_name": "train",
            "query_id_column": "query_id",
            "query_text_column": "query",
            "corpus_subset_name": "corpus",
            "corpus_split_name": "train",
            "corpus_id_column": "passage_id",
            "corpus_text_column": "passage",
            "corpus_title_column": None,
            "hard_negative_selection": {
                "model_priority": [
                    "dense_a",
                    "dense_b",
                    "dense_c",
                    "bm25",
                ],
                "deprioritized_models": ["bm25"],
                "append_unlisted_models": True,
                "drop_positive_overlaps": True,
                "dedupe": True,
                "require_negatives": True,
            },
        }
    )


class MSMARCOHardNegativesDatasetTest(unittest.TestCase):
    def test_row_to_meta_item_selects_dense_models_before_bm25(self) -> None:
        dataset = MSMARCOHardNegativesDataset(cfg=build_dataset_cfg())
        row = {
            "qid": 100,
            "pos": [1000],
            "neg": {
                "bm25": [9001, 9002],
                "dense_b": [2001, 2002],
                "dense_c": [3001, 3002],
            },
        }

        meta_item = dataset._row_to_meta_item(
            row,
            0,
            num_positives=1,
            num_negatives=3,
            rng=random.Random(0),
        )

        self.assertEqual(meta_item.qid, "100")
        self.assertEqual(meta_item.pos_ids, ["1000"])
        self.assertEqual(meta_item.neg_ids, ["2001", "2002", "3001"])

    def test_row_to_meta_item_spills_to_bm25_when_needed(self) -> None:
        dataset = MSMARCOHardNegativesDataset(cfg=build_dataset_cfg())
        row = {
            "qid": 101,
            "pos": [1001],
            "neg": {
                "dense_b": [2101],
                "bm25": [9101, 9102],
            },
        }

        meta_item = dataset._row_to_meta_item(
            row,
            0,
            num_positives=1,
            num_negatives=3,
            rng=random.Random(0),
        )

        self.assertEqual(meta_item.neg_ids, ["2101", "9101", "9102"])

    def test_row_to_meta_item_filters_positive_overlaps_and_dedupes(self) -> None:
        dataset = MSMARCOHardNegativesDataset(cfg=build_dataset_cfg())
        row = {
            "qid": 102,
            "pos": [1100, 1101],
            "neg": {
                "dense_a": [1100, 2200, 2201],
                "dense_b": [2201, 2300],
                "bm25": [2200, 2400],
            },
        }

        meta_item = dataset._row_to_meta_item(
            row,
            0,
            num_positives=1,
            num_negatives=10,
            rng=random.Random(0),
        )

        self.assertEqual(meta_item.pos_ids, ["1100"])
        self.assertEqual(meta_item.neg_ids, ["2200", "2201", "2300", "2400"])

    def test_row_to_meta_item_requires_positive_ids(self) -> None:
        dataset = MSMARCOHardNegativesDataset(cfg=build_dataset_cfg())
        row = {"qid": "q_missing", "pos": [], "neg": {"dense_a": [1, 2]}}

        with self.assertRaisesRegex(ValueError, "missing positive ids"):
            dataset._row_to_meta_item(
                row,
                0,
                num_positives=1,
                num_negatives=1,
                rng=random.Random(0),
            )

    def test_registry_resolves_dataset_type(self) -> None:
        builder = resolve_dataset_builder(build_dataset_cfg())
        self.assertIs(builder, MSMARCOHardNegativesDataset)

    def test_resolve_meta_dataset_filters_rows_missing_positive_ids(self) -> None:
        class _StubMSMARCOHardNegativesDataset(MSMARCOHardNegativesDataset):
            def __init__(self, cfg: DictConfig, rows: list[dict[str, object]]) -> None:
                self._rows: list[dict[str, object]] = rows
                super().__init__(cfg)

            def _load_hf_dataset(
                self,
                hf_name: str,
                hf_subset: str | None,
                split: str,
                cache_dir: str | None,
                data_files: dict[str, object] | None,
            ) -> Dataset:
                _ = hf_name, hf_subset, split, cache_dir, data_files
                return Dataset.from_list(self._rows)

        rows = [
            {"qid": "q1", "pos": [1], "neg": {"dense_a": [2]}},
            {"qid": "q2", "pos": [], "neg": {"dense_a": [3]}},
            {"qid": "q3", "pos": [4], "neg": {"dense_a": [5]}},
        ]
        dataset = _StubMSMARCOHardNegativesDataset(build_dataset_cfg(), rows)
        qids = [str(row["qid"]) for row in dataset.meta_dataset]
        self.assertEqual(qids, ["q1", "q3"])

    def test_resolve_meta_dataset_filters_rows_without_usable_negatives(self) -> None:
        class _StubMSMARCOHardNegativesDataset(MSMARCOHardNegativesDataset):
            def __init__(self, cfg: DictConfig, rows: list[dict[str, object]]) -> None:
                self._rows: list[dict[str, object]] = rows
                super().__init__(cfg)

            def _load_hf_dataset(
                self,
                hf_name: str,
                hf_subset: str | None,
                split: str,
                cache_dir: str | None,
                data_files: dict[str, object] | None,
            ) -> Dataset:
                _ = hf_name, hf_subset, split, cache_dir, data_files
                return Dataset.from_list(self._rows)

        rows = [
            {"qid": "q1", "pos": [1], "neg": {"dense_a": [1]}},
            {"qid": "q2", "pos": [2], "neg": {"dense_a": [3]}},
        ]
        dataset = _StubMSMARCOHardNegativesDataset(build_dataset_cfg(), rows)
        qids = [str(row["qid"]) for row in dataset.meta_dataset]
        self.assertEqual(qids, ["q2"])

    def test_resolve_meta_dataset_materializes_precomputed_fields(self) -> None:
        class _StubMSMARCOHardNegativesDataset(MSMARCOHardNegativesDataset):
            def __init__(self, cfg: DictConfig, rows: list[dict[str, object]]) -> None:
                self._rows: list[dict[str, object]] = rows
                super().__init__(cfg)

            def _load_hf_dataset(
                self,
                hf_name: str,
                hf_subset: str | None,
                split: str,
                cache_dir: str | None,
                data_files: dict[str, object] | None,
            ) -> Dataset:
                _ = hf_name, hf_subset, split, cache_dir, data_files
                return Dataset.from_list(self._rows)

        rows = [
            {
                "qid": "q1",
                "pos": [1],
                "neg": {"dense_a": [10, 11], "bm25": [20]},
            }
        ]
        dataset = _StubMSMARCOHardNegativesDataset(build_dataset_cfg(), rows)
        row = dict(dataset.meta_dataset[0])
        self.assertEqual(row["__splade_hn_qid"], "q1")
        self.assertEqual(row["__splade_hn_pos_ids"], ["1"])
        self.assertEqual(row["__splade_hn_neg_ids"], ["10", "11", "20"])

    def test_row_to_meta_item_prefers_precomputed_negative_ids(self) -> None:
        dataset = MSMARCOHardNegativesDataset(cfg=build_dataset_cfg())
        row = {
            "qid": "q_precomputed",
            "pos": [100],
            "neg": {"dense_a": [200, 201, 202]},
            "__splade_hn_neg_ids": ["900", "901", "902"],
        }

        meta_item = dataset._row_to_meta_item(
            row,
            0,
            num_positives=1,
            num_negatives=2,
            rng=random.Random(0),
        )

        self.assertEqual(meta_item.neg_ids, ["900", "901"])


if __name__ == "__main__":
    unittest.main()
