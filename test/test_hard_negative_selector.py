import unittest

from src.data.dataset.hard_negative_selector import (
    HardNegativeSelectionSettings,
    resolve_model_order,
    select_hard_negative_doc_ids,
)


class HardNegativeSelectorTest(unittest.TestCase):
    def test_resolve_model_order_moves_deprioritized_models_to_end(self) -> None:
        neg_map = {
            "bm25": [100],
            "dense_a": [10, 11],
            "dense_b": [20, 21],
        }
        settings = HardNegativeSelectionSettings(
            model_priority=("bm25", "dense_a", "dense_b"),
            deprioritized_models=("bm25",),
            append_unlisted_models=True,
        )

        resolved = resolve_model_order(neg_map, settings)

        self.assertEqual(resolved, ["dense_a", "dense_b", "bm25"])

    def test_selection_spills_to_next_model_until_target_count(self) -> None:
        neg_value = {
            "dense_a": [10, 11],
            "dense_b": [20, 21],
            "bm25": [100, 101],
        }
        settings = HardNegativeSelectionSettings(
            model_priority=("dense_a", "dense_b", "bm25"),
            deprioritized_models=("bm25",),
            append_unlisted_models=True,
        )

        selected = select_hard_negative_doc_ids(
            neg_value,
            positive_doc_ids=["1"],
            target_count=5,
            settings=settings,
        )

        self.assertEqual(selected, ["10", "11", "20", "21", "100"])

    def test_selection_dedupes_and_skips_positive_overlaps(self) -> None:
        neg_value = {
            "dense_a": [10, 11, 12],
            "dense_b": [11, 13],
            "bm25": [12, 14],
        }
        settings = HardNegativeSelectionSettings(
            model_priority=("dense_a", "dense_b", "bm25"),
            deprioritized_models=("bm25",),
            append_unlisted_models=True,
            drop_positive_overlaps=True,
            dedupe=True,
        )

        selected = select_hard_negative_doc_ids(
            neg_value,
            positive_doc_ids=["10", "14"],
            target_count=5,
            settings=settings,
        )

        self.assertEqual(selected, ["11", "12", "13"])

    def test_selection_handles_non_mapping_neg_value(self) -> None:
        settings = HardNegativeSelectionSettings(
            model_priority=(),
            deprioritized_models=("bm25",),
        )

        selected = select_hard_negative_doc_ids(
            [7, 8, 9],
            positive_doc_ids=[],
            target_count=2,
            settings=settings,
        )

        self.assertEqual(selected, ["7", "8"])


if __name__ == "__main__":
    unittest.main()
