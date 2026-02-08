import unittest

from src.data.pd_module.scoring import _labels_to_sources
from src.data.pd_module.scoring_hard_negatives import _merge_negatives_with_sources


class DocSourcesTest(unittest.TestCase):
    def test_merge_negatives_with_sources_dedup(self) -> None:
        value = {
            "bm25": ["d1", "d2"],
            "splade": ["d2", "d3"],
            "other": ["d4"],
        }
        doc_ids, sources = _merge_negatives_with_sources(value, ["bm25", "splade"])
        self.assertEqual(doc_ids, ["d1", "d2", "d3"])
        self.assertEqual(sources, ["bm25", "bm25", "splade"])
        self.assertEqual(len(doc_ids), len(sources))

    def test_labels_to_sources_with_doc_source(self) -> None:
        labels = [1.0, 0.0, -1.0]
        sources = _labels_to_sources(labels, neg_source="spladev2")
        self.assertEqual(sources, ["pos", "spladev2", "spladev2"])
        self.assertEqual(len(sources), len(labels))


if __name__ == "__main__":
    unittest.main()
