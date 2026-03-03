import random
import unittest

from omegaconf import DictConfig, OmegaConf

from src.data.dataset.msmarco_distill_scores import MSMARCODistillScoresDataset


def build_dataset_cfg(*, score_scores_column: str = "scores") -> DictConfig:
    return OmegaConf.create(
        {
            "name": "msmarco_multi_teacher_scores",
            "type": "msmarco_distill_scores",
            "split": "train",
            "hf_name": "dummy/dataset",
            "hf_subset": None,
            "hf_cache_dir": None,
            "hf_max_samples": None,
            "hf_skip_samples": 0,
            "hf_data_files": None,
            "query_corpus_hf_name": "sentence-transformers/msmarco",
            "query_corpus_hf_cache_dir": None,
            "query_corpus_hf_data_files": None,
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
            "score_scores_column": score_scores_column,
        }
    )


class TeacherScoreColumnResolutionTest(unittest.TestCase):
    def test_prefers_configured_score_column(self) -> None:
        dataset = MSMARCODistillScoresDataset(
            cfg=build_dataset_cfg(score_scores_column="teacher_scores")
        )
        row = {
            "query_id": "q1",
            "doc_ids": ["d_pos", "d_neg_1", "d_neg_2"],
            "labels": [1.0, 0.0, 0.0],
            "teacher_scores": [10.0, 7.0, 5.0],
            "scores": [1.0, 2.0, 3.0],
        }

        meta_item = dataset._row_to_meta_item(
            row,
            0,
            num_positives=1,
            num_negatives=2,
            rng=random.Random(0),
        )

        self.assertEqual(meta_item.pos_ids, ["d_pos"])
        self.assertEqual(meta_item.neg_ids, ["d_neg_1", "d_neg_2"])
        self.assertEqual(meta_item.pos_scores, [10.0])
        self.assertEqual(meta_item.neg_scores, [7.0, 5.0])

    def test_falls_back_to_teacher_scores_when_scores_missing(self) -> None:
        dataset = MSMARCODistillScoresDataset(
            cfg=build_dataset_cfg(score_scores_column="scores")
        )
        row = {
            "query_id": "q2",
            "doc_ids": ["d_pos", "d_neg"],
            "labels": [1.0, 0.0],
            "teacher_scores": [0.9, 0.1],
        }

        meta_item = dataset._row_to_meta_item(
            row,
            0,
            num_positives=1,
            num_negatives=1,
            rng=random.Random(0),
        )

        self.assertEqual(meta_item.pos_scores, [0.9])
        self.assertEqual(meta_item.neg_scores, [0.1])


if __name__ == "__main__":
    unittest.main()
