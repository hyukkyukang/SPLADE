import unittest

from script.preprocess.build_multi_teacher_scores_dataset import (
    _build_teacher_scores,
    _normalize_scores,
)


class BuildMultiTeacherScoresDatasetTest(unittest.TestCase):
    def test_min_max_normalize(self) -> None:
        scores = [2.0, 4.0, 6.0]
        normalized = _normalize_scores(scores, epsilon=1e-6)
        self.assertEqual(normalized, [0.0, 0.5, 1.0])

    def test_min_max_normalize_flat_values(self) -> None:
        scores = [5.0, 5.0, 5.0]
        normalized = _normalize_scores(scores, epsilon=1e-6)
        self.assertEqual(normalized, [0.0, 0.0, 0.0])

    def test_teacher_score_mean(self) -> None:
        columns = [
            [0.0, 1.0, 0.5],
            [1.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ]
        teacher_scores = _build_teacher_scores(columns)
        self.assertEqual(teacher_scores, [0.5, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
