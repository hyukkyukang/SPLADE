import unittest

from script.evaluate import _extract_benchmark_argument


class EvaluateBenchmarkRoutingTest(unittest.TestCase):
    def test_no_benchmark_keeps_argv(self) -> None:
        benchmark_name, forwarded_argv = _extract_benchmark_argument(
            ["script/evaluate.py", "model=splade_v2_pp"]
        )
        self.assertIsNone(benchmark_name)
        self.assertEqual(forwarded_argv, ["script/evaluate.py", "model=splade_v2_pp"])

    def test_extracts_split_benchmark_argument(self) -> None:
        benchmark_name, forwarded_argv = _extract_benchmark_argument(
            [
                "script/evaluate.py",
                "--benchmark",
                "nanobeir",
                "testing.checkpoint_path=/tmp/model.ckpt",
            ]
        )
        self.assertEqual(benchmark_name, "nanobeir")
        self.assertEqual(
            forwarded_argv,
            ["script/evaluate.py", "testing.checkpoint_path=/tmp/model.ckpt"],
        )

    def test_extracts_equals_benchmark_argument(self) -> None:
        benchmark_name, forwarded_argv = _extract_benchmark_argument(
            [
                "script/evaluate.py",
                "--benchmark=mteb",
                "nanobeir.datasets=[msmarco]",
            ]
        )
        self.assertEqual(benchmark_name, "mteb")
        self.assertEqual(
            forwarded_argv,
            ["script/evaluate.py", "nanobeir.datasets=[msmarco]"],
        )

    def test_extracts_true_mteb_benchmark_argument(self) -> None:
        benchmark_name, forwarded_argv = _extract_benchmark_argument(
            [
                "script/evaluate.py",
                "--benchmark=true_mteb",
                "mteb.tasks=[NFCorpus,SciFact]",
            ]
        )
        self.assertEqual(benchmark_name, "true_mteb")
        self.assertEqual(
            forwarded_argv,
            ["script/evaluate.py", "mteb.tasks=[NFCorpus,SciFact]"],
        )

    def test_raises_on_missing_benchmark_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing value for --benchmark"):
            _extract_benchmark_argument(["script/evaluate.py", "--benchmark"])


if __name__ == "__main__":
    unittest.main()
