from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


plot_term_count_histogram = _load_module(
    "plot_term_count_histogram",
    "script/preprocess/patent/plot_term_count_histogram.py",
)
truncate_patent_term_export = _load_module(
    "truncate_patent_term_export",
    "script/preprocess/patent/truncate_patent_term_export.py",
)


class PatentTermExportScriptsTest(unittest.TestCase):
    def test_count_terms_per_doc_supports_source_token_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "terms.json"
            input_path.write_text(
                json.dumps(
                    {
                        "doc-flat": {"alpha": 1.0, "beta": 0.5},
                        "doc-source": {
                            "w0:p1:id100:foo": {"term_a": 0.9, "term_c": 0.4},
                            "w0:p3:id101:bar": {"term_b": 0.8},
                        },
                        "doc-empty": {},
                    }
                ),
                encoding="utf-8",
            )

            stats = plot_term_count_histogram.count_terms_per_doc(input_path)
            summary = plot_term_count_histogram.build_summary(input_path, stats)

        self.assertEqual(stats.counts, [2, 3, 0])
        self.assertEqual(stats.payload_mode_counts["flat_terms"], 1)
        self.assertEqual(stats.payload_mode_counts["source_token_terms"], 1)
        self.assertEqual(stats.payload_mode_counts["empty"], 1)
        self.assertEqual(stats.source_bucket_counts, [0, 2, 0])
        self.assertEqual(summary["payload_mode"], "mixed")
        self.assertEqual(summary["source_bucket_stats"]["max_source_buckets_per_doc"], 2)

    def test_truncate_term_payload_keeps_global_top_k_for_source_token_payloads(self) -> None:
        payload = {
            "w0:p1:id100:foo": {"term_a": 0.9, "term_c": 0.6},
            "w0:p3:id101:bar": {"term_b": 0.8},
        }

        truncated = truncate_patent_term_export.truncate_term_payload(payload, top_k=2)

        self.assertEqual(
            truncated,
            {
                "w0:p1:id100:foo": {"term_a": 0.9},
                "w0:p3:id101:bar": {"term_b": 0.8},
            },
        )

    def test_truncate_term_payload_supports_flat_payloads(self) -> None:
        payload = {"alpha": 0.2, "beta": 0.9, "gamma": 0.5}

        truncated = truncate_patent_term_export.truncate_term_payload(payload, top_k=2)

        self.assertEqual(truncated, {"beta": 0.9, "gamma": 0.5})


if __name__ == "__main__":
    unittest.main()
