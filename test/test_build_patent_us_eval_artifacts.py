import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from script.preprocess.patent.build_patent_us_eval_artifacts import (
    collect_benchmark_qrels,
    collect_qa_tsv_artifacts,
    load_query_texts_from_hf,
    load_query_texts_from_parquet,
    write_eval_artifacts,
)


class BuildPatentUsEvalArtifactsTest(unittest.TestCase):
    def test_collect_benchmark_qrels_expands_all_label_ids(self) -> None:
        ordered_query_ids, qrels_rows, metadata = collect_benchmark_qrels(
            [
                {"question_id": "US100", "label_id": ["US101", "US102", "US101"]},
                {"question_id": "US200", "label_id": ["US201"]},
                {"question_id": "US300", "label_id": []},
            ]
        )

        self.assertEqual(ordered_query_ids, ["US100", "US200"])
        self.assertEqual(
            qrels_rows,
            [
                ("US100", "US101", 1.0),
                ("US100", "US102", 1.0),
                ("US200", "US201", 1.0),
            ],
        )
        self.assertEqual(metadata["empty_label_rows"], 1)
        self.assertEqual(metadata["duplicate_qrel_pairs"], 1)

    def test_load_query_texts_from_parquet_uses_patent_template(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            corpus_path = Path(tmp_dir) / "patent_docs.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "doc_id": "US100",
                            "title": "Valve Assembly",
                            "abstract": "A valve assembly with a hinged seal.",
                            "claims": "1. A valve assembly...",
                            "description": "Detailed valve description.",
                        },
                        {
                            "doc_id": "US101",
                            "title": "Widget Controller",
                            "abstract": "A controller that manages widget timing.",
                            "claims": "1. A widget controller...",
                            "description": "Detailed controller description.",
                        },
                    ]
                ),
                corpus_path,
            )

            query_text_by_id = load_query_texts_from_parquet(
                corpus_glob=str(corpus_path),
                query_ids=["US101"],
            )

            self.assertEqual(
                query_text_by_id,
                {
                    "US101": "Title: Widget Controller\n"
                    "Abstract: A controller that manages widget timing.\n"
                    "Claims: 1. A widget controller...\n"
                    "Description: Detailed controller description."
                },
            )

    def test_load_query_texts_from_hf_streams_corpus_lookup(self) -> None:
        streamed_rows = [
            {
                "doc_id": "US100",
                "title": "Valve Assembly",
                "abstract": "A valve assembly with a hinged seal.",
                "claims": "1. A valve assembly...",
                "description": "Detailed valve description.",
            },
            {
                "doc_id": "US101",
                "title": "Widget Controller",
                "abstract": "A controller that manages widget timing.",
                "claims": "1. A widget controller...",
                "description": "Detailed controller description.",
            },
        ]

        with patch(
            "script.preprocess.patent.build_patent_us_eval_artifacts.load_dataset",
            return_value=streamed_rows,
        ) as mocked_load_dataset:
            query_text_by_id = load_query_texts_from_hf(
                corpus_repo="Hyukkyu/patent-us-corpus-small",
                query_ids=["US101"],
            )

        self.assertEqual(
            query_text_by_id,
            {
                "US101": "Title: Widget Controller\n"
                "Abstract: A controller that manages widget timing.\n"
                "Claims: 1. A widget controller...\n"
                "Description: Detailed controller description."
            },
        )
        _, kwargs = mocked_load_dataset.call_args
        self.assertTrue(kwargs["streaming"])

    def test_load_query_texts_from_parquet_supports_dpr_title_abstract_template(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            corpus_path = Path(tmp_dir) / "patent_docs.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "doc_id": "US101",
                            "title": "Widget Controller",
                            "abstract": "A controller that manages widget timing.",
                            "claims": "1. A widget controller...",
                            "description": "Detailed controller description.",
                        }
                    ]
                ),
                corpus_path,
            )

            query_text_by_id = load_query_texts_from_parquet(
                corpus_glob=str(corpus_path),
                query_ids=["US101"],
                query_text_template="plain_title_abstract",
            )

            self.assertEqual(
                query_text_by_id,
                {"US101": "Widget Controller\nA controller that manages widget timing."},
            )

    def test_write_eval_artifacts_filters_missing_queries_and_writes_multi_qrels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "eval"

            metadata = write_eval_artifacts(
                ordered_query_ids=["US100", "US200"],
                qrels_rows=[
                    ("US100", "US101", 1.0),
                    ("US100", "US102", 1.0),
                    ("US200", "US201", 1.0),
                ],
                query_text_by_id={
                    "US100": "Title: Query 100",
                },
                output_dir=output_dir,
                metadata={"benchmark_repo": "Hyukkyu/patent-us-small"},
            )

            queries = pq.read_table(output_dir / "queries.parquet").to_pylist()
            qrels = pq.read_table(output_dir / "qrels.parquet").to_pylist()

            self.assertEqual(
                queries,
                [{"query_id": "US100", "text": "Title: Query 100"}],
            )
            self.assertEqual(
                qrels,
                [
                    {"query_id": "US100", "doc_id": "US101", "score": 1.0},
                    {"query_id": "US100", "doc_id": "US102", "score": 1.0},
                ],
            )
            self.assertEqual(metadata["query_count"], 1)
            self.assertEqual(metadata["qrels_count"], 2)
            self.assertEqual(metadata["missing_query_count"], 1)

    def test_collect_qa_tsv_artifacts_uses_raw_queries_and_query_ids(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            qa_path = Path(tmp_dir) / "qa.tsv"
            qa_path.write_text(
                "\n".join(
                    [
                        "How do I seal a valve?\tUS101\tUS900",
                        "How do I seal a valve?\tUS102\tUS900",
                        "How do I drive a widget?\tUS201\tUS901",
                    ]
                ),
                encoding="utf-8",
            )

            ordered_query_ids, qrels_rows, query_text_by_id, metadata = (
                collect_qa_tsv_artifacts(
                    qa_tsv=qa_path,
                    query_column=0,
                    label_column=1,
                    query_id_column=2,
                )
            )

            self.assertEqual(ordered_query_ids, ["US900", "US901"])
            self.assertEqual(
                qrels_rows,
                [
                    ("US900", "US101", 1.0),
                    ("US900", "US102", 1.0),
                    ("US901", "US201", 1.0),
                ],
            )
            self.assertEqual(
                query_text_by_id,
                {
                    "US900": "How do I seal a valve?",
                    "US901": "How do I drive a widget?",
                },
            )
            self.assertEqual(metadata["generated_query_ids"], 0)
            self.assertEqual(metadata["conflicting_query_text_rows"], 0)

    def test_collect_qa_tsv_artifacts_generates_query_ids_and_splits_labels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            qa_path = Path(tmp_dir) / "qa.tsv"
            qa_path.write_text(
                "\n".join(
                    [
                        "query\tlabels",
                        "Valve query\tUS101|US102",
                        "Widget query\tUS201",
                    ]
                ),
                encoding="utf-8",
            )

            ordered_query_ids, qrels_rows, query_text_by_id, metadata = (
                collect_qa_tsv_artifacts(
                    qa_tsv=qa_path,
                    query_column=0,
                    label_column=1,
                    query_id_column=None,
                    label_separator="|",
                    has_header=True,
                )
            )

            self.assertEqual(ordered_query_ids, ["q0", "q1"])
            self.assertEqual(
                qrels_rows,
                [
                    ("q0", "US101", 1.0),
                    ("q0", "US102", 1.0),
                    ("q1", "US201", 1.0),
                ],
            )
            self.assertEqual(
                query_text_by_id,
                {
                    "q0": "Valve query",
                    "q1": "Widget query",
                },
            )
            self.assertEqual(metadata["generated_query_ids"], 2)
            self.assertTrue(metadata["qa_has_header"])


if __name__ == "__main__":
    unittest.main()
