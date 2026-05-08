import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from script.preprocess.patent.build_patent_us_in_batch_metadata import (
    build_title_abstract_key,
    extract_ctx_title_abstract,
    iter_json_array,
    resolve_corpus_paths,
    write_patent_us_in_batch_parquet,
)


class BuildPatentUsInBatchMetadataTest(unittest.TestCase):
    def test_extract_ctx_title_abstract_splits_sep_text(self) -> None:
        title, abstract = extract_ctx_title_abstract(
            {
                "title": "",
                "text": "Widget Patent [SEP] A system for building widgets efficiently.",
            }
        )

        self.assertEqual(title, "Widget Patent")
        self.assertEqual(abstract, "A system for building widgets efficiently.")

    def test_iter_json_array_reads_streamed_array(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rows.json"
            path.write_text(
                json.dumps([{"value": 1}, {"value": 2}], ensure_ascii=False),
                encoding="utf-8",
            )

            rows = list(iter_json_array(path, chunk_size=4))

            self.assertEqual(rows, [{"value": 1}, {"value": 2}])

    def test_write_patent_us_in_batch_parquet_resolves_doc_ids(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_json_path = tmp_path / "usc102103_train.json"
            corpus_path = tmp_path / "patent_us_docs_slice00of24-00000.parquet"
            output_path = tmp_path / "stage1.parquet"

            raw_rows = [
                {
                    "question": "How is the valve assembly configured?",
                    "answers": [],
                    "negative_ctxs": [],
                    "hard_negatives_ctxs": [],
                    "positive_ctxs": [
                        {
                            "title": "",
                            "text": "Valve Assembly [SEP] A valve assembly with a hinged seal.",
                        }
                    ],
                },
                {
                    "question": "What does the widget controller do?",
                    "answers": [],
                    "negative_ctxs": [],
                    "hard_negatives_ctxs": [],
                    "positive_ctxs": [
                        {
                            "title": "Widget Controller",
                            "text": "A controller that manages widget timing.",
                        },
                        {
                            "title": "",
                            "text": "Unknown Patent [SEP] This one should not match.",
                        },
                    ],
                },
            ]
            raw_json_path.write_text(
                json.dumps(raw_rows, ensure_ascii=False),
                encoding="utf-8",
            )

            corpus_table = pa.Table.from_pylist(
                [
                    {
                        "doc_id": "US100",
                        "title": "Valve Assembly",
                        "abstract": "A valve assembly with a hinged seal.",
                        "claims": "1. A valve assembly...",
                        "description": "Detailed valve description.",
                        "application_id": "US100",
                    },
                    {
                        "doc_id": "US200",
                        "title": "Widget Controller",
                        "abstract": "A controller that manages widget timing.",
                        "claims": "1. A widget controller...",
                        "description": "Detailed controller description.",
                        "application_id": "US200",
                    },
                ]
            )
            pq.write_table(corpus_table, corpus_path)

            stats = write_patent_us_in_batch_parquet(
                raw_json_path=raw_json_path,
                corpus_paths=[corpus_path],
                output_path=output_path,
            )

            rows = pq.read_table(output_path.as_posix()).to_pylist()
            self.assertEqual(stats.raw_rows, 2)
            self.assertEqual(stats.emitted_rows, 2)
            self.assertEqual(stats.source_positive_contexts, 3)
            self.assertEqual(stats.matched_positive_contexts, 2)
            self.assertEqual(stats.unresolved_positive_contexts, 1)
            self.assertEqual(
                rows,
                [
                    {
                        "query_id": "q_"
                        + hashlib.sha1(
                            "How is the valve assembly configured?".encode("utf-8")
                        ).hexdigest(),
                        "query_text": "How is the valve assembly configured?",
                        "pos_doc_ids": ["US100"],
                        "source_positive_count": 1,
                        "matched_positive_count": 1,
                    },
                    {
                        "query_id": "q_"
                        + hashlib.sha1(
                            "What does the widget controller do?".encode("utf-8")
                        ).hexdigest(),
                        "query_text": "What does the widget controller do?",
                        "pos_doc_ids": ["US200"],
                        "source_positive_count": 2,
                        "matched_positive_count": 1,
                    },
                ],
            )

    def test_write_patent_us_in_batch_parquet_drops_ambiguous_matches(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_json_path = tmp_path / "usc102103_train.json"
            corpus_path = tmp_path / "patent_us_docs_slice00of24-00000.parquet"
            output_path = tmp_path / "stage1.parquet"

            raw_json_path.write_text(
                json.dumps(
                    [
                        {
                            "question": "Which patent is the matching one?",
                            "answers": [],
                            "negative_ctxs": [],
                            "hard_negatives_ctxs": [],
                            "positive_ctxs": [
                                {
                                    "title": "",
                                    "text": "Duplicate Patent [SEP] A duplicated abstract.",
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            duplicate_key = build_title_abstract_key(
                "Duplicate Patent",
                "A duplicated abstract.",
            )
            self.assertIsNotNone(duplicate_key)

            corpus_table = pa.Table.from_pylist(
                [
                    {
                        "doc_id": "US300",
                        "title": "Duplicate Patent",
                        "abstract": "A duplicated abstract.",
                        "claims": "1. First duplicate",
                        "description": "desc",
                        "application_id": "US300",
                    },
                    {
                        "doc_id": "US301",
                        "title": "Duplicate Patent",
                        "abstract": "A duplicated abstract.",
                        "claims": "1. Second duplicate",
                        "description": "desc",
                        "application_id": "US301",
                    },
                ]
            )
            pq.write_table(corpus_table, corpus_path)

            stats = write_patent_us_in_batch_parquet(
                raw_json_path=raw_json_path,
                corpus_paths=[corpus_path],
                output_path=output_path,
            )

            rows = pq.read_table(output_path.as_posix()).to_pylist()
            self.assertEqual(rows, [])
            self.assertEqual(stats.raw_rows, 1)
            self.assertEqual(stats.emitted_rows, 0)
            self.assertEqual(stats.ambiguous_target_keys, 1)
            self.assertEqual(stats.ambiguous_positive_contexts, 1)
            self.assertEqual(stats.dropped_rows_without_matches, 1)

    def test_resolve_corpus_paths_errors_on_empty_glob(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                resolve_corpus_paths(str(Path(tmp_dir) / "*.parquet"))


if __name__ == "__main__":
    unittest.main()
