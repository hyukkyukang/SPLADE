import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from script.preprocess.patent.build_patent_claim_passages import build_passage_corpus
from src.data.patent_passages import (
    build_patent_claim_passages,
    build_title_prefixed_claim_passage,
    split_into_sentence_chunks,
)


class PatentPassagesTest(unittest.TestCase):
    def test_split_into_sentence_chunks_groups_sentences_by_word_budget(self) -> None:
        text = "One two. Three four. Five six seven."
        chunks = split_into_sentence_chunks(text, max_words=4)
        self.assertEqual(
            chunks,
            [
                "One two. Three four.",
                "Five six seven.",
            ],
        )

    def test_build_title_prefixed_claim_passage_trims_title_before_chunk(self) -> None:
        title = "One Two Three Four"
        claim_chunk = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        passage = build_title_prefixed_claim_passage(
            title=title,
            claim_chunk=claim_chunk,
            max_title_prefixed_words=12,
        )
        self.assertEqual(
            passage,
            "One Two alpha beta gamma delta epsilon zeta eta theta iota kappa",
        )

    def test_build_patent_claim_passages_emits_grouped_claim_passages(self) -> None:
        passages = build_patent_claim_passages(
            {
                "doc_id": "US100",
                "parent_doc_id": "US100",
                "title": "Valve Assembly",
                "claims": (
                    "A first claim sentence. "
                    "A second claim sentence. "
                    "A third claim sentence."
                ),
            },
            max_claim_chunk_words=8,
            max_title_prefixed_words=20,
        )
        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0]["passage_id"], "US100&&&claim&&&0")
        self.assertEqual(passages[0]["parent_doc_id"], "US100")
        self.assertEqual(passages[0]["chunk_type"], "claim")
        self.assertIn("Valve Assembly", passages[0]["text"])

    def test_build_passage_corpus_writes_parent_doc_ids(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "passages.parquet"
            metadata = build_passage_corpus(
                row_iter=[
                    {
                        "doc_id": "US100",
                        "parent_doc_id": "US100",
                        "title": "Valve Assembly",
                        "claims": "A first claim sentence. A second claim sentence.",
                    }
                ],
                output_path=output_path,
                group_id_column="parent_doc_id",
                max_claim_chunk_words=100,
                max_title_prefixed_words=100,
                write_batch_size=2,
            )
            rows = pq.read_table(output_path).to_pylist()
            self.assertEqual(metadata["document_count"], 1)
            self.assertEqual(metadata["passage_count"], 1)
            self.assertEqual(rows[0]["passage_id"], "US100&&&claim&&&0")
            self.assertEqual(rows[0]["parent_doc_id"], "US100")


if __name__ == "__main__":
    unittest.main()
