import unittest
from unittest.mock import patch

from datasets import Dataset
from omegaconf import OmegaConf

from src.data.dataset.corpus_only import CorpusOnlyDataset


def build_dataset_cfg() -> object:
    return OmegaConf.create(
        {
            "name": "patent_us_corpus_small",
            "type": "corpus_only",
            "split": "train",
            "corpus_split": "train",
            "hf_split": "train",
            "hf_name": None,
            "hf_subset": None,
            "hf_cache_dir": None,
            "hf_max_samples": None,
            "hf_skip_samples": 0,
            "hf_data_files": None,
            "query_corpus_hf_name": "Hyukkyu/patent-us-corpus-small",
            "query_corpus_hf_cache_dir": None,
            "query_corpus_hf_data_files": None,
            "query_hf_data_files": None,
            "corpus_hf_data_files": None,
            "query_lookup_hf_name": None,
            "query_lookup_hf_subset": None,
            "query_lookup_hf_split": "train",
            "query_lookup_hf_cache_dir": None,
            "query_lookup_hf_data_files": None,
            "query_lookup_id_column": "query_id",
            "query_lookup_text_column": "text",
            "negative_sampling": {
                "strategy": "random",
                "top_k": None,
                "random_k": None,
                "random_pool": None,
            },
            "local_triplets_dir": None,
            "query_subset_name": None,
            "query_split_name": "train",
            "query_id_column": "doc_id",
            "query_text_column": "abstract",
            "corpus_subset_name": None,
            "corpus_split_name": "train",
            "corpus_id_column": "doc_id",
            "corpus_text_column": "abstract",
            "corpus_title_column": "title",
            "corpus_additional_text_columns": ["claims", "description"],
            "corpus_text_template": "patent_document_v1",
        }
    )


class CorpusTextTemplateTest(unittest.TestCase):
    def test_corpus_text_uses_named_template(self) -> None:
        dataset = CorpusOnlyDataset(cfg=build_dataset_cfg())
        source_dataset = Dataset.from_list(
            [
                {
                    "doc_id": "US100",
                    "title": "Valve Assembly",
                    "abstract": "A valve assembly with a hinged seal.",
                    "claims": "1. A valve assembly...",
                    "description": "Detailed valve description.",
                }
            ]
        )
        with patch.object(dataset, "_load_hf_dataset", return_value=source_dataset):
            text = dataset.corpus_text(0)

        self.assertEqual(
            text,
            "Title: Valve Assembly\n"
            "Abstract: A valve assembly with a hinged seal.\n"
            "Claims: 1. A valve assembly...\n"
            "Description: Detailed valve description.",
        )


if __name__ == "__main__":
    unittest.main()
