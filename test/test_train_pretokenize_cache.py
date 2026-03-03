import tempfile
import unittest
from pathlib import Path
from typing import Any

from datasets import Dataset
from omegaconf import OmegaConf

from src.data.dataclass import MetaItem
from src.data.pd_module.train import TrainingPDModule


class CountingTokenizer:
    def __init__(self) -> None:
        self.calls: int = 0
        self.pad_token_id: int = 0
        self.name_or_path: str = "dummy/tokenizer"
        self.is_fast: bool = True

    @staticmethod
    def _encode_text(text: str, max_length: int) -> list[int]:
        token_ids: list[int] = [2 + (ord(ch) % 17) for ch in text]
        if not token_ids:
            token_ids = [1]
        return token_ids[:max_length]

    def __call__(
        self,
        texts: str | list[str],
        *,
        padding: str | bool = False,
        truncation: bool = True,
        max_length: int = 16,
        return_attention_mask: bool = True,
    ) -> dict[str, list[list[int]]]:
        _ = truncation
        _ = return_attention_mask
        self.calls += 1
        text_list: list[str] = [texts] if isinstance(texts, str) else list(texts)
        encoded_rows: list[list[int]] = [
            self._encode_text(text, max_length=max_length) for text in text_list
        ]

        if padding == "max_length":
            input_ids: list[list[int]] = [
                row + [self.pad_token_id] * (max_length - len(row)) for row in encoded_rows
            ]
            attention_mask: list[list[int]] = [
                [1] * len(row) + [0] * (max_length - len(row)) for row in encoded_rows
            ]
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        if padding is False:
            input_ids = [list(row) for row in encoded_rows]
            attention_mask = [[1] * len(row) for row in encoded_rows]
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        if padding is True:
            padded_len: int = max((len(row) for row in encoded_rows), default=0)
            input_ids = [
                row + [self.pad_token_id] * (padded_len - len(row)) for row in encoded_rows
            ]
            attention_mask = [
                [1] * len(row) + [0] * (padded_len - len(row)) for row in encoded_rows
            ]
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        raise ValueError(f"Unsupported padding mode in test tokenizer: {padding}")


class DummyTrainDataset:
    def __init__(self) -> None:
        self._meta_dataset: Dataset = Dataset.from_list(
            [
                {"query_id": "q1", "positive_id": "d1", "negative_id": "d2"},
                {"query_id": "q2", "positive_id": "d2", "negative_id": "d3"},
            ]
        )
        self._query_rows: Dataset = Dataset.from_list(
            [
                {"query_id": "q1", "query": "hello world"},
                {"query_id": "q2", "query": "good bye"},
            ]
        )
        self._doc_rows: Dataset = Dataset.from_list(
            [
                {"doc_id": "d1", "text": "document one"},
                {"doc_id": "d2", "text": "document two"},
                {"doc_id": "d3", "text": "document three"},
            ]
        )
        self._query_id_to_idx: dict[str, int] = {"q1": 0, "q2": 1}
        self._doc_id_to_idx: dict[str, int] = {"d1": 0, "d2": 1, "d3": 2}
        self.prepare_meta_dataset_calls: int = 0

    @property
    def meta_dataset(self) -> Dataset:
        return self._meta_dataset

    @property
    def query_dataset(self) -> Dataset:
        return self._query_rows

    @property
    def corpus_dataset(self) -> Dataset:
        return self._doc_rows

    @property
    def query_dataset_id_to_idx(self) -> dict[str, int]:
        return self._query_id_to_idx

    @property
    def corpus_dataset_id_to_idx(self) -> dict[str, int]:
        return self._doc_id_to_idx

    def prepare_meta_dataset(self) -> None:
        self.prepare_meta_dataset_calls += 1

    def build_meta_item(
        self,
        row: dict[str, Any],
        index: int,
        *,
        num_positives: int,
        num_negatives: int,
        rng: Any,
        load_teacher_scores: bool,
        require_teacher_scores: bool,
    ) -> MetaItem:
        _ = index
        _ = rng
        _ = load_teacher_scores
        _ = require_teacher_scores
        pos_ids: list[str] = [str(row["positive_id"])] if num_positives > 0 else []
        neg_ids: list[str] = [str(row["negative_id"])] if num_negatives > 0 else []
        return MetaItem(
            qid=str(row["query_id"]),
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            pos_scores=None,
            neg_scores=None,
            query_text=None,
            pos_texts=None,
            neg_texts=None,
        )

    def query_text(self, idx: int) -> str:
        return str(self._query_rows[idx]["query"])

    def corpus_text(self, idx: int) -> str:
        return str(self._doc_rows[idx]["text"])

    def resolve_query_text(self, meta_item: MetaItem) -> str:
        idx: int | None = self._query_id_to_idx.get(meta_item.qid)
        if idx is None:
            return ""
        return self.query_text(idx)

    def resolve_doc_texts(
        self, doc_ids: list[str], inline_texts: list[str] | None
    ) -> list[str]:
        if inline_texts is not None:
            return list(inline_texts)
        return [self.corpus_text(self._doc_id_to_idx[doc_id]) for doc_id in doc_ids]

    def lookup_query_texts(self, qids: list[str]) -> dict[str, str]:
        _ = qids
        return {}


class DummyMissingQueryDataset(DummyTrainDataset):
    def __init__(self) -> None:
        super().__init__()
        self._meta_dataset = Dataset.from_list(
            [
                {"query_id": "q_missing", "positive_id": "d1", "negative_id": "d2"},
            ]
        )
        self.lookup_calls: int = 0
        self._lookup_map: dict[str, str] = {"q_missing": "fallback missing query"}

    def lookup_query_texts(self, qids: list[str]) -> dict[str, str]:
        self.lookup_calls += 1
        return {qid: self._lookup_map[qid] for qid in qids if qid in self._lookup_map}


def _build_cfg(cache_root: str) -> Any:
    return OmegaConf.create(
        {
            "name": "dummy_train",
            "split": "train",
            "max_query_length": 8,
            "max_doc_length": 12,
            "max_padding": True,
            "num_positives": 1,
            "num_negatives": 1,
            "hf_name": "dummy/hf",
            "query_corpus_hf_name": None,
            "pretokenize": {
                "enabled": True,
                "output_dir": cache_root,
                "overwrite": False,
                "loading_mode": "streaming",
                "streaming_index_backend": "sqlite",
                "streaming_max_cached_shards": 1,
                "streaming_row_cache_size": 64,
                "query_shard_size": 10,
                "doc_shard_size": 10,
                "write_dtype": "int32",
                "allow_runtime_tokenize_fallback": False,
                "require_cache_complete": True,
            },
        }
    )


class TrainPretokenizeCacheTest(unittest.TestCase):
    def test_training_pd_module_uses_cache_without_runtime_tokenization(self) -> None:
        with tempfile.TemporaryDirectory(prefix="train_pretoken_cache_") as tmp_dir:
            cfg = _build_cfg(tmp_dir)
            tokenizer = CountingTokenizer()
            module = TrainingPDModule(
                cfg=cfg,
                tokenizer=tokenizer,
                seed=7,
                cache_namespace="train",
            )
            module._dataset = DummyTrainDataset()

            module.prepare_data()
            self.assertGreater(tokenizer.calls, 0)
            cache_dir: Path = Path(tmp_dir) / "train"
            self.assertTrue((cache_dir / "manifest.json").is_file())
            self.assertTrue((cache_dir / "build.done").is_file())
            self.assertTrue(any(cache_dir.glob("queries-*.parquet")))
            self.assertTrue(any(cache_dir.glob("docs-*.parquet")))
            self.assertTrue((cache_dir / "queries.index.sqlite").is_file())
            self.assertTrue((cache_dir / "docs.index.sqlite").is_file())
            self.assertTrue((cache_dir / "queries.row_index.npy").is_file())
            self.assertTrue((cache_dir / "docs.row_index.npy").is_file())
            self.assertTrue((cache_dir / "meta.query_global_rows.npy").is_file())
            self.assertTrue((cache_dir / "meta.doc_global_rows.npy").is_file())
            self.assertTrue((cache_dir / "meta.doc_counts.npy").is_file())

            module.setup()
            calls_after_setup: int = tokenizer.calls
            item = module[0]
            self.assertEqual(tokenizer.calls, calls_after_setup)
            self.assertEqual(item.qid, "q1")
            self.assertEqual(item.query_input_ids.ndim, 1)
            self.assertEqual(item.doc_input_ids.ndim, 2)

    def test_missing_query_id_is_resolved_via_lookup_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="train_pretoken_lookup_") as tmp_dir:
            cfg = _build_cfg(tmp_dir)
            tokenizer = CountingTokenizer()
            module = TrainingPDModule(
                cfg=cfg,
                tokenizer=tokenizer,
                seed=11,
                cache_namespace="train",
            )
            dataset = DummyMissingQueryDataset()
            module._dataset = dataset

            module.prepare_data()
            cache_dir: Path = Path(tmp_dir) / "train"
            self.assertTrue((cache_dir / "queries.index.sqlite").is_file())
            self.assertTrue((cache_dir / "docs.index.sqlite").is_file())
            self.assertGreater(dataset.lookup_calls, 0)

            module.setup()
            calls_after_setup: int = tokenizer.calls
            item = module[0]
            self.assertEqual(item.qid, "q_missing")
            self.assertEqual(tokenizer.calls, calls_after_setup)


if __name__ == "__main__":
    unittest.main()
