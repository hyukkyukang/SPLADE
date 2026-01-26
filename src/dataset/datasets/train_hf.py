from __future__ import annotations

import random
from functools import cached_property
from typing import Iterable

from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info

from src.dataset.collators import TrainCollator
from src.dataset.datasets.train import TrainSample


class HFMSMarcoTrainDataset:
    is_streaming = False

    def __init__(
        self,
        cfg,
        global_cfg,
        tokenizer,
        load_teacher_scores: bool | None = None,
        require_teacher_scores: bool | None = None,
    ) -> None:
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.tokenizer = tokenizer

        self.hf_name = cfg.hf_name
        self.hf_subset = cfg.hf_subset
        self.hf_split = cfg.hf_split
        self.hf_text_name = cfg.hf_text_name
        self.hf_cache_dir = cfg.hf_cache_dir
        self.hf_max_samples = cfg.hf_max_samples
        self.hf_teacher_name = cfg.hf_teacher_name
        self.hf_teacher_subset = cfg.hf_teacher_subset
        self.hf_teacher_split = cfg.hf_teacher_split
        self.hf_teacher_cache_dir = cfg.hf_teacher_cache_dir
        self.hf_teacher_max_samples = cfg.hf_teacher_max_samples

        self.num_positives = cfg.num_positives
        self.num_negatives = cfg.num_negatives

        distill_cfg = global_cfg.training.distill
        if load_teacher_scores is None:
            load_teacher_scores = bool(distill_cfg.enabled)
        self.load_teacher_scores = bool(load_teacher_scores)
        if require_teacher_scores is None:
            self.require_teacher_scores = bool(
                self.load_teacher_scores and distill_cfg.fail_on_missing
            )
        else:
            self.require_teacher_scores = bool(require_teacher_scores)
        self.teacher_score_key = distill_cfg.teacher_score_key

        self.dataset = None
        self.query_map: dict[str, str] = {}
        self.corpus_map: dict[str, str] = {}
        self.teacher_scores: dict[tuple[str, str], float] = {}

    @cached_property
    def collator(self) -> TrainCollator:
        return TrainCollator(
            tokenizer=self.tokenizer,
            max_query_length=self.cfg.max_query_length,
            max_doc_length=self.cfg.max_doc_length,
            require_teacher_scores=self.require_teacher_scores,
        )

    def prepare_data(self) -> None:
        dataset = load_dataset(
            self.hf_name,
            self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
        )
        has_text_triplets = (
            {"query", "positive", "negative"}.issubset(dataset.column_names)
            or {"anchor", "positive", "negative"}.issubset(dataset.column_names)
        )
        if not has_text_triplets:
            text_name = self.hf_text_name or self.hf_name
            load_dataset(
                text_name, "queries", split="train", cache_dir=self.hf_cache_dir
            )
            load_dataset(
                text_name, "corpus", split="train", cache_dir=self.hf_cache_dir
            )
        if (
            self.load_teacher_scores
            and self.hf_teacher_name
            and self.hf_teacher_subset
        ):
            load_dataset(
                self.hf_teacher_name,
                self.hf_teacher_subset,
                split=self.hf_teacher_split,
                cache_dir=self.hf_teacher_cache_dir or self.hf_cache_dir,
            )

    def setup(self) -> None:
        self.dataset = load_dataset(
            self.hf_name,
            self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
        )
        if self.hf_max_samples is not None:
            self.dataset = self.dataset.select(
                range(min(self.hf_max_samples, len(self.dataset)))
            )

        has_text_triplets = (
            {"query", "positive", "negative"}.issubset(self.dataset.column_names)
            or {"anchor", "positive", "negative"}.issubset(self.dataset.column_names)
        )
        if has_text_triplets:
            self.query_map = {}
            self.corpus_map = {}
        else:
            text_name = self.hf_text_name or self.hf_name
            if self.hf_max_samples is not None:
                required_qids, required_doc_ids = self._collect_required_text_ids()
                self.query_map, self.corpus_map = self._load_queries_and_corpus_subset(
                    text_name, self.hf_cache_dir, required_qids, required_doc_ids
                )
            else:
                self.query_map, self.corpus_map = self._load_queries_and_corpus(
                    text_name, self.hf_cache_dir
                )

        if self.load_teacher_scores:
            self.teacher_scores = self._load_teacher_scores(
                self.hf_teacher_name,
                self.hf_teacher_subset,
                self.hf_teacher_split,
                self.hf_teacher_cache_dir or self.hf_cache_dir,
                self.hf_teacher_max_samples,
            )
        else:
            self.teacher_scores = {}

    def _assert_setup(self) -> None:
        if self.dataset is None:
            raise RuntimeError("Dataset is not set up. Call setup() first.")

    @staticmethod
    def _load_queries_and_corpus(
        hf_name: str, hf_cache_dir: str | None
    ) -> tuple[dict[str, str], dict[str, str]]:
        queries_ds = load_dataset(
            hf_name, "queries", split="train", cache_dir=hf_cache_dir
        )
        corpus_ds = load_dataset(
            hf_name, "corpus", split="train", cache_dir=hf_cache_dir
        )

        query_map: dict[str, str] = {}
        for row in queries_ds:
            qid = str(row.get("query_id") or row.get("_id") or row.get("id"))
            text = row.get("text") or row.get("query") or ""
            query_map[qid] = text

        corpus_map: dict[str, str] = {}
        for row in corpus_ds:
            doc_id = str(
                row.get("doc_id")
                or row.get("passage_id")
                or row.get("_id")
                or row.get("id")
            )
            text = row.get("text") or row.get("passage") or row.get("contents") or ""
            corpus_map[doc_id] = text

        return query_map, corpus_map

    @staticmethod
    def _load_queries_and_corpus_subset(
        hf_name: str,
        hf_cache_dir: str | None,
        required_qids: set[str],
        required_doc_ids: set[str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        if not required_qids and not required_doc_ids:
            return {}, {}

        queries_ds = load_dataset(
            hf_name, "queries", split="train", cache_dir=hf_cache_dir, streaming=True
        )
        corpus_ds = load_dataset(
            hf_name, "corpus", split="train", cache_dir=hf_cache_dir, streaming=True
        )

        query_map: dict[str, str] = {}
        remaining_qids = set(required_qids)
        if remaining_qids:
            for row in queries_ds:
                qid = str(row.get("query_id") or row.get("_id") or row.get("id"))
                if qid in remaining_qids:
                    text = row.get("text") or row.get("query") or ""
                    query_map[qid] = text
                    remaining_qids.remove(qid)
                    if not remaining_qids:
                        break

        corpus_map: dict[str, str] = {}
        remaining_doc_ids = set(required_doc_ids)
        if remaining_doc_ids:
            for row in corpus_ds:
                doc_id = str(
                    row.get("doc_id")
                    or row.get("passage_id")
                    or row.get("_id")
                    or row.get("id")
                )
                if doc_id in remaining_doc_ids:
                    text = (
                        row.get("text")
                        or row.get("passage")
                        or row.get("contents")
                        or ""
                    )
                    corpus_map[doc_id] = text
                    remaining_doc_ids.remove(doc_id)
                    if not remaining_doc_ids:
                        break

        return query_map, corpus_map

    def _collect_required_text_ids(self) -> tuple[set[str], set[str]]:
        self._assert_setup()
        required_qids: set[str] = set()
        required_doc_ids: set[str] = set()
        for row in self.dataset:
            if "query" in row and "positive" in row and "negative" in row:
                continue
            if "query_id" in row and "positive_id" in row:
                qid = str(row["query_id"])
                required_qids.add(qid)
                required_doc_ids.add(str(row["positive_id"]))
                required_doc_ids.add(str(row["negative_id"]))
                continue
            if "query_id" in row and "doc_ids" in row and "labels" in row:
                qid = str(row["query_id"])
                required_qids.add(qid)
                required_doc_ids.update(str(doc_id) for doc_id in row["doc_ids"])
                continue
            raise ValueError(f"Unsupported MSMARCO HF row format: {row.keys()}")
        return required_qids, required_doc_ids

    @staticmethod
    def _load_teacher_scores(
        hf_name: str | None,
        subset: str | None,
        split: str,
        hf_cache_dir: str | None,
        max_samples: int | None,
    ) -> dict[tuple[str, str], float]:
        if hf_name is None or subset is None:
            return {}
        teacher_ds = load_dataset(hf_name, subset, split=split, cache_dir=hf_cache_dir)
        if max_samples is not None:
            teacher_ds = teacher_ds.select(range(min(max_samples, len(teacher_ds))))
        scores: dict[tuple[str, str], float] = {}
        for row in teacher_ds:
            qid = str(row.get("query_id") or row.get("qid") or row.get("_id"))
            if subset == "pair":
                doc_id = str(
                    row.get("corpus_id")
                    or row.get("doc_id")
                    or row.get("passage_id")
                    or row.get("pid")
                    or row.get("_id")
                )
                score = row.get("score")
                if score is None:
                    continue
                scores[(qid, doc_id)] = float(score)
            elif subset == "triplet":
                pos_id = str(
                    row.get("positive_id") or row.get("pos_id") or row.get("doc_pos_id")
                )
                neg_id = str(
                    row.get("negative_id") or row.get("neg_id") or row.get("doc_neg_id")
                )
                score = row.get("score") or row.get("scores")
                if isinstance(score, (list, tuple)) and len(score) == 2:
                    scores[(qid, pos_id)] = float(score[0])
                    scores[(qid, neg_id)] = float(score[1])
            elif subset == "list":
                doc_ids = row.get("corpus_id") or row.get("doc_ids") or []
                score_list = row.get("score") or row.get("scores") or []
                for doc_id, score in zip(doc_ids, score_list):
                    scores[(qid, str(doc_id))] = float(score)
        return scores

    def __len__(self) -> int:
        self._assert_setup()
        return len(self.dataset)

    def __getitem__(self, index: int) -> TrainSample:
        self._assert_setup()
        row = self.dataset[index]

        if "query" in row and "positive" in row and "negative" in row:
            query_text = row["query"]
            pos_texts = [row["positive"]]
            neg_texts = [row["negative"]]
            pos_ids = [None]
            neg_ids = [None]
            qid = str(row.get("query_id") or row.get("qid") or index)
            row_teacher_scores = None
        elif "anchor" in row and "positive" in row and "negative" in row:
            query_text = row["anchor"]
            pos_texts = [row["positive"]]
            neg_texts = [row["negative"]]
            pos_ids = [None]
            neg_ids = [None]
            qid = str(row.get("query_id") or row.get("qid") or index)
            row_teacher_scores = None
        elif "query_id" in row and "positive_id" in row:
            qid = str(row["query_id"])
            query_text = self.query_map.get(qid, "")
            pos_ids = [str(row["positive_id"])]
            neg_ids = [str(row["negative_id"])]
            pos_texts = [self.corpus_map.get(pos_ids[0], "")]
            neg_texts = [self.corpus_map.get(neg_ids[0], "")]
            score_values = row.get("score") or row.get("scores")
            if isinstance(score_values, (list, tuple)) and len(score_values) == 2:
                row_teacher_scores = [float(score_values[0]), float(score_values[1])]
            else:
                row_teacher_scores = None
        elif "query_id" in row and "doc_ids" in row and "labels" in row:
            qid = str(row["query_id"])
            query_text = self.query_map.get(qid, "")
            doc_ids = [str(doc_id) for doc_id in row["doc_ids"]]
            labels = row["labels"]
            pos_ids = [doc_id for doc_id, label in zip(doc_ids, labels) if label > 0]
            neg_ids = [doc_id for doc_id, label in zip(doc_ids, labels) if label <= 0]
            pos_ids = self._sample_ids(pos_ids, self.num_positives)
            neg_ids = self._sample_ids(neg_ids, self.num_negatives)
            pos_texts = [self.corpus_map.get(doc_id, "") for doc_id in pos_ids]
            neg_texts = [self.corpus_map.get(doc_id, "") for doc_id in neg_ids]
            row_teacher_scores = None
        else:
            raise ValueError(f"Unsupported MSMARCO HF row format: {row.keys()}")

        docs = pos_texts + neg_texts
        doc_ids = pos_ids + neg_ids
        if row_teacher_scores is not None and len(row_teacher_scores) == len(doc_ids):
            teacher_scores = row_teacher_scores
        else:
            teacher_scores = [
                self._get_teacher_score(qid, doc_id) for doc_id in doc_ids
            ]

        if self.require_teacher_scores and any(
            score != score for score in teacher_scores
        ):
            raise ValueError(f"Missing teacher score in HF sample for query {qid}")

        return TrainSample(
            query=query_text,
            docs=docs,
            pos_count=len(pos_texts),
            teacher_scores=teacher_scores,
        )

    def _sample_ids(self, items: list[str], count: int) -> list[str]:
        if count <= 0:
            return []
        if len(items) <= count:
            return items
        return random.sample(items, count)

    def _get_teacher_score(self, qid: str, doc_id: str | None) -> float:
        if doc_id is None:
            return float("nan")
        score = self.teacher_scores.get((qid, str(doc_id)))
        if score is None:
            return float("nan")
        return float(score)


class HFMSMarcoTrainIterableDataset(IterableDataset):
    is_streaming = True

    def __init__(
        self,
        cfg,
        global_cfg,
        tokenizer,
        load_teacher_scores: bool | None = None,
        require_teacher_scores: bool | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.tokenizer = tokenizer

        self.hf_name = cfg.hf_name
        self.hf_subset = cfg.hf_subset
        self.hf_split = cfg.hf_split
        self.hf_text_name = cfg.hf_text_name
        self.hf_cache_dir = cfg.hf_cache_dir
        self.hf_max_samples = cfg.hf_max_samples
        self.hf_teacher_name = cfg.hf_teacher_name
        self.hf_teacher_subset = cfg.hf_teacher_subset
        self.hf_teacher_split = cfg.hf_teacher_split
        self.hf_teacher_cache_dir = cfg.hf_teacher_cache_dir
        self.hf_teacher_max_samples = cfg.hf_teacher_max_samples

        self.num_positives = cfg.num_positives
        self.num_negatives = cfg.num_negatives

        distill_cfg = global_cfg.training.distill
        if load_teacher_scores is None:
            load_teacher_scores = bool(distill_cfg.enabled)
        self.load_teacher_scores = bool(load_teacher_scores)
        if require_teacher_scores is None:
            self.require_teacher_scores = bool(
                self.load_teacher_scores and distill_cfg.fail_on_missing
            )
        else:
            self.require_teacher_scores = bool(require_teacher_scores)
        self.teacher_score_key = distill_cfg.teacher_score_key

        self.shuffle_buffer_size = int(getattr(cfg, "hf_shuffle_buffer_size", 0) or 0)
        self.seed = global_cfg.seed

        self.dataset = None
        self._length = None
        self.query_map: dict[str, str] = {}
        self.corpus_map: dict[str, str] = {}
        self.teacher_scores: dict[tuple[str, str], float] = {}

    @cached_property
    def collator(self) -> TrainCollator:
        return TrainCollator(
            tokenizer=self.tokenizer,
            max_query_length=self.cfg.max_query_length,
            max_doc_length=self.cfg.max_doc_length,
            require_teacher_scores=self.require_teacher_scores,
        )

    def prepare_data(self) -> None:
        load_dataset(
            self.hf_name,
            self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
            streaming=True,
        )

    def setup(self) -> None:
        self.dataset = load_dataset(
            self.hf_name,
            self.hf_subset,
            split=self.hf_split,
            cache_dir=self.hf_cache_dir,
            streaming=True,
        )
        if self.hf_max_samples is not None:
            self.dataset = self.dataset.take(self.hf_max_samples)

        self._length = self._resolve_length(self.hf_max_samples)

        has_text_triplets = (
            {"query", "positive", "negative"}.issubset(self.dataset.column_names)
            or {"anchor", "positive", "negative"}.issubset(self.dataset.column_names)
        )
        if not has_text_triplets:
            raise ValueError(
                "Streaming mode requires datasets with inline query/positive/negative "
                "text columns. Disable streaming or use a dataset that includes text."
            )

        has_inline_scores = any(
            key in self.dataset.column_names for key in ("score", "scores")
        )
        if self.load_teacher_scores and self.hf_teacher_name and self.hf_teacher_subset:
            raise ValueError(
                "Streaming mode does not support separate teacher score datasets. "
                "Disable distillation or use a dataset with inline score columns."
            )
        if self.require_teacher_scores and not has_inline_scores:
            raise ValueError(
                "Distillation requires teacher scores, but streaming dataset does not "
                "include score columns. Disable distillation or use inline scores."
            )

        self.query_map = {}
        self.corpus_map = {}
        self.teacher_scores = {}
    def _assert_setup(self) -> None:
        if self.dataset is None:
            raise RuntimeError("Dataset is not set up. Call setup() first.")

    def __len__(self) -> int:
        self._assert_setup()
        if self._length is None:
            raise TypeError("Length is not known for this streaming dataset.")
        return self._length

    def __iter__(self) -> Iterable[TrainSample]:
        self._assert_setup()
        dataset = self._prepare_stream(self.dataset)
        for index, row in enumerate(dataset):
            yield self._row_to_sample(row, index)

    def _prepare_stream(self, dataset):
        if self.shuffle_buffer_size > 0:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer_size, seed=self.seed
            )
        return self._shard_for_workers(dataset)

    def _shard_for_workers(self, dataset):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        world_size = 1
        rank = 0
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
        except Exception:
            pass

        total_shards = world_size * num_workers
        if total_shards <= 1:
            return dataset
        shard_index = rank * num_workers + worker_id
        return dataset.shard(
            num_shards=total_shards, index=shard_index, contiguous=True
        )

    def _resolve_length(self, hf_max_samples: int | None) -> int | None:
        if hf_max_samples is not None:
            return int(hf_max_samples)
        try:
            split_info = self.dataset.info.splits.get(self.hf_split)
            if split_info and split_info.num_examples:
                return int(split_info.num_examples)
        except Exception:
            return None
        return None

    def _row_to_sample(self, row, index: int) -> TrainSample:
        if "query" in row and "positive" in row and "negative" in row:
            query_text = row["query"]
            pos_texts = [row["positive"]]
            neg_texts = [row["negative"]]
            pos_ids = [None]
            neg_ids = [None]
            qid = str(row.get("query_id") or row.get("qid") or index)
            row_teacher_scores = None
        elif "anchor" in row and "positive" in row and "negative" in row:
            query_text = row["anchor"]
            pos_texts = [row["positive"]]
            neg_texts = [row["negative"]]
            pos_ids = [None]
            neg_ids = [None]
            qid = str(row.get("query_id") or row.get("qid") or index)
            row_teacher_scores = None
        elif "query_id" in row and "positive_id" in row:
            qid = str(row["query_id"])
            query_text = self.query_map.get(qid, "")
            pos_ids = [str(row["positive_id"])]
            neg_ids = [str(row["negative_id"])]
            pos_texts = [self.corpus_map.get(pos_ids[0], "")]
            neg_texts = [self.corpus_map.get(neg_ids[0], "")]
            score_values = row.get("score") or row.get("scores")
            if isinstance(score_values, (list, tuple)) and len(score_values) == 2:
                row_teacher_scores = [float(score_values[0]), float(score_values[1])]
            else:
                row_teacher_scores = None
        elif "query_id" in row and "doc_ids" in row and "labels" in row:
            qid = str(row["query_id"])
            query_text = self.query_map.get(qid, "")
            doc_ids = [str(doc_id) for doc_id in row["doc_ids"]]
            labels = row["labels"]
            pos_ids = [doc_id for doc_id, label in zip(doc_ids, labels) if label > 0]
            neg_ids = [doc_id for doc_id, label in zip(doc_ids, labels) if label <= 0]
            pos_ids = self._sample_ids(pos_ids, self.num_positives)
            neg_ids = self._sample_ids(neg_ids, self.num_negatives)
            pos_texts = [self.corpus_map.get(doc_id, "") for doc_id in pos_ids]
            neg_texts = [self.corpus_map.get(doc_id, "") for doc_id in neg_ids]
            row_teacher_scores = None
        else:
            raise ValueError(f"Unsupported MSMARCO HF row format: {row.keys()}")

        docs = pos_texts + neg_texts
        doc_ids = pos_ids + neg_ids
        if row_teacher_scores is not None and len(row_teacher_scores) == len(doc_ids):
            teacher_scores = row_teacher_scores
        else:
            teacher_scores = [
                self._get_teacher_score(qid, doc_id) for doc_id in doc_ids
            ]

        if self.require_teacher_scores and any(
            score != score for score in teacher_scores
        ):
            raise ValueError(f"Missing teacher score in HF sample for query {qid}")

        return TrainSample(
            query=query_text,
            docs=docs,
            pos_count=len(pos_texts),
            teacher_scores=teacher_scores,
        )

    def _sample_ids(self, items: list[str], count: int) -> list[str]:
        if count <= 0:
            return []
        if len(items) <= count:
            return items
        return random.sample(items, count)

    def _get_teacher_score(self, qid: str, doc_id: str | None) -> float:
        if doc_id is None:
            return float("nan")
        score = self.teacher_scores.get((qid, str(doc_id)))
        if score is None:
            return float("nan")
        return float(score)
