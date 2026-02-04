import abc
import logging
import os
import random
from functools import cached_property
from typing import Any, ContextManager, Mapping

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from omegaconf import DictConfig
from torch.utils.data import get_worker_info

from src.data.dataclass import MetaItem
from src.data.utils import id_to_idx, resolve_dataset_column
from src.utils.logging import loading_status

logger: logging.Logger = logging.getLogger("BaseDataset")

QUERY_SUBSET_NAME_KEY: str = "query_subset_name"
QUERY_SPLIT_NAME_KEY: str = "query_split_name"
QUERY_ID_COLUMN_KEY: str = "query_id_column"
QUERY_TEXT_COLUMN_KEY: str = "query_text_column"
CORPUS_SUBSET_NAME_KEY: str = "corpus_subset_name"
CORPUS_SPLIT_NAME_KEY: str = "corpus_split_name"
CORPUS_ID_COLUMN_KEY: str = "corpus_id_column"
CORPUS_TEXT_COLUMN_KEY: str = "corpus_text_column"
CORPUS_TITLE_COLUMN_KEY: str = "corpus_title_column"


class BaseDataset(abc.ABC):
    """Abstract base class for dataset metadata and text access."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg: DictConfig = cfg
        self.name: str = str(self.cfg.name)

        self.hf_name: str | None = self._normalize_optional_str(self.cfg.hf_name)
        self.hf_subset: str | None = self._normalize_optional_str(self.cfg.hf_subset)
        self.hf_split: str = str(self.cfg.split)
        self.hf_cache_dir: str | None = self._normalize_optional_str(
            self.cfg.hf_cache_dir
        )
        self.hf_max_samples: int | None = (
            None if self.cfg.hf_max_samples is None else int(self.cfg.hf_max_samples)
        )
        self.hf_skip_samples: int = int(self.cfg.hf_skip_samples)
        self.hf_data_files: Mapping[str, Any] | None = self.cfg.hf_data_files
        self.query_corpus_hf_name: str | None = self._normalize_optional_str(
            self.cfg.query_corpus_hf_name
        )
        self.query_corpus_hf_cache_dir: str | None = self._normalize_optional_str(
            self.cfg.query_corpus_hf_cache_dir
        )
        self.query_corpus_hf_data_files: Mapping[str, Any] | None = (
            self.cfg.query_corpus_hf_data_files
        )
        self.use_hf: bool = bool(
            self.hf_name is not None or self.query_corpus_hf_name is not None
        )

        self.local_triplets_dir: str | None = self._normalize_optional_str(
            self.cfg.local_triplets_dir
        )

        # Column-name maps should be filled by child classes.
        self.query_column_names: dict[str, str] = {
            QUERY_SUBSET_NAME_KEY: "",
            QUERY_SPLIT_NAME_KEY: "",
            QUERY_ID_COLUMN_KEY: "",
            QUERY_TEXT_COLUMN_KEY: "",
        }
        self.corpus_column_names: dict[str, str] = {
            CORPUS_SUBSET_NAME_KEY: "",
            CORPUS_SPLIT_NAME_KEY: "",
            CORPUS_ID_COLUMN_KEY: "",
            CORPUS_TEXT_COLUMN_KEY: "",
            CORPUS_TITLE_COLUMN_KEY: "",
        }

    # --- Property methods ---
    @property
    def rank_id(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def worker_id(self) -> int:
        worker_info: Any | None = get_worker_info()
        return int(worker_info.id) if worker_info is not None else 0

    @property
    def all_qids(self) -> set[str]:
        """Get all query IDs in the dataset."""
        query_ids: list[Any] = list(self.query_dataset[self.query_id_column_name])
        # Normalize IDs to strings for consistent downstream lookups.
        return {str(qid) for qid in query_ids}

    @property
    def all_dids(self) -> set[str]:
        """Get all document IDs in the corpus."""
        doc_ids: list[Any] = list(self.corpus_dataset[self.corpus_id_column_name])
        # Normalize IDs to strings for consistent downstream lookups.
        return {str(doc_id) for doc_id in doc_ids}

    @property
    def huggingface_name(self) -> str:
        """Return the Hugging Face dataset name from config."""
        if self.query_corpus_hf_name is not None:
            return self.query_corpus_hf_name
        hf_name_value: Any | None = self.cfg.get("huggingface_name")
        if hf_name_value is None:
            hf_name_value = self.hf_name
        if hf_name_value is None:
            raise ValueError(
                "Missing dataset name in config (huggingface_name/hf_name)"
            )
        return str(hf_name_value)

    @cached_property
    def query_dataset(self) -> Dataset:
        """Get the query dataset containing all queries."""
        self._ensure_hf_enabled()
        with self._loading(
            logger, f"query dataset for {self.huggingface_name}", only_once=True
        ):
            subset_name: str = self.query_column_names[QUERY_SUBSET_NAME_KEY]
            split_name: str = self.query_column_names[QUERY_SPLIT_NAME_KEY]
            text_cache_dir: str | None = (
                self.query_corpus_hf_cache_dir
                if self.query_corpus_hf_cache_dir is not None
                else self.hf_cache_dir
            )
            dataset: Dataset = self._load_hf_dataset(
                self.huggingface_name,
                subset_name,
                split_name,
                text_cache_dir,
                self.query_corpus_hf_data_files,
            )
        return dataset

    @cached_property
    def corpus_dataset(self) -> Dataset:
        """Get the corpus dataset containing all documents/passages."""
        self._ensure_hf_enabled()
        with self._loading(
            logger, f"corpus dataset for {self.huggingface_name}", only_once=True
        ):
            subset_name: str = self.corpus_column_names[CORPUS_SUBSET_NAME_KEY]
            split_name: str = self.corpus_column_names[CORPUS_SPLIT_NAME_KEY]
            text_cache_dir: str | None = (
                self.query_corpus_hf_cache_dir
                if self.query_corpus_hf_cache_dir is not None
                else self.hf_cache_dir
            )
            dataset: Dataset = self._load_hf_dataset(
                self.huggingface_name,
                subset_name,
                split_name,
                text_cache_dir,
                self.query_corpus_hf_data_files,
            )
        return dataset

    @cached_property
    def meta_dataset(self) -> Dataset:
        """Return the dataset providing training metadata rows."""
        return self._resolve_meta_dataset()

    @property
    def query_id_column_name(self) -> str:
        """Return the column name for query IDs."""
        return self.query_column_names[QUERY_ID_COLUMN_KEY]

    @property
    def corpus_id_column_name(self) -> str:
        """Return the column name for document IDs."""
        return self.corpus_column_names[CORPUS_ID_COLUMN_KEY]

    @property
    def corpus_title_column_name(self) -> str | None:
        """Return the column name for document titles, if available."""
        return self.corpus_column_names.get(CORPUS_TITLE_COLUMN_KEY)

    @property
    def query_text_column_name(self) -> str:
        """Get the column name for query text."""
        return self.query_column_names[QUERY_TEXT_COLUMN_KEY]

    @property
    def corpus_text_column_name(self) -> str:
        """Get the column name for corpus text."""
        return self.corpus_column_names[CORPUS_TEXT_COLUMN_KEY]

    @cached_property
    def query_dataset_id_to_idx(self) -> dict[str, int]:
        """Create a mapping from query IDs to their indices in the query dataset."""
        enable_tqdm: bool = self.rank_id == 0 and self.worker_id == 0
        # Use resolve_dataset_column() for fast PyArrow access that respects filtering.
        return id_to_idx(
            resolve_dataset_column(self.query_dataset, self.query_id_column_name),
            "Mapping query ids to indices",
            enable_tqdm,
        )

    @cached_property
    def corpus_dataset_id_to_idx(self) -> dict[str, int]:
        """Create a mapping from document IDs to their indices in the corpus dataset."""
        enable_tqdm: bool = self.rank_id == 0 and self.worker_id == 0
        # Use resolve_dataset_column() for fast PyArrow access that respects filtering.
        return id_to_idx(
            resolve_dataset_column(self.corpus_dataset, self.corpus_id_column_name),
            "Mapping corpus ids to indices",
            enable_tqdm,
        )

    # --- Protected methods ---
    @abc.abstractmethod
    def _resolve_meta_dataset(self) -> Dataset:
        """Return the metadata dataset for this dataset type."""
        raise NotImplementedError

    def _ensure_hf_enabled(self) -> None:
        if self.hf_name is None and self.query_corpus_hf_name is None:
            raise RuntimeError("HuggingFace datasets are disabled (hf_name is null).")

    def _normalize_optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized: str = value.strip().lower()
            if normalized in {"", "none", "null"}:
                return None
        return str(value)

    def _load_hf_dataset(
        self,
        hf_name: str,
        hf_subset: str | None,
        split: str,
        cache_dir: str | None,
        data_files: Mapping[str, Any] | None,
    ) -> Dataset:
        if data_files:
            return load_dataset(
                hf_name,
                name=hf_subset,
                split=split,
                cache_dir=cache_dir,
                data_files=dict(data_files),
            )
        return load_dataset(
            hf_name,
            name=hf_subset,
            split=split,
            cache_dir=cache_dir,
        )

    def _apply_hf_sample_window(self, dataset: Dataset) -> Dataset:
        skip_samples: int = int(self.hf_skip_samples)
        max_samples: int | None = self.hf_max_samples
        if skip_samples <= 0 and max_samples is None:
            return dataset
        dataset_length: int = int(len(dataset))
        start_index: int = min(skip_samples, dataset_length)
        end_index: int = dataset_length
        if max_samples is not None:
            end_index = min(start_index + int(max_samples), dataset_length)
        indices: range = range(start_index, end_index)
        return dataset.select(indices)

    def _load_local_triplets(self) -> Dataset:
        if self.local_triplets_dir is None:
            raise ValueError("local_triplets_dir must be set for local triplets.")
        raw_path: str = os.path.join(self.local_triplets_dir, "raw.tsv")
        if not os.path.isfile(raw_path):
            raise FileNotFoundError(f"Missing local triplets file: {raw_path}")
        rows: list[dict[str, Any]] = []
        with open(raw_path, "r", encoding="utf-8") as reader:
            for row_idx, line in enumerate(reader):
                parsed: tuple[str, str, str, str] | None = self._parse_triplet_line(
                    line, row_idx
                )
                if parsed is None:
                    continue
                qid: str
                query_text: str
                pos_text: str
                neg_text: str
                qid, query_text, pos_text, neg_text = parsed
                rows.append(
                    {
                        "query_id": qid,
                        "query": query_text,
                        "positive": pos_text,
                        "negative": neg_text,
                    }
                )
        return Dataset.from_list(rows)

    def _parse_triplet_line(
        self, line: str, row_idx: int
    ) -> tuple[str, str, str, str] | None:
        stripped: str = line.strip()
        if not stripped:
            return None
        parts: list[str] = stripped.split("\t")
        if len(parts) == 3:
            query_text: str
            pos_text: str
            neg_text: str
            query_text, pos_text, neg_text = parts
            qid: str = str(row_idx)
        elif len(parts) == 4:
            qid = parts[0].strip()
            query_text = parts[1]
            pos_text = parts[2]
            neg_text = parts[3]
        else:
            return None
        return qid.strip(), query_text.strip(), pos_text.strip(), neg_text.strip()

    def _parse_inline_scores(
        self, score_values: Any, doc_ids: list[str]
    ) -> list[float] | None:
        if score_values is None:
            return None
        if isinstance(score_values, (list, tuple)):
            if len(score_values) == len(doc_ids):
                return [float(score) for score in score_values]
            return None
        if isinstance(score_values, (int, float)):
            if len(doc_ids) == 1:
                return [float(score_values)]
            return None
        return None

    def _sample_items(
        self, items: list[Any], count: int, rng: random.Random
    ) -> list[Any]:
        if count <= 0:
            return []
        if len(items) <= count:
            return items
        return rng.sample(items, count)

    def _get_query_text_from_id(self, qid: str) -> str:
        return self.query_text(self.query_dataset_id_to_idx[qid])

    def _get_corpus_text_from_id(self, doc_id: str) -> str:
        return self.corpus_text(self.corpus_dataset_id_to_idx[doc_id])

    def _row_to_meta_item(
        self,
        row: dict[str, Any],
        index: int,
        *,
        num_positives: int,
        num_negatives: int,
        rng: random.Random,
    ) -> MetaItem:
        score_values: Any | None = row.get("score") or row.get("scores")
        qid: str = ""
        pos_ids: list[str] = []
        neg_ids: list[str] = []
        pos_scores: list[float] | None = None
        neg_scores: list[float] | None = None
        query_text: str | None = None
        pos_texts: list[str] | None = None
        neg_texts: list[str] | None = None

        if "query" in row and "positive" in row and "negative" in row:
            query_text = str(row["query"])
            pos_texts = [str(row["positive"])]
            neg_texts = [str(row["negative"])]
            qid = str(row.get("query_id") or row.get("qid") or index)
            pos_ids = [""]
            neg_ids = [""]
        elif "anchor" in row and "positive" in row and "negative" in row:
            query_text = str(row["anchor"])
            pos_texts = [str(row["positive"])]
            neg_texts = [str(row["negative"])]
            qid = str(row.get("query_id") or row.get("qid") or index)
            pos_ids = [""]
            neg_ids = [""]
        elif "query_id" in row and "positive_id" in row:
            qid = str(row["query_id"])
            pos_ids = [str(row.get("positive_id") or "")]
            neg_ids = [str(row.get("negative_id") or "")]
        elif "query_id" in row and "doc_ids" in row and "labels" in row:
            qid = str(row["query_id"])
            row_doc_ids = [str(doc_id) for doc_id in row["doc_ids"]]
            labels = [float(value) for value in row["labels"]]
            pos_ids = [
                doc_id for doc_id, label in zip(row_doc_ids, labels) if label > 0
            ]
            neg_ids = [
                doc_id for doc_id, label in zip(row_doc_ids, labels) if label <= 0
            ]
            pos_ids = self._sample_items(pos_ids, num_positives, rng)
            neg_ids = self._sample_items(neg_ids, num_negatives, rng)
            if isinstance(score_values, (list, tuple)) and len(score_values) == len(
                row_doc_ids
            ):
                score_map = {
                    doc_id: float(score)
                    for doc_id, score in zip(row_doc_ids, score_values)
                }
                pos_scores = [score_map.get(doc_id, float("nan")) for doc_id in pos_ids]
                neg_scores = [score_map.get(doc_id, float("nan")) for doc_id in neg_ids]
        else:
            raise ValueError(f"Unsupported dataset row format: {row.keys()}")

        if pos_scores is None or neg_scores is None:
            doc_ids: list[str] = pos_ids + neg_ids
            inline_scores: list[float] | None = self._parse_inline_scores(
                score_values, doc_ids
            )
            if inline_scores is not None and len(inline_scores) == len(doc_ids):
                pos_scores = inline_scores[: len(pos_ids)]
                neg_scores = inline_scores[len(pos_ids) :]

        return MetaItem(
            qid=str(qid),
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            query_text=query_text,
            pos_texts=pos_texts,
            neg_texts=neg_texts,
        )

    def _loading(
        self,
        logger: logging.Logger,
        subject: str,
        *,
        loading_msg: str | None = None,
        done_msg: str | None = None,
        only_once: bool = False,
    ) -> ContextManager[None]:
        return loading_status(
            logger=logger,
            subject=subject,
            loading_msg=loading_msg,
            done_msg=done_msg,
            only_once=only_once,
            rank_id=self.rank_id,
            worker_id=self.worker_id,
        )

    # --- Public methods ---
    def resolve_query_text(self, meta_item: MetaItem) -> str:
        if meta_item.query_text is not None:
            return meta_item.query_text
        try:
            return self._get_query_text_from_id(meta_item.qid)
        except KeyError:
            return ""

    def resolve_doc_texts(
        self, doc_ids: list[str], inline_texts: list[str] | None
    ) -> list[str]:
        if inline_texts is not None:
            return [str(text) for text in inline_texts]
        texts: list[str] = []
        for doc_id in doc_ids:
            if not doc_id:
                texts.append("")
                continue
            try:
                texts.append(self._get_corpus_text_from_id(doc_id))
            except KeyError:
                texts.append("")
        return texts

    def build_meta_item(
        self,
        row: dict[str, Any],
        index: int,
        *,
        num_positives: int,
        num_negatives: int,
        rng: random.Random,
        load_teacher_scores: bool,
        require_teacher_scores: bool,
    ) -> MetaItem:
        _ = load_teacher_scores
        meta_item: MetaItem = self._row_to_meta_item(
            row,
            index,
            num_positives=num_positives,
            num_negatives=num_negatives,
            rng=rng,
        )
        pos_scores: list[float] | None = meta_item.pos_scores
        neg_scores: list[float] | None = meta_item.neg_scores
        if require_teacher_scores:
            if pos_scores is None or neg_scores is None:
                raise ValueError(f"Missing teacher scores for query {meta_item.qid}")
            if any(score != score for score in pos_scores + neg_scores):
                raise ValueError(f"Missing teacher scores for query {meta_item.qid}")

        return MetaItem(
            qid=meta_item.qid,
            pos_ids=meta_item.pos_ids,
            neg_ids=meta_item.neg_ids,
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            query_text=meta_item.query_text,
            pos_texts=meta_item.pos_texts,
            neg_texts=meta_item.neg_texts,
        )

    def prepare_meta_dataset(self) -> None:
        _ = self.meta_dataset

    def prepare_text_datasets(self) -> None:
        _ = self.query_dataset
        _ = self.corpus_dataset

    def query_text(self, idx: int) -> str:
        """Get the text of a query."""
        raw_value: Any = self.query_dataset[idx][self.query_text_column_name]
        return "" if raw_value is None else str(raw_value)

    def corpus_text(self, idx: int) -> str:
        """Get the text of a document in the corpus, including titles when present."""
        title_column_name: str | None = self.corpus_title_column_name
        title_value: Any | None = (
            self.corpus_dataset[idx][title_column_name]
            if title_column_name is not None
            else None
        )
        text_value: Any = self.corpus_dataset[idx][self.corpus_text_column_name]
        title: str = "" if title_value is None else str(title_value)
        text: str = "" if text_value is None else str(text_value)
        if title:
            return f"{title} {text}".strip()
        return text.strip()

    def download_data(self) -> None:
        """Download the dataset from HuggingFace Hub."""
        snapshot_download(repo_id=self.huggingface_name, repo_type="dataset")
