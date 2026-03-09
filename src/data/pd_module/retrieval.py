import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.collator import UniversalCollator
from src.data.dataclass import RetrievalDataItem
from src.data.pd_module import PDModule
from src.data.pd_module.utils import tokenize_text, tokenize_text_windows
from src.data.utils import resolve_dataset_column
from src.utils.dist import is_rank_zero, maybe_barrier
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.normalize import normalize_optional_str

logger = get_logger("RetrievalPDModule")


class RetrievalPDModule(PDModule):
    """Retrieval PyTorch datasets module for evaluation/inference."""

    # --- Special methods ---
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        *,
        seed: int,
        load_teacher_scores: bool | None = None,
        require_teacher_scores: bool | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            tokenizer=tokenizer,
            seed=seed,
            load_teacher_scores=load_teacher_scores,
            require_teacher_scores=require_teacher_scores,
        )
        self._use_qrels: bool = bool(self.cfg.use_qrels)
        self._use_triplet_positives: bool = bool(self.cfg.use_triplet_positives)
        self._filter_queries_with_positives: bool = bool(
            self.cfg.filter_queries_with_positives
        )
        cache_path_value: str | None = normalize_optional_str(
            self.cfg.triplet_positive_cache_path
        )
        self._triplet_positive_cache_path: str | None = cache_path_value
        self._triplet_positive_cache_overwrite: bool = bool(
            self.cfg.triplet_positive_cache_overwrite
        )
        max_queries_value: Any | None = self.cfg.max_queries
        self._max_queries: int | None = (
            None if max_queries_value is None else int(max_queries_value)
        )
        self.hf_name: str = (
            str(self.cfg.query_corpus_hf_name)
            if self.cfg.query_corpus_hf_name is not None
            else str(self.cfg.hf_name)
        )
        self.hf_split: str = str(self.cfg.split)
        self._beir_mode: bool = (
            str(self.cfg.type).lower() == "beir" or self.cfg.beir_dataset is not None
        )
        text_cache_dir: str | None = self.cfg.query_corpus_hf_cache_dir
        self.hf_cache_dir: str | None = (
            None
            if text_cache_dir is None and self.cfg.hf_cache_dir is None
            else str(
                text_cache_dir if text_cache_dir is not None else self.cfg.hf_cache_dir
            )
        )
        self.qrels_hf_name: str | None = normalize_optional_str(
            self.cfg.qrels_hf_name
        )
        self.qrels_hf_subset: str | None = normalize_optional_str(
            self.cfg.qrels_hf_subset
        )
        qrels_split_override: str | None = normalize_optional_str(
            self.cfg.qrels_hf_split
        )
        self.qrels_hf_split: str = qrels_split_override or self.hf_split
        qrels_cache_override: str | None = normalize_optional_str(
            self.cfg.qrels_hf_cache_dir
        )
        self.qrels_hf_cache_dir: str | None = (
            qrels_cache_override
            if qrels_cache_override is not None
            else self.hf_cache_dir
        )
        self.qrels_hf_data_files: Any | None = self.cfg.qrels_hf_data_files
        self._query_ids: list[str] = []
        self._query_id_to_idx: dict[str, int] = {}
        self._qrels: Dict[str, Dict[str, float]] = {}
        self._qrels_dataset: Dataset | None = None
        long_query_strategy_value: str = str(
            self.cfg.get("query_long_doc_strategy", "truncate")
        ).lower()
        if long_query_strategy_value not in {"truncate", "sliding_window"}:
            raise ValueError(
                "dataset.query_long_doc_strategy must be 'truncate' or "
                f"'sliding_window'. Got: {long_query_strategy_value}"
            )
        self._query_long_doc_strategy: str = long_query_strategy_value
        self._query_sliding_window_overlap_tokens: int = max(
            0, int(self.cfg.get("query_sliding_window_overlap_tokens", 0))
        )

    def __len__(self) -> int:
        self._ensure_query_index()
        return len(self._query_ids)

    def __getitem__(self, idx: int) -> RetrievalDataItem:
        self._ensure_query_index()
        qid: str = self._query_ids[int(idx)]
        query_idx: int = self._query_id_to_idx[qid]
        query_text: str = self.dataset.query_text(query_idx)
        query_input_ids: torch.Tensor
        query_attention_mask: torch.Tensor
        if self._query_long_doc_strategy == "sliding_window":
            query_input_ids, query_attention_mask = tokenize_text_windows(
                self.tokenizer,
                query_text,
                max_length=self.max_query_length,
                max_padding=self.max_padding,
                overlap_tokens=self._query_sliding_window_overlap_tokens,
            )
        else:
            single_input_ids: torch.Tensor
            single_attention_mask: torch.Tensor
            single_input_ids, single_attention_mask = tokenize_text(
                self.tokenizer,
                query_text,
                max_length=self.max_query_length,
                max_padding=self.max_padding,
            )
            query_input_ids = single_input_ids.unsqueeze(0)
            query_attention_mask = single_attention_mask.unsqueeze(0)
        return RetrievalDataItem(
            data_idx=int(idx),
            qid=qid,
            relevance_judgments=self.get_relevance_judgments(qid),
            query_text=query_text,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
        )

    # --- Property methods ---
    @property
    def collator(self) -> UniversalCollator:
        return UniversalCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_padding=self.max_padding,
            max_query_length=self.max_query_length,
        )

    @property
    def qrels_dict(self) -> Dict[str, Dict[str, float]]:
        return self._qrels

    # --- Protected methods ---
    def _ensure_query_index(self) -> None:
        if self._query_ids:
            return
        self._query_id_to_idx = dict(self.dataset.query_dataset_id_to_idx)
        self._query_ids = list(self._query_id_to_idx.keys())
        if self._max_queries is not None and self._max_queries > 0:
            self._query_ids = self._query_ids[: self._max_queries]

    def _resolve_triplet_positive_cache_path(
        self, *, required: bool = False
    ) -> Path | None:
        cache_path_value: str | None = self._triplet_positive_cache_path
        if cache_path_value is None:
            if required:
                raise ValueError(
                    "dataset.triplet_positive_cache_path must be set when "
                    "dataset.use_triplet_positives=true."
                )
            return None
        return Path(cache_path_value)

    def _load_triplet_positive_cache(
        self, cache_path: Path
    ) -> Dict[str, Dict[str, float]]:
        table: pa.Table = pq.read_table(cache_path, columns=["qid", "doc_ids"])
        qids: list[str] = [str(qid) for qid in table.column("qid").to_pylist()]
        doc_ids_list: list[list[str] | None] = table.column("doc_ids").to_pylist()
        positives: Dict[str, Dict[str, float]] = {}
        for qid, doc_ids in zip(qids, doc_ids_list):
            if not qid or not doc_ids:
                continue
            positives[qid] = {str(doc_id): 1.0 for doc_id in doc_ids}
        return positives

    def _write_triplet_positive_cache(
        self, cache_path: Path, positives: Dict[str, Dict[str, float]]
    ) -> None:
        qids: list[str] = []
        doc_ids_list: list[list[str]] = []
        for qid in sorted(positives):
            doc_map: Dict[str, float] = positives[qid]
            if not doc_map:
                continue
            qids.append(str(qid))
            doc_ids_list.append(sorted(str(doc_id) for doc_id in doc_map.keys()))
        table: pa.Table = pa.Table.from_pydict(
            {"qid": qids, "doc_ids": doc_ids_list},
            schema=pa.schema(
                [("qid", pa.string()), ("doc_ids", pa.list_(pa.string()))]
            ),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path = cache_path.with_name(f"{cache_path.name}.tmp")
        pq.write_table(table, tmp_path)
        tmp_path.replace(cache_path)

    @staticmethod
    def _resolve_iterable_len(value: Iterable[Mapping[str, Any]]) -> int | None:
        try:
            return len(value)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            return None

    @staticmethod
    def _resolve_triplet_column_name(
        column_names: Iterable[str], candidates: Iterable[str]
    ) -> str | None:
        column_set: set[str] = set(column_names)
        for candidate in candidates:
            if candidate in column_set:
                return candidate
        return None

    def _build_positives_from_triplet_columns(
        self,
        triplet_dataset: Dataset,
        qid_column: str,
        pos_column: str,
        allowed_queries: set[str],
        *,
        enable_progress: bool,
    ) -> tuple[Dict[str, Dict[str, float]], int, int] | None:
        qid_values: pa.Array | pa.ChunkedArray = resolve_dataset_column(
            triplet_dataset, qid_column
        )
        pos_values: pa.Array | pa.ChunkedArray = resolve_dataset_column(
            triplet_dataset, pos_column
        )

        qid_chunks: list[pa.Array] = (
            qid_values.chunks
            if isinstance(qid_values, pa.ChunkedArray)
            else [qid_values]
        )
        pos_chunks: list[pa.Array] = (
            pos_values.chunks
            if isinstance(pos_values, pa.ChunkedArray)
            else [pos_values]
        )
        if len(qid_chunks) != len(pos_chunks):
            return None

        positives_by_qid: dict[str, set[str]] = {}
        row_count: int = 0
        positive_pairs: int = 0
        progress: tqdm | None = None
        if enable_progress:
            progress = tqdm(
                total=len(triplet_dataset),
                desc="Scanning triplet positives",
                mininterval=30.0,
            )

        for qid_chunk, pos_chunk in zip(qid_chunks, pos_chunks):
            qids: list[Any] = qid_chunk.to_pylist()
            pos_ids: list[Any] = pos_chunk.to_pylist()
            row_count += len(qids)
            if progress is not None:
                progress.update(len(qids))

            for qid_value, pos_value in zip(qids, pos_ids):
                if qid_value is None or pos_value is None:
                    continue
                qid: str = str(qid_value)
                if not qid or qid not in allowed_queries:
                    continue
                pos_id: str = str(pos_value)
                if not pos_id:
                    continue
                doc_set: set[str] = positives_by_qid.setdefault(qid, set())
                if pos_id not in doc_set:
                    doc_set.add(pos_id)
                    positive_pairs += 1

        if progress is not None:
            progress.close()

        positives: Dict[str, Dict[str, float]] = {
            qid: {doc_id: 1.0 for doc_id in doc_ids}
            for qid, doc_ids in positives_by_qid.items()
        }
        return positives, row_count, positive_pairs

    def _build_positives_from_triplets(
        self,
        triplet_rows: Iterable[Mapping[str, Any]],
        allowed_queries: set[str],
        *,
        enable_progress: bool,
    ) -> tuple[Dict[str, Dict[str, float]], int, int]:
        if isinstance(triplet_rows, Dataset):
            qid_column: str | None = self._resolve_triplet_column_name(
                triplet_rows.column_names, ("query_id", "qid", "_id")
            )
            pos_column: str | None = self._resolve_triplet_column_name(
                triplet_rows.column_names, ("positive_id", "pos_id", "doc_pos_id")
            )
            if qid_column and pos_column:
                column_result = self._build_positives_from_triplet_columns(
                    triplet_rows,
                    qid_column,
                    pos_column,
                    allowed_queries,
                    enable_progress=enable_progress,
                )
                if column_result is not None:
                    return column_result

        positives_by_qid: dict[str, set[str]] = {}
        row_count: int = 0
        positive_pairs: int = 0
        iterator: Iterable[Mapping[str, Any]] = triplet_rows
        if enable_progress:
            total_rows: int | None = self._resolve_iterable_len(triplet_rows)
            iterator = tqdm(
                triplet_rows,
                total=total_rows,
                desc="Scanning triplet positives",
                mininterval=30.0,
            )
        for raw_row in iterator:
            row: Mapping[str, Any] = raw_row
            row_count += 1
            qid: str = str(
                row.get("query_id") or row.get("qid") or row.get("_id") or ""
            )
            if not qid or qid not in allowed_queries:
                continue
            pos_id: str = str(
                row.get("positive_id")
                or row.get("pos_id")
                or row.get("doc_pos_id")
                or ""
            )
            if not pos_id:
                continue
            doc_set: set[str] = positives_by_qid.setdefault(qid, set())
            if pos_id not in doc_set:
                doc_set.add(pos_id)
                positive_pairs += 1
        positives: Dict[str, Dict[str, float]] = {
            qid: {doc_id: 1.0 for doc_id in doc_ids}
            for qid, doc_ids in positives_by_qid.items()
        }
        return positives, row_count, positive_pairs

    def _load_hf_split(
        self, hf_name: str, config: str, split: str, cache_dir: str | None
    ) -> Dataset:
        if self._beir_mode:
            # BEIR datasets use a single config and expose corpus/queries/qrels as splits.
            return load_dataset(hf_name, split=config, cache_dir=cache_dir)
        return load_dataset(hf_name, config, split=split, cache_dir=cache_dir)

    def _load_qrels_dataset(self) -> Dataset:
        qrels_name: str = self.qrels_hf_name or self.hf_name
        qrels_subset: str | None = self.qrels_hf_subset
        qrels_split: str = self.qrels_hf_split
        qrels_cache_dir: str | None = self.qrels_hf_cache_dir
        qrels_data_files: Any | None = self.qrels_hf_data_files

        if (
            self.qrels_hf_name is None
            and qrels_subset is None
            and qrels_data_files is None
        ):
            return self._load_hf_split(
                qrels_name, "qrels", qrels_split, qrels_cache_dir
            )

        data_files: dict[str, Any] | None = (
            dict(qrels_data_files) if qrels_data_files is not None else None
        )
        if qrels_subset is not None:
            return load_dataset(
                qrels_name,
                qrels_subset,
                split=qrels_split,
                cache_dir=qrels_cache_dir,
                data_files=data_files,
            )
        return load_dataset(
            qrels_name,
            split=qrels_split,
            cache_dir=qrels_cache_dir,
            data_files=data_files,
        )

    # --- Public methods ---
    def prepare_data(self) -> None:
        _ = self.dataset.query_dataset
        if self._use_qrels:
            _ = self._load_qrels_dataset()
            return
        if not self._use_triplet_positives:
            return

        cache_path: Path = self._resolve_triplet_positive_cache_path(required=True)
        if cache_path.exists() and not self._triplet_positive_cache_overwrite:
            log_if_rank_zero(
                logger, f"Triplet positives cache hit: {cache_path.as_posix()}"
            )
            return
        if not is_rank_zero():
            return

        log_if_rank_zero(
            logger,
            "Scanning triplet metadata to build positives cache. This can take a "
            "while on MSMARCO.",
        )
        log_if_rank_zero(
            logger, f"Building triplet positives cache: {cache_path.as_posix()}"
        )
        try:
            self.dataset.prepare_meta_dataset()
        except NotImplementedError as exc:
            raise ValueError(
                "dataset.use_triplet_positives requires a dataset with "
                "triplet metadata."
            ) from exc
        self._ensure_query_index()
        allowed_queries: set[str] = set(self._query_ids)
        start_time: float = time.perf_counter()
        positives: Dict[str, Dict[str, float]]
        row_count: int
        positive_pairs: int
        positives, row_count, positive_pairs = self._build_positives_from_triplets(
            self.dataset.meta_dataset, allowed_queries, enable_progress=True
        )
        self._write_triplet_positive_cache(cache_path, positives)
        elapsed: float = time.perf_counter() - start_time
        log_if_rank_zero(
            logger,
            "Saved triplet positives cache to "
            f"{cache_path.as_posix()} ({len(positives)} queries, "
            f"{positive_pairs} positives from {row_count} rows in {elapsed:.1f}s).",
        )

    def setup(self) -> None:
        self._ensure_query_index()
        if not self._use_qrels:
            self._qrels_dataset = None
            if self._use_triplet_positives:
                cache_path: Path = self._resolve_triplet_positive_cache_path(
                    required=True
                )
                if not cache_path.exists():
                    if is_rank_zero():
                        log_if_rank_zero(
                            logger,
                            "Triplet positives cache missing; rebuilding at "
                            f"{cache_path.as_posix()}.",
                            level="warning",
                        )
                        try:
                            self.dataset.prepare_meta_dataset()
                        except NotImplementedError as exc:
                            raise ValueError(
                                "dataset.use_triplet_positives requires a dataset with "
                                "triplet metadata."
                            ) from exc
                        allowed_queries: set[str] = set(self._query_ids)
                        positives, row_count, positive_pairs = (
                            self._build_positives_from_triplets(
                                self.dataset.meta_dataset,
                                allowed_queries,
                                enable_progress=True,
                            )
                        )
                        self._write_triplet_positive_cache(cache_path, positives)
                        log_if_rank_zero(
                            logger,
                            "Saved triplet positives cache to "
                            f"{cache_path.as_posix()} ({len(positives)} queries, "
                            f"{positive_pairs} positives from {row_count} rows).",
                        )
                    maybe_barrier()
                else:
                    maybe_barrier()

                if not cache_path.exists():
                    raise ValueError(
                        "Triplet positives cache missing after build: "
                        f"{cache_path.as_posix()}."
                    )
                self._qrels = self._load_triplet_positive_cache(cache_path)
                log_if_rank_zero(
                    logger,
                    "Loaded triplet positives cache from "
                    f"{cache_path.as_posix()} ({len(self._qrels)} queries).",
                )
                if self._filter_queries_with_positives:
                    before_count: int = len(self._query_ids)
                    qrels_query_ids: set[str] = set(self._qrels.keys())
                    if qrels_query_ids:
                        self._query_ids = [
                            qid
                            for qid in self._query_ids
                            if qid in qrels_query_ids
                        ]
                    else:
                        self._query_ids = []
                    log_if_rank_zero(
                        logger,
                        "Filtered queries with positives: "
                        f"{before_count} -> {len(self._query_ids)}.",
                    )
            else:
                self._qrels = {}
            return

        self._qrels_dataset = self._load_qrels_dataset()

        self._qrels = {}
        if self._qrels_dataset is not None:
            allowed_queries: set[str] = set(self._query_ids)
            for raw_row in self._qrels_dataset:
                row: dict[str, Any] = raw_row
                qid: str = str(
                    row.get("query-id")
                    or row.get("query_id")
                    or row.get("qid")
                    or row.get("_id")
                )
                if qid not in allowed_queries:
                    continue
                doc_id: str = str(
                    row.get("corpus-id")
                    or row.get("doc_id")
                    or row.get("pid")
                    or row.get("docid")
                )
                score: float = float(
                    row.get("score") or row.get("relevance") or row.get("rel") or 0
                )
                self._qrels.setdefault(qid, {})[doc_id] = score
            # Restrict evaluation to queries with qrels to avoid scoring unlabeled queries.
            qrels_query_ids: set[str] = set(self._qrels.keys())
            if qrels_query_ids:
                self._query_ids = [
                    qid for qid in self._query_ids if qid in qrels_query_ids
                ]

    def get_relevance_judgments(self, qid: str) -> Dict[str, float]:
        return self._qrels.get(qid, {})
