import logging
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
from src.utils.dist import is_rank_zero, maybe_barrier
from src.utils.logging import log_if_rank_zero

logger: logging.Logger = logging.getLogger("RetrievalPDModule")


class RetrievalPDModule(PDModule):
    """Retrieval PyTorch datasets module for evaluation/inference."""

    # --- Special methods ---
    @staticmethod
    def _normalize_optional_str(value: Any | None) -> str | None:
        if value is None:
            return None
        normalized: str = str(value).strip()
        return normalized if normalized else None

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
        self._use_qrels: bool = bool(getattr(self.cfg, "use_qrels", True))
        self._use_triplet_positives: bool = bool(
            getattr(self.cfg, "use_triplet_positives", False)
        )
        cache_path_value: str | None = self._normalize_optional_str(
            getattr(self.cfg, "triplet_positive_cache_path", None)
        )
        self._triplet_positive_cache_path: str | None = cache_path_value
        self._triplet_positive_cache_overwrite: bool = bool(
            getattr(self.cfg, "triplet_positive_cache_overwrite", False)
        )
        max_queries_value: Any | None = getattr(self.cfg, "max_queries", None)
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
            str(getattr(self.cfg, "type", "")).lower() == "beir"
            or self.cfg.get("beir_dataset") is not None
        )
        text_cache_dir: str | None = self.cfg.query_corpus_hf_cache_dir
        self.hf_cache_dir: str | None = (
            None
            if text_cache_dir is None and self.cfg.hf_cache_dir is None
            else str(
                text_cache_dir if text_cache_dir is not None else self.cfg.hf_cache_dir
            )
        )
        self.qrels_hf_name: str | None = self._normalize_optional_str(
            getattr(self.cfg, "qrels_hf_name", None)
        )
        self.qrels_hf_subset: str | None = self._normalize_optional_str(
            getattr(self.cfg, "qrels_hf_subset", None)
        )
        qrels_split_override: str | None = self._normalize_optional_str(
            getattr(self.cfg, "qrels_hf_split", None)
        )
        self.qrels_hf_split: str = qrels_split_override or self.hf_split
        qrels_cache_override: str | None = self._normalize_optional_str(
            getattr(self.cfg, "qrels_hf_cache_dir", None)
        )
        self.qrels_hf_cache_dir: str | None = (
            qrels_cache_override if qrels_cache_override is not None else self.hf_cache_dir
        )
        self.qrels_hf_data_files: Any | None = getattr(
            self.cfg, "qrels_hf_data_files", None
        )
        self._query_ids: list[str] = []
        self._query_id_to_idx: dict[str, int] = {}
        self._qrels: Dict[str, Dict[str, float]] = {}
        self._qrels_dataset: Dataset | None = None

    def __len__(self) -> int:
        self._ensure_query_index()
        return len(self._query_ids)

    def __getitem__(self, idx: int) -> RetrievalDataItem:
        self._ensure_query_index()
        qid: str = self._query_ids[int(idx)]
        query_idx: int = self._query_id_to_idx[qid]
        query_text: str = self.dataset.query_text(query_idx)
        tokens: dict[str, torch.Tensor] = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        query_input_ids: torch.Tensor = tokens["input_ids"].squeeze(0)
        query_attention_mask: torch.Tensor = tokens["attention_mask"].squeeze(0)
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

    def _build_positives_from_triplets(
        self,
        triplet_rows: Iterable[Mapping[str, Any]],
        allowed_queries: set[str],
        *,
        enable_progress: bool,
    ) -> tuple[Dict[str, Dict[str, float]], int, int]:
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
                row.get("positive_id") or row.get("pos_id") or row.get("doc_pos_id") or ""
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
            return self._load_hf_split(qrels_name, "qrels", qrels_split, qrels_cache_dir)

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
                self._query_ids = [qid for qid in self._query_ids if qid in qrels_query_ids]

    def get_relevance_judgments(self, qid: str) -> Dict[str, float]:
        return self._qrels.get(qid, {})
