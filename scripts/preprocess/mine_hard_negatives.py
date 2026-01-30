import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

import faiss
import hydra
import numpy as np
import torch
from datasets import Dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from config.path import ABS_CONFIG_DIR
from src.data.dataset.msmarco import MSMARCO
from src.data.dataset.retrieval import RetrievalDataset
from src.data.dataclass import Document, Query
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils import set_seed
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.model_utils import build_splade_model, load_splade_checkpoint
from src.utils.script_setup import configure_script_environment
from src.utils.transformers import build_tokenizer

logger: logging.Logger = get_logger("scripts.preprocess.mine_hard_negatives", __file__)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=True,
    suppress_lightning_tips=False,
    suppress_httpx=False,
    suppress_dataloader_workers=True,
)


@dataclass(frozen=True)
class MiningSettings:
    """Typed container for mining settings."""

    output_dir: str
    output_basename: str
    output_format: str
    negatives_per_query: int
    top_k: int
    max_queries: int | None
    max_docs: int | None
    query_batch_size: int
    doc_batch_size: int
    use_cpu: bool
    faiss_index_type: str
    faiss_use_gpu: bool
    faiss_gpu_device: int
    normalize_embeddings: bool
    checkpoint_path: str | None
    use_qrels: bool


def _parse_mining_settings(cfg: DictConfig) -> MiningSettings:
    """Parse Hydra config into a typed MiningSettings instance."""
    mining_cfg: DictConfig = cfg.mining
    output_dir: str = str(mining_cfg.output_dir)
    output_basename: str = str(mining_cfg.output_basename)
    output_format: str = str(mining_cfg.output_format).lower()
    negatives_per_query: int = int(mining_cfg.negatives_per_query)
    top_k: int = int(mining_cfg.top_k)
    max_queries: int | None = (
        None if mining_cfg.max_queries is None else int(mining_cfg.max_queries)
    )
    max_docs: int | None = (
        None if mining_cfg.max_docs is None else int(mining_cfg.max_docs)
    )
    query_batch_size: int = int(mining_cfg.query_batch_size)
    doc_batch_size: int = int(mining_cfg.doc_batch_size)
    use_cpu: bool = bool(mining_cfg.use_cpu)
    faiss_index_type: str = str(mining_cfg.faiss_index_type)
    faiss_use_gpu: bool = bool(mining_cfg.faiss_use_gpu)
    faiss_gpu_device: int = int(mining_cfg.faiss_gpu_device)
    normalize_embeddings: bool = bool(mining_cfg.normalize_embeddings)
    checkpoint_path: str | None = (
        None
        if mining_cfg.checkpoint_path in (None, "")
        else str(mining_cfg.checkpoint_path)
    )
    use_qrels: bool = bool(mining_cfg.use_qrels)
    return MiningSettings(
        output_dir=output_dir,
        output_basename=output_basename,
        output_format=output_format,
        negatives_per_query=negatives_per_query,
        top_k=top_k,
        max_queries=max_queries,
        max_docs=max_docs,
        query_batch_size=query_batch_size,
        doc_batch_size=doc_batch_size,
        use_cpu=use_cpu,
        faiss_index_type=faiss_index_type,
        faiss_use_gpu=faiss_use_gpu,
        faiss_gpu_device=faiss_gpu_device,
        normalize_embeddings=normalize_embeddings,
        checkpoint_path=checkpoint_path,
        use_qrels=use_qrels,
    )


def _resolve_device(use_cpu: bool) -> torch.device:
    """Resolve the device for mining computations."""
    device_type: str = "cpu" if use_cpu else "cuda"
    return torch.device(device_type)


def _build_faiss_index(
    dimension: int, index_type: str, use_gpu: bool, gpu_device: int
) -> faiss.Index:
    """Build a FAISS index for inner-product search."""
    base_index: faiss.Index = faiss.index_factory(
        dimension, index_type, faiss.METRIC_INNER_PRODUCT
    )
    index: faiss.Index = faiss.IndexIDMap2(base_index)
    gpu_resources_cls: Any = getattr(faiss, "StandardGpuResources", None)
    index_cpu_to_gpu: Any = getattr(faiss, "index_cpu_to_gpu", None)
    if use_gpu and gpu_resources_cls is not None and index_cpu_to_gpu is not None:
        try:
            resources: Any = gpu_resources_cls()
            index = index_cpu_to_gpu(resources, gpu_device, index)
        except Exception as exc:  # pylint: disable=broad-except
            log_if_rank_zero(
                logger,
                f"FAISS GPU unavailable, falling back to CPU: {exc}",
                level="warning",
            )
    elif use_gpu:
        log_if_rank_zero(
            logger, "FAISS GPU support not detected; using CPU index.", level="warning"
        )
    return index


def _encode_text_batch(
    model: SpladeModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
    device: torch.device,
    mode: str,
    normalize_embeddings: bool,
) -> np.ndarray:
    """Encode a batch of texts into SPLADE vectors."""
    tokens: Dict[str, torch.Tensor] = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids: torch.Tensor = tokens["input_ids"].to(device)
    attention_mask: torch.Tensor = tokens["attention_mask"].to(device)
    with torch.no_grad():
        if mode == "query":
            embeddings: torch.Tensor = model.encode_queries(input_ids, attention_mask)
        else:
            embeddings = model.encode_docs(input_ids, attention_mask)
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().numpy().astype("float32")


def _iter_dataset_rows(
    dataset: Dataset, id_column: str, text_column: str, max_rows: int | None
) -> Iterator[tuple[str, str]]:
    """Yield (id, text) tuples from a dataset with optional truncation."""
    row_count: int = 0
    for row in dataset:
        raw_id: Any = row.get(id_column)
        raw_text: Any = row.get(text_column)
        row_id: str = "" if raw_id is None else str(raw_id)
        text: str = "" if raw_text is None else str(raw_text)
        yield row_id, text
        row_count += 1
        if max_rows is not None and row_count >= max_rows:
            break


def _iter_batches(
    rows: Iterable[tuple[str, str]], batch_size: int
) -> Iterator[tuple[list[str], list[str]]]:
    """Batch (id, text) rows into fixed-size chunks."""
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    for row_id, text in rows:
        batch_ids.append(row_id)
        batch_texts.append(text)
        if len(batch_ids) >= batch_size:
            yield batch_ids, batch_texts
            batch_ids = []
            batch_texts = []
    if batch_ids:
        yield batch_ids, batch_texts


def _build_positives_from_triplets(
    triplet_rows: Iterable[Mapping[str, Any]], max_samples: int | None
) -> dict[str, set[str]]:
    """Build query->positive-id mapping from triplet-style rows."""
    positives_by_qid: dict[str, set[str]] = {}
    row_count: int = 0
    for row in triplet_rows:
        qid: str = str(row.get("query_id") or row.get("qid") or row.get("_id") or "")
        pos_id: str = str(
            row.get("positive_id") or row.get("pos_id") or row.get("doc_pos_id") or ""
        )
        if qid and pos_id:
            positives_by_qid.setdefault(qid, set()).add(pos_id)
        row_count += 1
        if max_samples is not None and row_count >= max_samples:
            break
    return positives_by_qid


def _build_positives_from_qrels(
    qrels: dict[str, dict[str, float]],
) -> dict[str, set[str]]:
    """Build query->positive-id mapping from qrels."""
    positives_by_qid: dict[str, set[str]] = {}
    for qid, doc_scores in qrels.items():
        positive_ids: set[str] = {
            doc_id for doc_id, score in doc_scores.items() if score > 0
        }
        if positive_ids:
            positives_by_qid[qid] = positive_ids
    return positives_by_qid


def _select_positive_id(positive_ids: set[str]) -> str | None:
    """Pick a stable positive id for a query."""
    if not positive_ids:
        return None
    sorted_ids: list[str] = sorted(positive_ids)
    return sorted_ids[0]


def _write_triplet_rows(
    output_path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """Write JSONL records to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name="mine_hard_negatives"
)
def main(cfg: DictConfig) -> None:
    settings: MiningSettings = _parse_mining_settings(cfg)
    set_seed(int(cfg.seed))

    device: torch.device = _resolve_device(settings.use_cpu)
    tokenizer: PreTrainedTokenizerBase = build_tokenizer(cfg.model.huggingface_name)
    model: SpladeModel = build_splade_model(cfg, use_cpu=settings.use_cpu)
    model = model.to(device)
    model.eval()

    if not settings.checkpoint_path:
        raise ValueError(
            "Mining requires a trained SPLADEv1 checkpoint. Set mining.checkpoint_path."
        )
    missing_keys: list[str]
    unexpected_keys: list[str]
    missing_keys, unexpected_keys = load_splade_checkpoint(
        model, settings.checkpoint_path
    )
    log_if_rank_zero(
        logger,
        f"Loaded checkpoint. Missing: {len(missing_keys)}, unexpected: "
        f"{len(unexpected_keys)}",
    )

    positives_by_qid: dict[str, set[str]]
    if settings.use_qrels:
        retrieval_cfg: DictConfig = cfg.retrieval_dataset
        retrieval_dataset: RetrievalDataset = RetrievalDataset(
            cfg=retrieval_cfg, global_cfg=cfg, tokenizer=tokenizer
        )
        retrieval_dataset.prepare_data()
        retrieval_dataset.setup()
        positives_by_qid: dict[str, set[str]] = _build_positives_from_qrels(
            retrieval_dataset.qrels_dict
        )
        query_rows: Iterable[Query] = retrieval_dataset.queries
        corpus_rows: Iterable[Document] = retrieval_dataset.corpus
        query_max_length: int = int(retrieval_cfg.max_query_length)
        doc_max_length: int = int(retrieval_cfg.max_doc_length)
        query_iter: Iterable[tuple[str, str]] = (
            (query.query_id, query.text) for query in query_rows
        )
        corpus_iter: Iterable[tuple[str, str]] = (
            (doc.doc_id, doc.text) for doc in corpus_rows
        )
    else:
        train_cfg: DictConfig = cfg.train_dataset
        msmarco_dataset: MSMARCO = MSMARCO(
            cfg=train_cfg,
            global_cfg=cfg,
            tokenizer=tokenizer,
            load_teacher_scores=False,
            require_teacher_scores=False,
        )
        msmarco_dataset.prepare_data()
        msmarco_dataset.setup()
        max_triplets: int | None = (
            None if train_cfg.hf_max_samples is None else int(train_cfg.hf_max_samples)
        )
        positives_by_qid = _build_positives_from_triplets(
            msmarco_dataset.dataset, max_triplets
        )
        query_max_length = int(train_cfg.max_query_length)
        doc_max_length = int(train_cfg.max_doc_length)
        query_iter = _iter_dataset_rows(
            msmarco_dataset.query_dataset,
            msmarco_dataset.query_id_column,
            msmarco_dataset.query_text_column,
            settings.max_queries,
        )
        corpus_iter = _iter_dataset_rows(
            msmarco_dataset.corpus_dataset,
            msmarco_dataset.corpus_id_column,
            msmarco_dataset.corpus_text_column,
            settings.max_docs,
        )

    vocab_size: int = int(model.encoder.mlm.config.vocab_size)
    index: faiss.Index = _build_faiss_index(
        vocab_size,
        settings.faiss_index_type,
        settings.faiss_use_gpu,
        settings.faiss_gpu_device,
    )

    doc_ids: list[str] = []
    next_doc_id: int = 0
    log_if_rank_zero(logger, "Encoding corpus and building FAISS index.")
    for batch_ids, batch_texts in _iter_batches(corpus_iter, settings.doc_batch_size):
        batch_vectors: np.ndarray = _encode_text_batch(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            max_length=doc_max_length,
            device=device,
            mode="doc",
            normalize_embeddings=settings.normalize_embeddings,
        )
        batch_size: int = int(len(batch_ids))
        batch_indices: np.ndarray = np.arange(
            next_doc_id, next_doc_id + batch_size, dtype=np.int64
        )
        index.add_with_ids(batch_vectors, batch_indices)
        doc_ids.extend(batch_ids)
        next_doc_id += batch_size

    log_if_rank_zero(
        logger, f"Mining hard negatives for {len(positives_by_qid)} queries."
    )
    output_path: Path = Path(settings.output_dir) / f"{settings.output_basename}.jsonl"

    def _record_iterator() -> Iterator[dict[str, Any]]:
        query_batches: Iterator[tuple[list[str], list[str]]] = _iter_batches(
            query_iter, settings.query_batch_size
        )
        for batch_ids, batch_texts in query_batches:
            filtered_ids: list[str] = []
            filtered_texts: list[str] = []
            for qid, text in zip(batch_ids, batch_texts):
                if qid in positives_by_qid:
                    filtered_ids.append(qid)
                    filtered_texts.append(text)
            if not filtered_ids:
                continue
            query_vectors: np.ndarray = _encode_text_batch(
                model=model,
                tokenizer=tokenizer,
                texts=filtered_texts,
                max_length=query_max_length,
                device=device,
                mode="query",
                normalize_embeddings=settings.normalize_embeddings,
            )
            _scores: np.ndarray
            indices: np.ndarray
            _scores, indices = index.search(query_vectors, settings.top_k)
            for row_idx, qid in enumerate(filtered_ids):
                positive_ids: set[str] = positives_by_qid.get(qid, set())
                positive_id: str | None = _select_positive_id(positive_ids)
                if positive_id is None:
                    continue
                negatives: list[str] = []
                for candidate in indices[row_idx]:
                    if int(candidate) < 0:
                        continue
                    doc_id: str = doc_ids[int(candidate)]
                    if doc_id in positive_ids:
                        continue
                    negatives.append(doc_id)
                    if len(negatives) >= settings.negatives_per_query:
                        break
                if not negatives:
                    continue
                if settings.output_format == "triplet":
                    for neg_id in negatives:
                        yield {
                            "query_id": qid,
                            "positive_id": positive_id,
                            "negative_id": neg_id,
                        }
                else:
                    doc_id_list: list[str] = [positive_id] + negatives
                    labels: list[int] = [1] + [0 for _ in negatives]
                    yield {
                        "query_id": qid,
                        "doc_ids": doc_id_list,
                        "labels": labels,
                    }

    _write_triplet_rows(output_path, _record_iterator())
    log_if_rank_zero(logger, f"Saved mined negatives to {output_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
