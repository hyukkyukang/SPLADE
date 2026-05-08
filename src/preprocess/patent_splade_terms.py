from __future__ import annotations

import glob
import json
import math
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.data.pl_module.common import build_inference_dataloader
from src.data.lens_formatting import build_doc_pooling_mask
from src.model.pl_module.utils import build_splade_model_with_checkpoint
from src.utils.model_utils import apply_checkpoint_model_config
from src.utils.output_space import OutputSpaceSpec, resolve_model_output_exclude_ids
from src.utils.transformers import build_tokenizer

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_DOCUMENT_ENCODING_MODE: str = "split_fields_windowed"
COMBINED_TRUNCATE_DOCUMENT_ENCODING_MODE: str = "combined_fields_truncate_head"
VALID_DOCUMENT_ENCODING_MODES: tuple[str, ...] = (
    DEFAULT_DOCUMENT_ENCODING_MODE,
    COMBINED_TRUNCATE_DOCUMENT_ENCODING_MODE,
)
DEFAULT_TERM_OUTPUT_MODE: str = "flat_terms"
SOURCE_TOKEN_TERM_OUTPUT_MODE: str = "source_token_terms"
VALID_TERM_OUTPUT_MODES: tuple[str, ...] = (
    DEFAULT_TERM_OUTPUT_MODE,
    SOURCE_TOKEN_TERM_OUTPUT_MODE,
)

FlatTermWeights = dict[str, float]
SourceTokenTermWeights = dict[str, dict[str, float]]
PatentTermPayload = FlatTermWeights | SourceTokenTermWeights


@dataclass(frozen=True, slots=True)
class PatentDocument:
    doc_id: str
    title: str
    abstract: str
    claims: str


@dataclass(frozen=True, slots=True)
class EncodedWindow:
    doc_id: str
    window_index: int
    input_ids: list[int]
    attention_mask: list[int]


@dataclass(frozen=True, slots=True)
class PatentWindowTensorBatch:
    doc_ids: list[str]
    window_doc_indices: torch.Tensor
    window_indices: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(slots=True)
class AggregatedTermProvenance:
    vector: torch.Tensor
    winning_window_indices: torch.Tensor
    winning_token_positions: torch.Tensor
    winning_source_token_ids: torch.Tensor


def normalize_whitespace(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def normalize_patent_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_whitespace(value)
    if isinstance(value, dict):
        preferred_fields: tuple[str, ...] = ("body", "text", "value")
        for field_name in preferred_fields:
            field_value: Any | None = value.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                return normalize_whitespace(field_value)
        return normalize_whitespace(json.dumps(value, ensure_ascii=False))
    if isinstance(value, list):
        parts: list[str] = []
        item: Any
        for item in value:
            normalized: str = normalize_patent_text(item)
            if normalized:
                parts.append(normalized)
        return " ".join(parts).strip()
    return normalize_whitespace(str(value))


def sparsify_dense_vector(
    vector: np.ndarray,
    *,
    exclude_output_ids: Sequence[int],
    min_weight: float,
    top_k: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if vector.ndim != 1:
        raise ValueError("Expected a 1D dense vector.")
    masked_vector: np.ndarray = np.asarray(vector, dtype=np.float32).copy()
    if exclude_output_ids:
        exclude_array: np.ndarray = np.asarray(exclude_output_ids, dtype=np.int64)
        valid_mask: np.ndarray = (exclude_array >= 0) & (exclude_array < masked_vector.shape[0])
        if bool(valid_mask.any()):
            masked_vector[exclude_array[valid_mask]] = 0.0
    threshold: float = float(min_weight)
    active_indices: np.ndarray = np.flatnonzero(masked_vector > threshold)
    if int(active_indices.size) == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )
    active_values: np.ndarray = masked_vector[active_indices]
    if top_k is not None and int(active_indices.size) > int(top_k):
        top_order: np.ndarray = np.argsort(active_values)[-int(top_k) :]
        active_indices = active_indices[top_order]
        active_values = active_values[top_order]
    return active_indices.astype(np.int32, copy=False), active_values.astype(
        np.float32, copy=False
    )


def sorted_sparse_vector_entries(
    vector: torch.Tensor,
    *,
    exclude_output_ids: Sequence[int],
    min_weight: float,
    top_k: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    vector_np: np.ndarray = vector.detach().cpu().float().numpy()
    indices, values = sparsify_dense_vector(
        vector_np,
        exclude_output_ids=[int(output_id) for output_id in exclude_output_ids],
        min_weight=float(min_weight),
        top_k=top_k,
    )
    if int(indices.size) == 0:
        return indices, values
    order: np.ndarray = np.argsort(values)[::-1]
    return indices[order], values[order]


def build_source_token_key(
    *,
    window_index: int,
    token_position: int,
    source_token_id: int,
    source_token_text: str,
) -> str:
    normalized_text: str = str(source_token_text) if str(source_token_text) else "<empty>"
    return (
        f"w{int(window_index)}:"
        f"p{int(token_position)}:"
        f"id{int(source_token_id)}:{normalized_text}"
    )


def encode_docs_with_max_pool_provenance(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    doc_pooling: str = str(getattr(model, "doc_pooling", "")).strip().lower()
    if doc_pooling and doc_pooling != "max":
        raise ValueError(
            "source_token_terms only supports SPLADE doc_pooling='max'. "
            f"Got: {doc_pooling!r}"
        )
    encoder: Any = getattr(model, "encoder", None)
    if encoder is None:
        raise ValueError("Expected model.encoder for provenance-aware SPLADE export.")
    logits: torch.Tensor = encoder.encode_raw_logits(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    token_scores: torch.Tensor = encoder.activation(logits)
    pooled: torch.Tensor = encoder._pool_sparse(
        token_scores,
        pooling_mask,
        model._doc_pooling_mode,
    )
    reps: torch.Tensor = model.postprocess_doc_embeddings(pooled)
    mask: torch.Tensor = pooling_mask.unsqueeze(-1).to(dtype=torch.bool)
    neg_inf: torch.Tensor = encoder._neg_inf.to(
        dtype=token_scores.dtype,
        device=token_scores.device,
    )
    masked_scores: torch.Tensor = token_scores.masked_fill(~mask, neg_inf)
    _max_values: torch.Tensor
    argmax_positions: torch.Tensor
    _max_values, argmax_positions = masked_scores.max(dim=1)
    source_token_ids: torch.Tensor = input_ids.gather(1, argmax_positions)
    return reps, argmax_positions, source_token_ids


def initialize_aggregated_term_provenance(
    *,
    vector: torch.Tensor,
    window_index: int,
    token_positions: torch.Tensor,
    source_token_ids: torch.Tensor,
) -> AggregatedTermProvenance:
    return AggregatedTermProvenance(
        vector=vector.clone(),
        winning_window_indices=torch.full(
            vector.shape,
            int(window_index),
            dtype=torch.int32,
        ),
        winning_token_positions=token_positions.detach().cpu().to(dtype=torch.int32),
        winning_source_token_ids=source_token_ids.detach().cpu().to(dtype=torch.int32),
    )


def update_aggregated_term_provenance(
    aggregated: AggregatedTermProvenance | None,
    *,
    vector: torch.Tensor,
    window_index: int,
    token_positions: torch.Tensor,
    source_token_ids: torch.Tensor,
) -> AggregatedTermProvenance:
    if aggregated is None:
        return initialize_aggregated_term_provenance(
            vector=vector,
            window_index=window_index,
            token_positions=token_positions,
            source_token_ids=source_token_ids,
        )
    better_mask: torch.Tensor = vector > aggregated.vector
    if not bool(better_mask.any()):
        return aggregated
    current_window_indices: torch.Tensor = torch.full(
        aggregated.winning_window_indices.shape,
        int(window_index),
        dtype=torch.int32,
    )
    token_positions_cpu: torch.Tensor = token_positions.detach().cpu().to(dtype=torch.int32)
    source_token_ids_cpu: torch.Tensor = source_token_ids.detach().cpu().to(dtype=torch.int32)
    aggregated.vector = torch.where(better_mask, vector, aggregated.vector)
    aggregated.winning_window_indices = torch.where(
        better_mask,
        current_window_indices,
        aggregated.winning_window_indices,
    )
    aggregated.winning_token_positions = torch.where(
        better_mask,
        token_positions_cpu,
        aggregated.winning_token_positions,
    )
    aggregated.winning_source_token_ids = torch.where(
        better_mask,
        source_token_ids_cpu,
        aggregated.winning_source_token_ids,
    )
    return aggregated


def _iter_label_ids(row: dict[str, Any]) -> Iterable[str]:
    labels_field: Any | None = row.get("labels")
    if isinstance(labels_field, list):
        label_item: Any
        for label_item in labels_field:
            if isinstance(label_item, dict):
                label_id: Any | None = label_item.get("label_id")
                if label_id is not None:
                    yield str(label_id).strip()
            elif label_item is not None:
                yield str(label_item).strip()
    legacy_label_ids: Any | None = row.get("label_id")
    if isinstance(legacy_label_ids, list):
        label_id_value: Any
        for label_id_value in legacy_label_ids:
            if label_id_value is not None:
                yield str(label_id_value).strip()
    elif legacy_label_ids is not None:
        yield str(legacy_label_ids).strip()


def collect_patent_ids(dataset_paths: Sequence[str | Path]) -> list[str]:
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    dataset_path_value: str | Path
    for dataset_path_value in dataset_paths:
        dataset_path = Path(dataset_path_value)
        rows: Any = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list at {dataset_path}.")
        row: Any
        for row in rows:
            if not isinstance(row, dict):
                continue
            candidate_ids: list[str] = []
            question_id: Any | None = row.get("question_id")
            if question_id is not None:
                candidate_ids.append(str(question_id).strip())
            candidate_ids.extend(_iter_label_ids(row))
            candidate_id: str
            for candidate_id in candidate_ids:
                if not candidate_id or candidate_id in seen_ids:
                    continue
                seen_ids.add(candidate_id)
                ordered_ids.append(candidate_id)
    return ordered_ids


def batched_values[T](values: Sequence[T], batch_size: int) -> Iterable[list[T]]:
    resolved_batch_size: int = max(1, int(batch_size))
    start: int
    for start in range(0, len(values), resolved_batch_size):
        yield list(values[start : start + resolved_batch_size])


def select_contiguous_shard[T](
    values: Sequence[T],
    *,
    shard_index: int,
    shard_count: int,
) -> list[T]:
    resolved_shard_count: int = max(1, int(shard_count))
    resolved_shard_index: int = int(shard_index)
    if resolved_shard_index < 0 or resolved_shard_index >= resolved_shard_count:
        raise ValueError(
            f"shard_index={resolved_shard_index} must be in [0, {resolved_shard_count})."
        )
    total_values: int = len(values)
    start_idx: int = (total_values * resolved_shard_index) // resolved_shard_count
    end_idx: int = (total_values * (resolved_shard_index + 1)) // resolved_shard_count
    return list(values[start_idx:end_idx])


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body: str = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenSearch request failed with status={exc.code}: {body[:1000]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenSearch request failed: {exc}") from exc


class OpenSearchPatentSource:
    def __init__(
        self,
        *,
        base_url: str,
        index_name: str,
        timeout_seconds: int,
        source_fields: Sequence[str] | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.index_name = str(index_name)
        self.timeout_seconds = int(timeout_seconds)
        self.source_fields = list(source_fields or ["doc_id", "US_title", "US_abstract", "US_claims"])

    def fetch_documents(
        self,
        doc_ids: Sequence[str],
        *,
        batch_size: int,
        show_progress: bool = True,
    ) -> dict[str, PatentDocument]:
        resolved_batch_size: int = max(1, int(batch_size))
        results: dict[str, PatentDocument] = {}
        total_batches: int = max(1, math.ceil(len(doc_ids) / resolved_batch_size))
        iterator: Iterable[int] = range(0, len(doc_ids), resolved_batch_size)
        progress: Iterable[int]
        if show_progress:
            progress = tqdm(
                iterator,
                total=total_batches,
                desc="Fetching patent documents",
                unit="batch",
            )
        else:
            progress = iterator
        start: int
        for start in progress:
            batch_doc_ids: list[str] = list(doc_ids[start : start + resolved_batch_size])
            docs_payload: list[dict[str, Any]] = [
                {"_id": doc_id, "_source": self.source_fields} for doc_id in batch_doc_ids
            ]
            response = _http_post_json(
                f"{self.base_url}/{self.index_name}/_mget",
                {"docs": docs_payload},
                timeout_seconds=self.timeout_seconds,
            )
            doc_payload: Any
            for doc_payload in response.get("docs", []):
                if not isinstance(doc_payload, dict) or not bool(doc_payload.get("found")):
                    continue
                source: Any | None = doc_payload.get("_source")
                if not isinstance(source, dict):
                    continue
                doc_id: str = str(source.get("doc_id", doc_payload.get("_id", ""))).strip()
                if not doc_id:
                    continue
                results[doc_id] = PatentDocument(
                    doc_id=doc_id,
                    title=normalize_patent_text(source.get("US_title")),
                    abstract=normalize_patent_text(source.get("US_abstract")),
                    claims=normalize_patent_text(source.get("US_claims")),
                )
        return results


def resolve_patent_corpus_paths(corpus_path: str | Path) -> list[Path]:
    raw_corpus_path: str = str(corpus_path).strip()
    if not raw_corpus_path:
        raise ValueError("Patent corpus path must be a non-empty string.")
    candidate_paths: list[Path]
    if any(token in raw_corpus_path for token in ("*", "?", "[")):
        candidate_paths = [Path(path_str) for path_str in sorted(glob.glob(raw_corpus_path))]
    else:
        candidate = Path(raw_corpus_path).expanduser()
        if candidate.is_dir():
            candidate_paths = sorted(candidate.glob("patent_us_docs_slice*.parquet"))
        elif candidate.is_file() and candidate.suffix == ".parquet":
            candidate_paths = [candidate]
        elif candidate.is_file() and candidate.suffix == ".json":
            manifest: Any = json.loads(candidate.read_text(encoding="utf-8"))
            shard_paths_raw: Any = manifest.get("parquet", {}).get("shard_paths", [])
            shard_paths: list[Path] = [Path(path_str) for path_str in shard_paths_raw]
            if shard_paths and all(path.exists() for path in shard_paths):
                candidate_paths = shard_paths
            else:
                candidate_paths = [
                    candidate.parent / shard_path.name
                    for shard_path in shard_paths
                    if (candidate.parent / shard_path.name).exists()
                ]
        else:
            raise ValueError(f"Unsupported patent corpus path: {candidate}")
    resolved_paths: list[Path] = [path.resolve() for path in candidate_paths if path.is_file()]
    if not resolved_paths:
        raise ValueError(
            "No patent parquet shards found. "
            f"Resolved corpus path input: {raw_corpus_path}"
        )
    return resolved_paths


class LocalParquetPatentSource:
    def __init__(
        self,
        *,
        corpus_path: str | Path,
    ) -> None:
        self.parquet_paths: list[Path] = resolve_patent_corpus_paths(corpus_path)

    def fetch_documents(
        self,
        doc_ids: Sequence[str],
        *,
        batch_size: int,
        show_progress: bool = True,
    ) -> dict[str, PatentDocument]:
        try:
            import pyarrow.dataset as ds
        except Exception as exc:
            raise RuntimeError(
                "pyarrow is required to scan the local Hugging Face patent parquet corpus."
            ) from exc
        requested_doc_ids: list[str] = []
        seen_ids: set[str] = set()
        doc_id: str
        for doc_id in doc_ids:
            normalized_doc_id: str = str(doc_id).strip()
            if not normalized_doc_id or normalized_doc_id in seen_ids:
                continue
            seen_ids.add(normalized_doc_id)
            requested_doc_ids.append(normalized_doc_id)
        if not requested_doc_ids:
            return {}
        dataset = ds.dataset([str(path) for path in self.parquet_paths], format="parquet")
        scanner = dataset.scanner(
            columns=["doc_id", "title", "abstract", "claims"],
            filter=ds.field("doc_id").isin(requested_doc_ids),
            batch_size=max(1024, int(batch_size)),
            use_threads=True,
        )
        results: dict[str, PatentDocument] = {}
        record_batches: Iterable[Any] = scanner.to_batches()
        if show_progress:
            record_batches = tqdm(
                record_batches,
                desc="Scanning patent parquet corpus",
                unit="batch",
            )
        record_batch: Any
        for record_batch in record_batches:
            columns: dict[str, list[Any]] = record_batch.to_pydict()
            doc_id_values: list[Any] = list(columns.get("doc_id", []))
            title_values: list[Any] = list(columns.get("title", []))
            abstract_values: list[Any] = list(columns.get("abstract", []))
            claims_values: list[Any] = list(columns.get("claims", []))
            row_doc_id: Any
            row_title: Any
            row_abstract: Any
            row_claims: Any
            for row_doc_id, row_title, row_abstract, row_claims in zip(
                doc_id_values,
                title_values,
                abstract_values,
                claims_values,
                strict=True,
            ):
                resolved_doc_id: str = str(row_doc_id).strip()
                if not resolved_doc_id:
                    continue
                results[resolved_doc_id] = PatentDocument(
                    doc_id=resolved_doc_id,
                    title=normalize_patent_text(row_title),
                    abstract=normalize_patent_text(row_abstract),
                    claims=normalize_patent_text(row_claims),
                )
        return results


def load_runtime_model_cfg(
    *,
    model_name: str,
    trust_remote_code: bool = False,
    dtype_name: str = "float32",
) -> DictConfig:
    repo_root: Path = Path(__file__).resolve().parents[2]
    base_cfg: DictConfig = OmegaConf.load(repo_root / "config" / "model" / "_base.yaml")
    splade_cfg: DictConfig = OmegaConf.load(
        repo_root / "config" / "model" / "splade_v3_naver.yaml"
    )
    model_cfg: DictConfig = OmegaConf.merge(base_cfg, splade_cfg)
    model_cfg.huggingface_name = str(model_name)
    model_cfg.trust_remote_code = bool(trust_remote_code)
    model_cfg.dtype = str(dtype_name)
    return OmegaConf.create({"model": model_cfg})


def build_runtime_model_and_tokenizer(
    *,
    model_name: str,
    checkpoint_path: str | None,
    use_cpu: bool,
    trust_remote_code: bool = False,
    dtype_name: str = "float32",
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase, DictConfig]:
    cfg: DictConfig = load_runtime_model_cfg(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        dtype_name=dtype_name,
    )
    cfg = apply_checkpoint_model_config(
        cfg,
        checkpoint_path=checkpoint_path,
        logger=logger,
    )
    model = build_splade_model_with_checkpoint(
        cfg=cfg,
        use_cpu=bool(use_cpu),
        checkpoint_path=checkpoint_path,
        logger=logger,
    )
    tokenizer: PreTrainedTokenizerBase = build_tokenizer(
        str(cfg.model.huggingface_name),
        use_fast_tokenizer=bool(cfg.model.use_fast_tokenizer),
        trust_remote_code=bool(cfg.model.trust_remote_code),
        require_fast_tokenizer=bool(cfg.model.require_fast_tokenizer),
        local_files_only=cfg.model.get("local_files_only"),
        revision=cfg.model.get("model_revision"),
    )
    return model, tokenizer, cfg


def _prepare_single_sequence(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int],
) -> tuple[list[int], list[int]]:
    prefix_ids: list[int] = []
    suffix_ids: list[int] = []
    cls_token_id: int | None = getattr(tokenizer, "cls_token_id", None)
    sep_token_id: int | None = getattr(tokenizer, "sep_token_id", None)
    bos_token_id: int | None = getattr(tokenizer, "bos_token_id", None)
    eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None)
    if cls_token_id is not None:
        prefix_ids.append(int(cls_token_id))
    elif bos_token_id is not None:
        prefix_ids.append(int(bos_token_id))
    if sep_token_id is not None:
        suffix_ids.append(int(sep_token_id))
    elif eos_token_id is not None:
        suffix_ids.append(int(eos_token_id))
    input_ids: list[int] = [*prefix_ids, *token_ids, *suffix_ids]
    attention_mask: list[int] = [1] * len(input_ids)
    return input_ids, attention_mask


def _prefix_text(title: str, field_label: str) -> str:
    return f"Title: {title}\n{field_label}: "


def resolve_document_encoding_mode(document_encoding_mode: str) -> str:
    normalized_mode: str = str(document_encoding_mode).strip().lower()
    if normalized_mode in VALID_DOCUMENT_ENCODING_MODES:
        return normalized_mode
    valid_modes: str = ", ".join(VALID_DOCUMENT_ENCODING_MODES)
    raise ValueError(
        "document_encoding_mode must be one of: "
        f"{valid_modes}. Got: {document_encoding_mode!r}"
    )


def resolve_term_output_mode(term_output_mode: str) -> str:
    normalized_mode: str = str(term_output_mode).strip().lower()
    if normalized_mode in VALID_TERM_OUTPUT_MODES:
        return normalized_mode
    valid_modes: str = ", ".join(VALID_TERM_OUTPUT_MODES)
    raise ValueError(
        "term_output_mode must be one of: "
        f"{valid_modes}. Got: {term_output_mode!r}"
    )


def _encode_text_ids(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    try:
        encoded: Any = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
        input_ids: Any = encoded["input_ids"]
        if isinstance(input_ids, list):
            return [int(token_id) for token_id in input_ids]
    except Exception:
        pass
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


def build_truncated_text_window(
    tokenizer: PreTrainedTokenizerBase,
    *,
    text: str,
    max_length: int,
) -> tuple[list[int], list[int]] | None:
    normalized_text: str = normalize_patent_text(text)
    if not normalized_text:
        return None
    token_ids: list[int] = _encode_text_ids(tokenizer, normalized_text)
    special_token_count: int = int(tokenizer.num_special_tokens_to_add(pair=False))
    max_content_tokens: int = max(1, int(max_length) - special_token_count)
    return _prepare_single_sequence(tokenizer, token_ids[:max_content_tokens])


def build_prefixed_windows(
    tokenizer: PreTrainedTokenizerBase,
    *,
    title: str,
    field_label: str,
    field_text: str,
    max_length: int,
    overlap_tokens: int = 0,
) -> list[tuple[list[int], list[int]]]:
    normalized_title: str = normalize_patent_text(title)
    normalized_field_text: str = normalize_patent_text(field_text)
    if not normalized_field_text:
        return []
    prefix_ids: list[int] = _encode_text_ids(
        tokenizer,
        _prefix_text(normalized_title, field_label),
    )
    field_ids: list[int] = _encode_text_ids(tokenizer, normalized_field_text)
    special_token_count: int = int(tokenizer.num_special_tokens_to_add(pair=False))
    max_content_tokens: int = int(max_length) - special_token_count
    if max_content_tokens <= 0:
        raise ValueError("max_length must allow at least one special-token-wrapped token.")
    if len(prefix_ids) >= max_content_tokens:
        truncated_combined_ids: list[int] = (
            prefix_ids + field_ids
        )[:max_content_tokens]
        input_ids, attention_mask = _prepare_single_sequence(
            tokenizer, truncated_combined_ids
        )
        return [(input_ids, attention_mask)]
    available_field_tokens: int = max_content_tokens - len(prefix_ids)
    if available_field_tokens <= 0:
        input_ids, attention_mask = _prepare_single_sequence(tokenizer, prefix_ids)
        return [(input_ids, attention_mask)]
    stride: int = max(1, available_field_tokens - max(0, int(overlap_tokens)))
    windows: list[tuple[list[int], list[int]]] = []
    start_idx: int
    for start_idx in range(0, len(field_ids), stride):
        chunk_ids: list[int] = field_ids[start_idx : start_idx + available_field_tokens]
        if not chunk_ids:
            continue
        combined_ids: list[int] = prefix_ids + chunk_ids
        input_ids, attention_mask = _prepare_single_sequence(tokenizer, combined_ids)
        windows.append((input_ids, attention_mask))
        if start_idx + available_field_tokens >= len(field_ids):
            break
    return windows


def build_title_only_window(
    tokenizer: PreTrainedTokenizerBase,
    *,
    title: str,
    max_length: int,
) -> tuple[list[int], list[int]] | None:
    normalized_title: str = normalize_patent_text(title)
    if not normalized_title:
        return None
    return build_truncated_text_window(
        tokenizer,
        text=f"Title: {normalized_title}",
        max_length=max_length,
    )


def build_combined_truncated_window(
    tokenizer: PreTrainedTokenizerBase,
    *,
    title: str,
    abstract: str,
    claims: str,
    max_length: int,
) -> tuple[list[int], list[int]] | None:
    normalized_title: str = normalize_patent_text(title)
    normalized_abstract: str = normalize_patent_text(abstract)
    normalized_claims: str = normalize_patent_text(claims)
    parts: list[str] = []
    if normalized_title:
        parts.append(f"Title: {normalized_title}")
    if normalized_abstract:
        parts.append(f"Abstract: {normalized_abstract}")
    if normalized_claims:
        parts.append(f"Claims: {normalized_claims}")
    return build_truncated_text_window(
        tokenizer,
        text="\n".join(parts),
        max_length=max_length,
    )


def build_document_windows(
    tokenizer: PreTrainedTokenizerBase,
    document: PatentDocument,
    *,
    max_length: int,
    claim_overlap_tokens: int = 0,
    document_encoding_mode: str = DEFAULT_DOCUMENT_ENCODING_MODE,
) -> list[EncodedWindow]:
    resolved_encoding_mode: str = resolve_document_encoding_mode(document_encoding_mode)
    if resolved_encoding_mode == COMBINED_TRUNCATE_DOCUMENT_ENCODING_MODE:
        combined_window = build_combined_truncated_window(
            tokenizer,
            title=document.title,
            abstract=document.abstract,
            claims=document.claims,
            max_length=max_length,
        )
        if combined_window is None:
            return []
        input_ids, attention_mask = combined_window
        return [
            EncodedWindow(
                doc_id=document.doc_id,
                window_index=0,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        ]
    windows: list[EncodedWindow] = []
    abstract_windows = build_prefixed_windows(
        tokenizer,
        title=document.title,
        field_label="Abstract",
        field_text=document.abstract,
        max_length=max_length,
        overlap_tokens=0,
    )
    claims_windows = build_prefixed_windows(
        tokenizer,
        title=document.title,
        field_label="Claims",
        field_text=document.claims,
        max_length=max_length,
        overlap_tokens=claim_overlap_tokens,
    )
    window_index: int = 0
    input_ids: list[int]
    attention_mask: list[int]
    for input_ids, attention_mask in abstract_windows + claims_windows:
        windows.append(
            EncodedWindow(
                doc_id=document.doc_id,
                window_index=window_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )
        window_index += 1
    if windows:
        return windows
    title_only = build_title_only_window(
        tokenizer,
        title=document.title,
        max_length=max_length,
    )
    if title_only is None:
        return []
    input_ids, attention_mask = title_only
    return [
        EncodedWindow(
            doc_id=document.doc_id,
            window_index=0,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    ]


class PatentDocumentDataset(Dataset[PatentDocument]):
    def __init__(self, documents: Sequence[PatentDocument]) -> None:
        self.documents: list[PatentDocument] = list(documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> PatentDocument:
        return self.documents[int(index)]


class PatentWindowBatchCollator:
    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        claim_overlap_tokens: int,
        document_encoding_mode: str,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_length: int = int(max_length)
        self.claim_overlap_tokens: int = int(claim_overlap_tokens)
        self.document_encoding_mode: str = resolve_document_encoding_mode(
            document_encoding_mode
        )

    def __call__(
        self,
        documents: Sequence[PatentDocument],
    ) -> PatentWindowTensorBatch:
        doc_ids: list[str] = [document.doc_id for document in documents]
        window_doc_indices: list[int] = []
        window_indices: list[int] = []
        flat_windows: list[dict[str, list[int]]] = []
        doc_idx: int
        document: PatentDocument
        for doc_idx, document in enumerate(documents):
            encoded_windows: list[EncodedWindow] = build_document_windows(
                self.tokenizer,
                document,
                max_length=self.max_length,
                claim_overlap_tokens=self.claim_overlap_tokens,
                document_encoding_mode=self.document_encoding_mode,
            )
            window: EncodedWindow
            for window in encoded_windows:
                window_doc_indices.append(int(doc_idx))
                window_indices.append(int(window.window_index))
                flat_windows.append(
                    {
                        "input_ids": list(window.input_ids),
                        "attention_mask": list(window.attention_mask),
                    }
                )
        if not flat_windows:
            empty_long: torch.Tensor = torch.empty((0,), dtype=torch.long)
            empty_matrix: torch.Tensor = torch.empty((0, 0), dtype=torch.long)
            return PatentWindowTensorBatch(
                doc_ids=doc_ids,
                window_doc_indices=empty_long,
                window_indices=empty_long.clone(),
                input_ids=empty_matrix,
                attention_mask=empty_matrix.clone(),
            )
        padded: dict[str, torch.Tensor] = self.tokenizer.pad(
            flat_windows,
            padding=True,
            return_tensors="pt",
        )
        return PatentWindowTensorBatch(
            doc_ids=doc_ids,
            window_doc_indices=torch.tensor(window_doc_indices, dtype=torch.long),
            window_indices=torch.tensor(window_indices, dtype=torch.long),
            input_ids=padded["input_ids"],
            attention_mask=padded["attention_mask"],
        )


def encode_document_windows(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    model_cfg: DictConfig,
    windows: Sequence[EncodedWindow],
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
) -> dict[str, torch.Tensor]:
    if not windows:
        return {}
    aggregated: dict[str, torch.Tensor] = {}
    model.eval()
    iterator: Iterable[int] = range(0, len(windows), max(1, int(batch_size)))
    progress: Iterable[int]
    if show_progress:
        progress = tqdm(
            iterator,
            total=max(1, math.ceil(len(windows) / max(1, int(batch_size)))),
            desc="Encoding SPLADE windows",
            unit="batch",
        )
    else:
        progress = iterator
    with torch.no_grad():
        start: int
        for start in progress:
            batch_windows: Sequence[EncodedWindow] = windows[start : start + int(batch_size)]
            padded = tokenizer.pad(
                [
                    {
                        "input_ids": window.input_ids,
                        "attention_mask": window.attention_mask,
                    }
                    for window in batch_windows
                ],
                padding=True,
                return_tensors="pt",
            )
            input_ids: torch.Tensor = padded["input_ids"].to(device)
            attention_mask: torch.Tensor = padded["attention_mask"].to(device)
            pooling_mask: torch.Tensor = build_doc_pooling_mask(
                attention_mask,
                model_cfg,
            ).to(device)
            reps: torch.Tensor = model.encode_docs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pooling_mask=pooling_mask,
            )
            window: EncodedWindow
            rep: torch.Tensor
            for window, rep in zip(batch_windows, reps, strict=True):
                rep_cpu: torch.Tensor = rep.detach().cpu().float()
                existing: torch.Tensor | None = aggregated.get(window.doc_id)
                if existing is None:
                    aggregated[window.doc_id] = rep_cpu
                else:
                    aggregated[window.doc_id] = torch.maximum(existing, rep_cpu)
    return aggregated


def merge_aggregated_vectors(
    destination: dict[str, torch.Tensor],
    updates: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    doc_id: str
    vector: torch.Tensor
    for doc_id, vector in updates.items():
        existing: torch.Tensor | None = destination.get(doc_id)
        if existing is None:
            destination[doc_id] = vector
        else:
            destination[doc_id] = torch.maximum(existing, vector)
    return destination


def dense_vector_to_term_weights(
    vector: torch.Tensor,
    *,
    output_space: OutputSpaceSpec,
    tokenizer: PreTrainedTokenizerBase,
    exclude_output_ids: Sequence[int],
    min_weight: float,
    top_k: int | None,
) -> FlatTermWeights:
    sorted_indices, sorted_values = sorted_sparse_vector_entries(
        vector,
        exclude_output_ids=exclude_output_ids,
        min_weight=min_weight,
        top_k=top_k,
    )
    if int(sorted_indices.size) == 0:
        return {}
    if not bool(output_space.output_token_aligned):
        raise ValueError("Expected token-aligned SPLADE output space.")
    tokens: list[str] = list(tokenizer.convert_ids_to_tokens(sorted_indices.tolist()))
    result: FlatTermWeights = {}
    token: str
    value: np.floating[Any]
    for token, value in zip(tokens, sorted_values, strict=True):
        if not token:
            continue
        result[str(token)] = float(value)
    return result


def dense_vector_to_source_token_term_weights(
    vector: torch.Tensor,
    *,
    output_space: OutputSpaceSpec,
    tokenizer: PreTrainedTokenizerBase,
    exclude_output_ids: Sequence[int],
    min_weight: float,
    top_k: int | None,
    winning_window_indices: torch.Tensor,
    winning_token_positions: torch.Tensor,
    winning_source_token_ids: torch.Tensor,
) -> SourceTokenTermWeights:
    sorted_indices, sorted_values = sorted_sparse_vector_entries(
        vector,
        exclude_output_ids=exclude_output_ids,
        min_weight=min_weight,
        top_k=top_k,
    )
    if int(sorted_indices.size) == 0:
        return {}
    if not bool(output_space.output_token_aligned):
        raise ValueError("Expected token-aligned SPLADE output space.")
    extracted_terms: list[str] = list(tokenizer.convert_ids_to_tokens(sorted_indices.tolist()))
    window_indices: np.ndarray = (
        winning_window_indices.detach().cpu().to(dtype=torch.int64).numpy()[sorted_indices]
    )
    token_positions: np.ndarray = (
        winning_token_positions.detach().cpu().to(dtype=torch.int64).numpy()[sorted_indices]
    )
    source_token_ids: np.ndarray = (
        winning_source_token_ids.detach().cpu().to(dtype=torch.int64).numpy()[sorted_indices]
    )
    source_tokens: list[str] = list(tokenizer.convert_ids_to_tokens(source_token_ids.tolist()))
    result: SourceTokenTermWeights = {}
    extracted_term: str
    source_token_text: str
    value: np.floating[Any]
    window_index_value: np.integer[Any]
    token_position_value: np.integer[Any]
    source_token_id_value: np.integer[Any]
    for (
        extracted_term,
        source_token_text,
        value,
        window_index_value,
        token_position_value,
        source_token_id_value,
    ) in zip(
        extracted_terms,
        source_tokens,
        sorted_values,
        window_indices,
        token_positions,
        source_token_ids,
        strict=True,
    ):
        if not extracted_term:
            continue
        source_key: str = build_source_token_key(
            window_index=int(window_index_value),
            token_position=int(token_position_value),
            source_token_id=int(source_token_id_value),
            source_token_text=str(source_token_text),
        )
        source_bucket: FlatTermWeights = result.setdefault(source_key, {})
        source_bucket[str(extracted_term)] = float(value)
    return result


def build_patent_source(
    *,
    patent_source: str,
    corpus_path: str | Path | None,
    opensearch_url: str,
    opensearch_index: str,
    timeout_seconds: int,
) -> Any:
    normalized_patent_source: str = str(patent_source).strip().lower()
    if normalized_patent_source in {"huggingface", "hf", "parquet"}:
        if corpus_path is None:
            raise ValueError("corpus_path is required when patent_source=huggingface.")
        return LocalParquetPatentSource(corpus_path=corpus_path)
    if normalized_patent_source == "opensearch":
        return OpenSearchPatentSource(
            base_url=opensearch_url,
            index_name=opensearch_index,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(
        "patent_source must be one of: huggingface, parquet, hf, opensearch."
    )


def iter_patent_term_weights(
    *,
    patent_ids: Sequence[str],
    documents_by_id: dict[str, PatentDocument],
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    model_cfg: DictConfig,
    output_space: OutputSpaceSpec,
    exclude_output_ids: Sequence[int],
    document_batch_size: int,
    encode_batch_size: int,
    max_length: int,
    claim_overlap_tokens: int,
    document_encoding_mode: str,
    term_output_mode: str,
    dataloader_num_workers: int,
    dataloader_prefetch_factor: int | None,
    device: torch.device,
    use_cpu: bool,
    min_weight: float,
    top_k: int | None,
    show_progress: bool,
) -> Iterable[tuple[str, PatentTermPayload]]:
    resolved_term_output_mode: str = resolve_term_output_mode(term_output_mode)
    ordered_documents: list[PatentDocument] = [
        documents_by_id[patent_id] for patent_id in patent_ids
    ]
    dataset = PatentDocumentDataset(ordered_documents)
    collator = PatentWindowBatchCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        claim_overlap_tokens=claim_overlap_tokens,
        document_encoding_mode=document_encoding_mode,
    )
    dataloader = build_inference_dataloader(
        dataset=dataset,
        batch_size=max(1, int(document_batch_size)),
        num_workers=max(0, int(dataloader_num_workers)),
        collate_fn=collator,
        use_cpu=bool(use_cpu),
        shuffle=False,
        drop_last=False,
        distributed_shuffle=False,
        prefetch_factor=dataloader_prefetch_factor,
    )
    progress: Iterable[PatentWindowTensorBatch]
    if show_progress:
        total_batches: int = max(
            1,
            math.ceil(len(ordered_documents) / max(1, int(document_batch_size))),
        )
        progress = tqdm(
            dataloader,
            total=total_batches,
            desc="Processing patent batches",
            unit="batch",
        )
    else:
        progress = dataloader
    model.eval()
    with torch.no_grad():
        batch: PatentWindowTensorBatch
        for batch in progress:
            aggregated_vectors: dict[int, torch.Tensor] = {}
            aggregated_provenance: dict[int, AggregatedTermProvenance] = {}
            if int(batch.input_ids.shape[0]) > 0:
                total_windows: int = int(batch.input_ids.shape[0])
                start: int
                for start in range(0, total_windows, max(1, int(encode_batch_size))):
                    end: int = min(total_windows, start + max(1, int(encode_batch_size)))
                    input_ids: torch.Tensor = batch.input_ids[start:end].to(
                        device,
                        non_blocking=(device.type == "cuda"),
                    )
                    attention_mask: torch.Tensor = batch.attention_mask[start:end].to(
                        device,
                        non_blocking=(device.type == "cuda"),
                    )
                    pooling_mask: torch.Tensor = build_doc_pooling_mask(
                        attention_mask,
                        model_cfg,
                    ).to(device, non_blocking=(device.type == "cuda"))
                    if resolved_term_output_mode == DEFAULT_TERM_OUTPUT_MODE:
                        reps: torch.Tensor = model.encode_docs(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pooling_mask=pooling_mask,
                        )
                        rep: torch.Tensor
                        doc_index: int
                        for doc_index, rep in zip(
                            batch.window_doc_indices[start:end].tolist(),
                            reps,
                            strict=True,
                        ):
                            rep_cpu: torch.Tensor = rep.detach().cpu().float()
                            existing: torch.Tensor | None = aggregated_vectors.get(int(doc_index))
                            if existing is None:
                                aggregated_vectors[int(doc_index)] = rep_cpu
                            else:
                                aggregated_vectors[int(doc_index)] = torch.maximum(
                                    existing,
                                    rep_cpu,
                                )
                        continue
                    reps, argmax_positions, source_token_ids = encode_docs_with_max_pool_provenance(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pooling_mask=pooling_mask,
                    )
                    rep: torch.Tensor
                    doc_index: int
                    window_index: int
                    token_positions: torch.Tensor
                    source_ids: torch.Tensor
                    for doc_index, window_index, rep, token_positions, source_ids in zip(
                        batch.window_doc_indices[start:end].tolist(),
                        batch.window_indices[start:end].tolist(),
                        reps,
                        argmax_positions,
                        source_token_ids,
                        strict=True,
                    ):
                        rep_cpu = rep.detach().cpu().float()
                        aggregated_provenance[int(doc_index)] = update_aggregated_term_provenance(
                            aggregated_provenance.get(int(doc_index)),
                            vector=rep_cpu,
                            window_index=int(window_index),
                            token_positions=token_positions,
                            source_token_ids=source_ids,
                        )
            local_doc_index: int
            patent_id: str
            for local_doc_index, patent_id in enumerate(batch.doc_ids):
                if resolved_term_output_mode == DEFAULT_TERM_OUTPUT_MODE:
                    vector: torch.Tensor | None = aggregated_vectors.get(int(local_doc_index))
                    if vector is None:
                        yield patent_id, {}
                        continue
                    yield patent_id, dense_vector_to_term_weights(
                        vector,
                        output_space=output_space,
                        tokenizer=tokenizer,
                        exclude_output_ids=exclude_output_ids,
                        min_weight=min_weight,
                        top_k=top_k,
                    )
                    continue
                provenance: AggregatedTermProvenance | None = aggregated_provenance.get(
                    int(local_doc_index)
                )
                if provenance is None:
                    yield patent_id, {}
                    continue
                yield patent_id, dense_vector_to_source_token_term_weights(
                    provenance.vector,
                    output_space=output_space,
                    tokenizer=tokenizer,
                    exclude_output_ids=exclude_output_ids,
                    min_weight=min_weight,
                    top_k=top_k,
                    winning_window_indices=provenance.winning_window_indices,
                    winning_token_positions=provenance.winning_token_positions,
                    winning_source_token_ids=provenance.winning_source_token_ids,
                )


def write_patent_term_weights(
    entries: Iterable[tuple[str, PatentTermPayload]],
    *,
    output_path: str | Path,
    output_format: str,
    collect_results: bool = False,
) -> dict[str, PatentTermPayload]:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    normalized_format: str = str(output_format).strip().lower()
    results: dict[str, PatentTermPayload] = {}
    with output_file.open("w", encoding="utf-8") as handle:
        if normalized_format == "jsonl":
            patent_id: str
            term_weights: PatentTermPayload
            for patent_id, term_weights in entries:
                json.dump(
                    {"doc_id": patent_id, "term_weights": term_weights},
                    handle,
                    ensure_ascii=False,
                )
                handle.write("\n")
                if collect_results:
                    results[patent_id] = term_weights
            return results
        if normalized_format != "json":
            raise ValueError("output_format must be one of: json, jsonl.")
        handle.write("{")
        first_entry: bool = True
        patent_id = ""
        term_weights: PatentTermPayload = {}
        for patent_id, term_weights in entries:
            if first_entry:
                handle.write("\n")
                first_entry = False
            else:
                handle.write(",\n")
            json.dump(patent_id, handle, ensure_ascii=False)
            handle.write(": ")
            json.dump(term_weights, handle, ensure_ascii=False)
            if collect_results:
                results[patent_id] = term_weights
        if not first_entry:
            handle.write("\n")
        handle.write("}\n")
    return results


def merge_patent_term_shards(
    *,
    shard_paths: Sequence[str | Path],
    output_path: str | Path,
    shard_format: str = "jsonl",
) -> None:
    normalized_format: str = str(shard_format).strip().lower()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as output_handle:
        output_handle.write("{")
        first_entry: bool = True
        shard_path_value: str | Path
        for shard_path_value in shard_paths:
            shard_path = Path(shard_path_value)
            if normalized_format == "jsonl":
                with shard_path.open("r", encoding="utf-8") as input_handle:
                    line: str
                    for line in input_handle:
                        stripped: str = line.strip()
                        if not stripped:
                            continue
                        payload: dict[str, Any] = json.loads(stripped)
                        doc_id: str = str(payload["doc_id"])
                        term_weights: PatentTermPayload = dict(payload["term_weights"])
                        if first_entry:
                            output_handle.write("\n")
                            first_entry = False
                        else:
                            output_handle.write(",\n")
                        json.dump(doc_id, output_handle, ensure_ascii=False)
                        output_handle.write(": ")
                        json.dump(term_weights, output_handle, ensure_ascii=False)
                continue
            if normalized_format == "json":
                payload = json.loads(shard_path.read_text(encoding="utf-8"))
                doc_id_value: str
                term_weight_map: PatentTermPayload
                for doc_id_value, term_weight_map in payload.items():
                    if first_entry:
                        output_handle.write("\n")
                        first_entry = False
                    else:
                        output_handle.write(",\n")
                    json.dump(str(doc_id_value), output_handle, ensure_ascii=False)
                    output_handle.write(": ")
                    json.dump(term_weight_map, output_handle, ensure_ascii=False)
                continue
            raise ValueError("shard_format must be one of: json, jsonl.")
        if not first_entry:
            output_handle.write("\n")
        output_handle.write("}\n")


def export_patent_splade_terms(
    *,
    dataset_paths: Sequence[str | Path],
    output_path: str | Path,
    model_name: str,
    checkpoint_path: str | None,
    patent_source: str,
    corpus_path: str | Path | None,
    opensearch_url: str,
    opensearch_index: str,
    opensearch_batch_size: int,
    document_batch_size: int,
    encode_batch_size: int,
    dataloader_num_workers: int,
    dataloader_prefetch_factor: int | None,
    max_length: int,
    claim_overlap_tokens: int,
    document_encoding_mode: str = DEFAULT_DOCUMENT_ENCODING_MODE,
    term_output_mode: str = DEFAULT_TERM_OUTPUT_MODE,
    min_weight: float,
    top_k: int | None,
    shard_index: int = 0,
    shard_count: int = 1,
    output_format: str = "json",
    collect_results: bool = False,
    use_cpu: bool,
    trust_remote_code: bool = False,
    timeout_seconds: int = 120,
    show_progress: bool = True,
) -> dict[str, PatentTermPayload]:
    patent_ids: list[str] = select_contiguous_shard(
        collect_patent_ids(dataset_paths),
        shard_index=shard_index,
        shard_count=shard_count,
    )
    if not patent_ids:
        return write_patent_term_weights(
            [],
            output_path=output_path,
            output_format=output_format,
            collect_results=collect_results,
        )
    model, tokenizer, cfg = build_runtime_model_and_tokenizer(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        use_cpu=use_cpu,
        trust_remote_code=trust_remote_code,
    )
    device: torch.device
    if bool(use_cpu) or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    model.to(device)
    source_exclude_token_ids: list[int] = [
        int(token_id) for token_id in tokenizer.all_special_ids
    ]
    exclude_output_ids: list[int] = resolve_model_output_exclude_ids(
        model,
        source_exclude_token_ids,
    )
    output_space: OutputSpaceSpec = model.encoder.output_space
    source = build_patent_source(
        patent_source=patent_source,
        corpus_path=corpus_path,
        opensearch_url=opensearch_url,
        opensearch_index=opensearch_index,
        timeout_seconds=timeout_seconds,
    )
    documents_by_id: dict[str, PatentDocument] = source.fetch_documents(
        patent_ids,
        batch_size=opensearch_batch_size,
        show_progress=show_progress,
    )
    missing_ids: list[str] = [doc_id for doc_id in patent_ids if doc_id not in documents_by_id]
    if missing_ids:
        raise RuntimeError(
            "Missing patent documents for "
            f"{len(missing_ids)} ids. Sample: {missing_ids[:20]}"
        )
    entries: Iterable[tuple[str, PatentTermPayload]] = iter_patent_term_weights(
        patent_ids=patent_ids,
        documents_by_id=documents_by_id,
        model=model,
        tokenizer=tokenizer,
        model_cfg=cfg.model,
        output_space=output_space,
        exclude_output_ids=exclude_output_ids,
        document_batch_size=document_batch_size,
        encode_batch_size=encode_batch_size,
        max_length=max_length,
        claim_overlap_tokens=claim_overlap_tokens,
        document_encoding_mode=document_encoding_mode,
        term_output_mode=term_output_mode,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        device=device,
        use_cpu=use_cpu,
        min_weight=min_weight,
        top_k=top_k,
        show_progress=show_progress,
    )
    return write_patent_term_weights(
        entries,
        output_path=output_path,
        output_format=output_format,
        collect_results=collect_results,
    )


__all__ = [
    "EncodedWindow",
    "LocalParquetPatentSource",
    "OpenSearchPatentSource",
    "PatentDocument",
    "PatentDocumentDataset",
    "PatentWindowBatchCollator",
    "PatentWindowTensorBatch",
    "build_document_windows",
    "build_combined_truncated_window",
    "build_patent_source",
    "build_prefixed_windows",
    "build_truncated_text_window",
    "build_title_only_window",
    "collect_patent_ids",
    "dense_vector_to_term_weights",
    "encode_document_windows",
    "export_patent_splade_terms",
    "iter_patent_term_weights",
    "load_runtime_model_cfg",
    "merge_patent_term_shards",
    "merge_aggregated_vectors",
    "_encode_text_ids",
    "_iter_label_ids",
    "normalize_patent_text",
    "normalize_whitespace",
    "resolve_document_encoding_mode",
    "resolve_patent_corpus_paths",
    "select_contiguous_shard",
    "sparsify_dense_vector",
    "write_patent_term_weights",
]
