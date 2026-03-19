import json
import logging
from pathlib import Path
from typing import Any, Iterable

import hydra
import numpy as np
import torch
from datasets import Dataset, IterableDataset, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from config.path import ABS_CONFIG_DIR
from src.data.lens_formatting import (
    build_doc_pooling_mask,
    build_query_pooling_mask,
    format_query_text,
    validate_lens_tokenizer,
)
from src.model.pl_module.utils import build_splade_model_with_checkpoint
from src.search.sparsify import sparsify_query_vector
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.model_utils import apply_checkpoint_model_config
from src.utils.output_space import OutputSpaceSpec, resolve_model_output_exclude_ids
from src.utils.script_setup import (
    configure_script_environment,
    initialize_run,
    normalize_optional_str,
)
from src.utils.transformers import build_tokenizer

logger: logging.Logger = get_logger("script.analyze_vocab_terms", __file__)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=True,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


def _load_triplets_dataset(cfg: DictConfig) -> Dataset:
    analysis_cfg: DictConfig = cfg.analysis
    hf_name: str = str(analysis_cfg.hf_name)
    hf_subset: str | None = normalize_optional_str(analysis_cfg.hf_subset)
    split: str = str(analysis_cfg.split)
    cache_dir: str | None = normalize_optional_str(analysis_cfg.hf_cache_dir)
    data_files: Any | None = analysis_cfg.hf_data_files
    if hf_subset is not None:
        return load_dataset(
            hf_name,
            name=hf_subset,
            split=split,
            cache_dir=cache_dir,
            data_files=data_files,
        )
    return load_dataset(
        hf_name,
        split=split,
        cache_dir=cache_dir,
        data_files=data_files,
    )


def _collect_sample_rows(
    dataset: Dataset,
    *,
    sample_count: int,
    seed: int,
    shuffle: bool,
) -> list[dict[str, Any]]:
    if sample_count <= 0:
        raise ValueError("analysis.sample_count must be positive.")
    samples: list[dict[str, Any]] = []
    if shuffle:
        dataset_length: int = int(len(dataset))
        if dataset_length <= 0:
            return samples
        replace: bool = sample_count > dataset_length
        rng = np.random.default_rng(seed)
        indices = rng.choice(dataset_length, size=sample_count, replace=replace)
        for idx in indices:
            row = dict(dataset[int(idx)])
            samples.append(row)
            if len(samples) >= sample_count:
                break
        return samples
    for row in dataset:
        samples.append(dict(row))
        if len(samples) >= sample_count:
            break
    return samples


def _extract_text_samples(
    rows: list[dict[str, Any]],
    *,
    query_text_column: str,
    positive_text_column: str,
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for row in rows:
        query_value: Any | None = row.get(query_text_column)
        positive_value: Any | None = row.get(positive_text_column)
        if query_value is None or positive_value is None:
            continue
        query_text: str = str(query_value)
        positive_text: str = str(positive_value)
        if not query_text.strip() or not positive_text.strip():
            continue
        samples.append({"query_text": query_text, "positive_text": positive_text})
    return samples


def _load_text_dataset(
    *,
    hf_name: str,
    hf_subset: str | None,
    split: str,
    cache_dir: str | None,
    data_files: Any | None,
    streaming: bool,
) -> Dataset | IterableDataset:
    if hf_subset is not None:
        return load_dataset(
            hf_name,
            name=hf_subset,
            split=split,
            cache_dir=cache_dir,
            data_files=data_files,
            streaming=streaming,
        )
    return load_dataset(
        hf_name,
        split=split,
        cache_dir=cache_dir,
        data_files=data_files,
        streaming=streaming,
    )


def _lookup_texts(
    dataset: Dataset | IterableDataset,
    *,
    id_column: str,
    text_column: str,
    wanted_ids: set[str],
) -> dict[str, str]:
    remaining: set[str] = set(wanted_ids)
    found: dict[str, str] = {}
    for row in dataset:
        row_id: Any | None = row.get(id_column)
        if row_id is None:
            continue
        row_id_str: str = str(row_id)
        if row_id_str not in remaining:
            continue
        text_value: Any | None = row.get(text_column)
        if text_value is None:
            continue
        text: str = str(text_value)
        if not text.strip():
            continue
        found[row_id_str] = text
        remaining.remove(row_id_str)
        if not remaining:
            break
    return found


def _lookup_texts_by_index(
    dataset: Dataset,
    *,
    id_column: str,
    text_column: str,
    wanted_ids: set[str],
) -> tuple[dict[str, str], set[str]]:
    found: dict[str, str] = {}
    missing: set[str] = set()
    dataset_length: int = int(len(dataset))
    for id_value in wanted_ids:
        try:
            idx = int(id_value)
        except ValueError:
            missing.add(id_value)
            continue
        if idx < 0 or idx >= dataset_length:
            missing.add(id_value)
            continue
        row = dataset[idx]
        row_id: Any | None = row.get(id_column)
        if row_id is None or str(row_id) != id_value:
            missing.add(id_value)
            continue
        text_value: Any | None = row.get(text_column)
        if text_value is None:
            missing.add(id_value)
            continue
        text: str = str(text_value)
        if not text.strip():
            missing.add(id_value)
            continue
        found[id_value] = text
    return found, missing


def _batch_texts(texts: list[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size <= 0:
        raise ValueError("analysis.batch_size must be positive.")
    for start in range(0, len(texts), batch_size):
        yield texts[start : start + batch_size]


def _encode_texts(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    model_cfg: DictConfig,
    max_length: int,
    batch_size: int,
    device: torch.device,
    encode_fn: Any,
    is_query: bool,
) -> torch.Tensor:
    if not texts:
        vocab_size: int = int(model.encoder.vocab_size)
        empty: torch.Tensor = torch.empty(
            (0, vocab_size), dtype=model.encoder.dtype, device=device
        )
        return empty
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in _batch_texts(texts, batch_size):
            batch_texts: list[str]
            if is_query:
                batch_texts = [format_query_text(text, model_cfg) for text in batch]
            else:
                batch_texts = list(batch)
            tokens: dict[str, torch.Tensor] = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            input_ids: torch.Tensor = tokens["input_ids"].to(device)
            attention_mask: torch.Tensor = tokens["attention_mask"].to(device)
            if is_query:
                pooling_mask: torch.Tensor = build_query_pooling_mask(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=tokenizer,
                    model_cfg=model_cfg,
                ).to(device)
            else:
                pooling_mask = build_doc_pooling_mask(
                    attention_mask=attention_mask,
                    model_cfg=model_cfg,
                ).to(device)
            batch_reps: torch.Tensor = encode_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pooling_mask=pooling_mask,
            )
            outputs.append(batch_reps)
    return torch.cat(outputs, dim=0)


def _resolve_device(use_cpu: bool) -> torch.device:
    if use_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _resolve_source_exclude_token_ids(
    cfg: DictConfig, tokenizer: PreTrainedTokenizerBase
) -> list[int]:
    configured_ids: Any | None = cfg.model.exclude_token_ids
    if configured_ids is not None and len(configured_ids) > 0:
        return [int(token_id) for token_id in configured_ids]
    return [int(token_id) for token_id in tokenizer.all_special_ids]


def _extract_terms(
    vector: np.ndarray,
    *,
    output_space: OutputSpaceSpec,
    tokenizer: PreTrainedTokenizerBase | None,
    exclude_output_ids: list[int],
    min_weight: float,
    top_k: int | None,
) -> list[dict[str, Any]]:
    indices, values = sparsify_query_vector(
        vector,
        exclude_output_ids=exclude_output_ids,
        min_weight=float(min_weight),
        top_k=top_k,
    )
    if indices.size == 0:
        return []
    order: np.ndarray = np.argsort(values)[::-1]
    indices = indices[order]
    values = values[order]
    token_labels: list[str | None]
    if output_space.output_token_aligned and tokenizer is not None:
        token_labels = list(tokenizer.convert_ids_to_tokens(indices.tolist()))
    else:
        token_labels = [None] * int(indices.size)
    terms: list[dict[str, Any]] = []
    for rank, (output_id, token, value) in enumerate(
        zip(indices, token_labels, values), start=1
    ):
        term_payload: dict[str, Any] = {
            "rank": int(rank),
            "output_id": int(output_id),
            "weight": float(value),
        }
        if output_space.output_token_aligned:
            term_payload["token_id"] = int(output_id)
            term_payload["token"] = token
        else:
            term_payload["token"] = None
        terms.append(term_payload)
    return terms


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name="analyze_vocab_terms"
)
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)
    cfg = apply_checkpoint_model_config(
        cfg, checkpoint_path=cfg.analysis.checkpoint_path, logger=logger
    )

    analysis_cfg: DictConfig = cfg.analysis
    dataset: Dataset = _load_triplets_dataset(cfg)
    log_if_rank_zero(logger, f"Loaded dataset with {len(dataset)} rows.")

    sample_count: int = int(analysis_cfg.sample_count)
    seed: int = int(analysis_cfg.seed)
    shuffle: bool = bool(analysis_cfg.shuffle)
    sample_rows = _collect_sample_rows(
        dataset,
        sample_count=sample_count,
        seed=seed,
        shuffle=shuffle,
    )

    column_names: set[str] = set(dataset.column_names)
    query_text_column: str = str(analysis_cfg.query_text_column)
    positive_text_column: str = str(analysis_cfg.positive_text_column)
    samples: list[dict[str, str]] = []
    if query_text_column in column_names and positive_text_column in column_names:
        samples = _extract_text_samples(
            sample_rows,
            query_text_column=query_text_column,
            positive_text_column=positive_text_column,
        )
    else:
        query_id_column: str = str(analysis_cfg.query_id_column)
        positive_id_column: str = str(analysis_cfg.positive_id_column)
        if (
            query_id_column not in column_names
            or positive_id_column not in column_names
        ):
            raise ValueError(
                "Triplets dataset must contain either text columns "
                f"({query_text_column}, {positive_text_column}) or id columns "
                f"({query_id_column}, {positive_id_column})."
            )
        query_ids: list[str] = [
            str(row[query_id_column])
            for row in sample_rows
            if row.get(query_id_column) is not None
        ]
        positive_ids: list[str] = [
            str(row[positive_id_column])
            for row in sample_rows
            if row.get(positive_id_column) is not None
        ]
        query_id_set: set[str] = set(query_ids)
        positive_id_set: set[str] = set(positive_ids)

        hf_name: str = str(analysis_cfg.hf_name)
        cache_dir: str | None = normalize_optional_str(analysis_cfg.hf_cache_dir)
        data_files: Any | None = analysis_cfg.hf_data_files
        query_dataset = _load_text_dataset(
            hf_name=hf_name,
            hf_subset=normalize_optional_str(analysis_cfg.query_subset_name),
            split=str(analysis_cfg.split),
            cache_dir=cache_dir,
            data_files=data_files,
            streaming=False,
        )
        corpus_dataset = _load_text_dataset(
            hf_name=hf_name,
            hf_subset=normalize_optional_str(analysis_cfg.corpus_subset_name),
            split=str(analysis_cfg.split),
            cache_dir=cache_dir,
            data_files=data_files,
            streaming=False,
        )
        query_texts_map = _lookup_texts(
            query_dataset,
            id_column=str(analysis_cfg.query_id_column_name),
            text_column=str(analysis_cfg.query_text_column_name),
            wanted_ids=query_id_set,
        )
        corpus_texts_map, missing_corpus_ids = _lookup_texts_by_index(
            corpus_dataset,
            id_column=str(analysis_cfg.corpus_id_column_name),
            text_column=str(analysis_cfg.corpus_text_column_name),
            wanted_ids=positive_id_set,
        )
        if missing_corpus_ids:
            fallback_texts = _lookup_texts(
                corpus_dataset,
                id_column=str(analysis_cfg.corpus_id_column_name),
                text_column=str(analysis_cfg.corpus_text_column_name),
                wanted_ids=missing_corpus_ids,
            )
            corpus_texts_map.update(fallback_texts)
        if len(query_texts_map) < len(query_id_set):
            log_if_rank_zero(
                logger,
                "Missing %d query texts from lookup.",
                len(query_id_set) - len(query_texts_map),
                level="warning",
            )
        if len(corpus_texts_map) < len(positive_id_set):
            log_if_rank_zero(
                logger,
                "Missing %d corpus texts from lookup.",
                len(positive_id_set) - len(corpus_texts_map),
                level="warning",
            )
        for row in sample_rows:
            query_id_value: Any | None = row.get(query_id_column)
            positive_id_value: Any | None = row.get(positive_id_column)
            if query_id_value is None or positive_id_value is None:
                continue
            query_text = query_texts_map.get(str(query_id_value))
            positive_text = corpus_texts_map.get(str(positive_id_value))
            if query_text is None or positive_text is None:
                continue
            samples.append({"query_text": query_text, "positive_text": positive_text})

    if not samples:
        log_if_rank_zero(logger, "No valid samples found.", level="warning")
    else:
        log_if_rank_zero(logger, f"Collected {len(samples)} samples.")

    device: torch.device = _resolve_device(bool(analysis_cfg.use_cpu))
    model = build_splade_model_with_checkpoint(
        cfg=cfg,
        use_cpu=bool(analysis_cfg.use_cpu),
        checkpoint_path=str(analysis_cfg.checkpoint_path),
        logger=logger,
    )
    model.to(device)
    tokenizer: PreTrainedTokenizerBase = build_tokenizer(
        str(cfg.model.huggingface_name),
        use_fast_tokenizer=bool(cfg.model.use_fast_tokenizer),
        trust_remote_code=bool(cfg.model.trust_remote_code),
        require_fast_tokenizer=bool(cfg.model.require_fast_tokenizer),
    )
    validate_lens_tokenizer(tokenizer, cfg.model)

    query_texts: list[str] = [item["query_text"] for item in samples]
    positive_texts: list[str] = [item["positive_text"] for item in samples]
    max_length: int = int(analysis_cfg.max_input_length)
    batch_size: int = int(analysis_cfg.batch_size)

    query_reps: torch.Tensor = _encode_texts(
        model,
        tokenizer,
        query_texts,
        model_cfg=cfg.model,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        encode_fn=model.encode_queries,
        is_query=True,
    )
    positive_reps: torch.Tensor = _encode_texts(
        model,
        tokenizer,
        positive_texts,
        model_cfg=cfg.model,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        encode_fn=model.encode_docs,
        is_query=False,
    )

    source_exclude_token_ids: list[int] = _resolve_source_exclude_token_ids(
        cfg, tokenizer
    )
    exclude_output_ids: list[int] = resolve_model_output_exclude_ids(
        model, source_exclude_token_ids
    )
    output_space: OutputSpaceSpec = model.encoder.output_space
    min_weight: float = float(analysis_cfg.min_weight)
    top_k_value: Any | None = analysis_cfg.top_k
    top_k: int | None = None if top_k_value is None else int(top_k_value)

    query_reps_cpu: np.ndarray = (
        query_reps.detach().cpu().float().numpy()
        if int(query_reps.numel()) > 0
        else np.empty((0, 0), dtype=np.float32)
    )
    positive_reps_cpu: np.ndarray = (
        positive_reps.detach().cpu().float().numpy()
        if int(positive_reps.numel()) > 0
        else np.empty((0, 0), dtype=np.float32)
    )

    sample_entries: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        query_terms = _extract_terms(
            query_reps_cpu[idx],
            output_space=output_space,
            tokenizer=tokenizer,
            exclude_output_ids=exclude_output_ids,
            min_weight=min_weight,
            top_k=top_k,
        )
        positive_terms = _extract_terms(
            positive_reps_cpu[idx],
            output_space=output_space,
            tokenizer=tokenizer,
            exclude_output_ids=exclude_output_ids,
            min_weight=min_weight,
            top_k=top_k,
        )
        sample_entries.append(
            {
                "sample_idx": int(idx),
                "query_text": sample["query_text"],
                "positive_text": sample["positive_text"],
                "query_terms": query_terms,
                "positive_terms": positive_terms,
            }
        )

    metadata: dict[str, Any] = {
        "checkpoint_path": str(analysis_cfg.checkpoint_path),
        "hf_name": str(analysis_cfg.hf_name),
        "hf_subset": normalize_optional_str(analysis_cfg.hf_subset),
        "split": str(analysis_cfg.split),
        "sample_count": sample_count,
        "sampled_count": len(samples),
        "shuffle": shuffle,
        "seed": seed,
        "compact_head_alignment": output_space.compact_head_alignment,
        "output_token_aligned": output_space.output_token_aligned,
        "source_exclude_token_ids": source_exclude_token_ids,
        "exclude_output_ids": exclude_output_ids,
        "top_k": top_k,
        "min_weight": min_weight,
        "max_input_length": max_length,
        "batch_size": batch_size,
        "use_cpu": bool(analysis_cfg.use_cpu),
    }
    output_payload: dict[str, Any] = {
        "metadata": metadata,
        "samples": sample_entries,
    }

    output_path: Path = Path(str(analysis_cfg.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)
    log_if_rank_zero(logger, f"Saved vocab terms to {output_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
