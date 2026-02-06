import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, IterableDataset, load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

LOGGER: logging.Logger = logging.getLogger("scripts.experiment.logit_stats")


@dataclass
class RunningLogitStats:
    """Accumulate streaming statistics for logits."""

    count: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    neg_count: int = 0
    zero_count: int = 0

    def update(self, logits: torch.Tensor) -> None:
        """Update running stats from a batch of logits."""
        logits_fp32: torch.Tensor = logits.float()
        batch_count: int = int(logits_fp32.numel())
        if batch_count == 0:
            return
        batch_sum: float = float(logits_fp32.sum().item())
        batch_sumsq: float = float((logits_fp32 * logits_fp32).sum().item())
        batch_min: float = float(logits_fp32.min().item())
        batch_max: float = float(logits_fp32.max().item())
        batch_neg: int = int((logits_fp32 < 0).sum().item())
        batch_zero: int = int((logits_fp32 == 0).sum().item())

        self.count += batch_count
        self.sum += batch_sum
        self.sumsq += batch_sumsq
        self.min_value = min(self.min_value, batch_min)
        self.max_value = max(self.max_value, batch_max)
        self.neg_count += batch_neg
        self.zero_count += batch_zero

    def finalize(self) -> dict[str, float]:
        """Compute final aggregate stats."""
        safe_count: int = max(self.count, 1)
        mean: float = self.sum / float(safe_count)
        variance: float = max((self.sumsq / float(safe_count)) - mean * mean, 0.0)
        std: float = math.sqrt(variance)
        neg_fraction: float = self.neg_count / float(safe_count)
        zero_fraction: float = self.zero_count / float(safe_count)
        return {
            "count": float(self.count),
            "mean": mean,
            "std": std,
            "min": self.min_value,
            "max": self.max_value,
            "negative_fraction": neg_fraction,
            "zero_fraction": zero_fraction,
        }


@dataclass
class RunningCountStats:
    """Accumulate streaming statistics for count values."""

    count: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def update(self, values: torch.Tensor) -> None:
        """Update running stats from a batch of counts."""
        values_fp32: torch.Tensor = values.float()
        batch_count: int = int(values_fp32.numel())
        if batch_count == 0:
            return
        batch_sum: float = float(values_fp32.sum().item())
        batch_sumsq: float = float((values_fp32 * values_fp32).sum().item())
        batch_min: float = float(values_fp32.min().item())
        batch_max: float = float(values_fp32.max().item())

        self.count += batch_count
        self.sum += batch_sum
        self.sumsq += batch_sumsq
        self.min_value = min(self.min_value, batch_min)
        self.max_value = max(self.max_value, batch_max)

    def finalize(self) -> dict[str, float]:
        """Compute final aggregate stats."""
        safe_count: int = max(self.count, 1)
        mean: float = self.sum / float(safe_count)
        variance: float = max((self.sumsq / float(safe_count)) - mean * mean, 0.0)
        std: float = math.sqrt(variance)
        return {
            "count": float(self.count),
            "mean": mean,
            "std": std,
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass(frozen=True)
class ModelResult:
    """Container for per-model statistics and sampled logits."""

    model_name: str
    stats: dict[str, float]
    sample_values: np.ndarray
    vocab_survival: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-safe dict."""
        payload: dict[str, Any] = asdict(self)
        payload["sample_values"] = []  # Avoid dumping raw logits in JSON.
        return payload


def _load_hf_dataset(
    hf_name: str,
    hf_subset: str | None,
    split: str,
    cache_dir: str | None,
    streaming: bool,
) -> Dataset | IterableDataset:
    """Load a Hugging Face dataset with optional subset."""
    if hf_subset:
        return load_dataset(
            hf_name,
            name=hf_subset,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    return load_dataset(
        hf_name,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )


def _resolve_text_column(column_names: Sequence[str], preferred: str | None) -> str:
    """Resolve the query text column from dataset columns."""
    column_set: set[str] = set(column_names)
    if preferred is not None:
        if preferred in column_set:
            return preferred
        raise ValueError(
            f"Requested text column '{preferred}' not in {sorted(column_set)}"
        )
    for candidate in ("text", "query", "anchor"):
        if candidate in column_set:
            return candidate
    raise ValueError(f"Unable to resolve query text column from {sorted(column_set)}")


def _normalize_optional_str(value: str | None) -> str | None:
    """Normalize optional CLI strings into None."""
    if value is None:
        return None
    normalized: str = value.strip()
    if normalized.lower() in {"", "none", "null"}:
        return None
    return normalized


def _apply_dataset_shuffle(
    dataset: Dataset | IterableDataset, *, seed: int, shuffle_buffer_size: int
) -> Dataset | IterableDataset:
    """Shuffle the dataset, respecting streaming vs map-style semantics."""
    if shuffle_buffer_size <= 0:
        return dataset
    if isinstance(dataset, IterableDataset):
        return dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    return dataset.shuffle(seed=seed)


def _apply_dataset_skip(
    dataset: Dataset | IterableDataset, *, skip_samples: int
) -> Dataset | IterableDataset:
    """Skip a fixed number of samples from the dataset."""
    if skip_samples <= 0:
        return dataset
    if isinstance(dataset, IterableDataset):
        return dataset.skip(skip_samples)
    dataset_length: int = int(len(dataset))
    start_index: int = min(skip_samples, dataset_length)
    return dataset.select(range(start_index, dataset_length))


def _collect_queries(
    dataset: Dataset | IterableDataset,
    text_column: str,
    sample_count: int,
) -> list[str]:
    """Collect a fixed number of query texts from a dataset."""
    queries: list[str] = []
    for row in dataset:
        text_value: Any = row.get(text_column)
        if text_value is None:
            continue
        query_text: str = str(text_value).strip()
        if not query_text:
            continue
        queries.append(query_text)
        if len(queries) >= sample_count:
            break
    return queries


def _load_msmarco_queries(
    *,
    hf_name: str,
    query_subset: str,
    fallback_subset: str | None,
    split: str,
    cache_dir: str | None,
    streaming: bool,
    shuffle_buffer_size: int,
    skip_samples: int,
    text_column: str | None,
    sample_count: int,
    seed: int,
) -> list[str]:
    """Load MSMARCO queries from Hugging Face datasets."""
    dataset: Dataset | IterableDataset
    dataset_source: str = f"{hf_name}:{query_subset}"
    try:
        dataset = _load_hf_dataset(hf_name, query_subset, split, cache_dir, streaming)
    except Exception as exc:  # pylint: disable=broad-except
        if fallback_subset is None:
            raise
        LOGGER.warning(
            "Failed to load queries subset '%s' from %s: %s; falling back to '%s'.",
            query_subset,
            hf_name,
            exc,
            fallback_subset,
        )
        dataset = _load_hf_dataset(
            hf_name, fallback_subset, split, cache_dir, streaming
        )
        dataset_source = f"{hf_name}:{fallback_subset}"

    dataset = _apply_dataset_skip(dataset, skip_samples=skip_samples)
    dataset = _apply_dataset_shuffle(
        dataset, seed=seed, shuffle_buffer_size=shuffle_buffer_size
    )
    column_names: Sequence[str] = dataset.column_names
    resolved_column: str = _resolve_text_column(column_names, text_column)
    queries: list[str] = _collect_queries(dataset, resolved_column, sample_count)
    if len(queries) < sample_count:
        LOGGER.warning(
            "Collected %d/%d queries from %s.",
            len(queries),
            sample_count,
            dataset_source,
        )
    return queries


def _resolve_device(requested: str) -> torch.device:
    """Resolve the compute device."""
    requested_lower: str = requested.lower()
    if requested_lower == "cpu":
        return torch.device("cpu")
    if requested_lower == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sample_logits(
    logits: torch.Tensor,
    sample_size: int,
    generator: torch.Generator,
) -> np.ndarray:
    """Take a random sample of logits for histogram/quantiles."""
    logits_flat: torch.Tensor = logits.detach().float().cpu().view(-1)
    total_logits: int = int(logits_flat.numel())
    if total_logits == 0 or sample_size <= 0:
        return np.empty((0,), dtype=np.float32)
    effective_size: int = min(sample_size, total_logits)
    perm: torch.Tensor = torch.randperm(total_logits, generator=generator)[
        :effective_size
    ]
    sample_values: torch.Tensor = logits_flat[perm]
    return sample_values.numpy()


def _compute_model_logits_stats(
    model_name: str,
    texts: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
    sample_size_per_batch: int,
    device: torch.device,
    seed: int,
) -> ModelResult:
    """Compute streaming stats and sampled logits for a model."""
    tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
    model: Any = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    stats: RunningLogitStats = RunningLogitStats()
    vocab_survival_stats: RunningCountStats = RunningCountStats()
    sample_chunks: list[np.ndarray] = []
    torch_generator: torch.Generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(seed)
    vocab_size: int = int(model.config.vocab_size)

    with torch.no_grad():
        batch_start: int = 0
        while batch_start < len(texts):
            batch_end: int = min(batch_start + batch_size, len(texts))
            batch_texts: Sequence[str] = texts[batch_start:batch_end]
            encoded: dict[str, torch.Tensor] = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids: torch.Tensor = encoded["input_ids"].to(device)
            attention_mask: torch.Tensor = encoded["attention_mask"].to(device)
            outputs: Any = model(input_ids=input_ids, attention_mask=attention_mask)
            logits: torch.Tensor = outputs.logits

            # Remove padding positions to avoid skewing distribution.
            valid_positions: torch.Tensor = attention_mask.to(dtype=torch.bool)
            valid_logits: torch.Tensor = logits[valid_positions]

            stats.update(valid_logits)
            sample_chunk: np.ndarray = _sample_logits(
                valid_logits,
                sample_size=sample_size_per_batch,
                generator=torch_generator,
            )
            sample_chunks.append(sample_chunk)

            token_scores: torch.Tensor = torch.log1p(torch.relu(logits))
            mask: torch.Tensor = attention_mask.unsqueeze(-1).to(token_scores.dtype)
            neg_inf: torch.Tensor = token_scores.new_tensor(float("-inf"))
            masked_scores: torch.Tensor = token_scores.masked_fill(mask == 0, neg_inf)
            pooled_max: torch.Tensor = torch.clamp(
                masked_scores.max(dim=1).values, min=0.0
            )
            survival_counts: torch.Tensor = (pooled_max > 0).sum(dim=1)
            vocab_survival_stats.update(survival_counts)
            batch_start = batch_end

    sample_values: np.ndarray = (
        np.concatenate(sample_chunks)
        if sample_chunks
        else np.empty((0,), dtype=np.float32)
    )
    survival_payload: dict[str, float] = vocab_survival_stats.finalize()
    vocab_size_value: float = float(vocab_size)
    survival_payload["vocab_size"] = vocab_size_value
    survival_payload["mean_fraction"] = survival_payload["mean"] / vocab_size_value
    survival_payload["std_fraction"] = survival_payload["std"] / vocab_size_value
    survival_payload["min_fraction"] = survival_payload["min"] / vocab_size_value
    survival_payload["max_fraction"] = survival_payload["max"] / vocab_size_value
    result: ModelResult = ModelResult(
        model_name=model_name,
        stats=stats.finalize(),
        sample_values=sample_values,
        vocab_survival=survival_payload,
    )
    return result


def _add_quantiles(stats: dict[str, Any], sample_values: np.ndarray) -> dict[str, Any]:
    """Add quantiles computed from sampled logits."""
    if sample_values.size == 0:
        stats["quantiles"] = {}
        return stats
    quantile_points: list[float] = [0.01, 0.05, 0.5, 0.95, 0.99]
    quantiles: dict[str, float] = {}
    for point in quantile_points:
        quantiles[f"{point:.2f}"] = float(np.quantile(sample_values, point))
    stats["quantiles"] = quantiles
    stats["sample_count"] = int(sample_values.size)
    return stats


def _write_results(
    output_path: str,
    payload: dict[str, Any],
) -> None:
    """Write the statistics to disk as JSON."""
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def _plot_histogram(
    output_path: str,
    model_results: Sequence[ModelResult],
) -> None:
    """Plot a histogram comparing sampled logits."""
    if not model_results:
        return
    sample_arrays: list[np.ndarray] = [
        result.sample_values
        for result in model_results
        if result.sample_values.size > 0
    ]
    if not sample_arrays:
        return
    combined: np.ndarray = np.concatenate(sample_arrays)
    min_value: float = float(np.min(combined))
    max_value: float = float(np.max(combined))
    bins: np.ndarray = np.linspace(min_value, max_value, num=201)

    plt.figure(figsize=(9.0, 5.0))
    for result in model_results:
        if result.sample_values.size == 0:
            continue
        plt.hist(
            result.sample_values,
            bins=bins,
            alpha=0.5,
            density=True,
            label=result.model_name,
        )
    plt.title("MLM Logit Distribution (Sampled)")
    plt.xlabel("Logit value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compare MLM logit distributions for BERT vs DistilBERT."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("scripts", "experiment", "output"),
        help="Directory for JSON results and plots.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=32,
        help="Number of MSMARCO queries to evaluate per model.",
    )
    parser.add_argument(
        "--msmarco_name",
        type=str,
        default="sentence-transformers/msmarco",
        help="HF dataset name for MSMARCO.",
    )
    parser.add_argument(
        "--msmarco_query_subset",
        type=str,
        default="queries",
        help="HF subset for MSMARCO queries.",
    )
    parser.add_argument(
        "--msmarco_fallback_subset",
        type=str,
        default="triplets",
        help="Fallback subset if queries subset is unavailable.",
    )
    parser.add_argument(
        "--msmarco_split",
        type=str,
        default="train",
        help="Dataset split for MSMARCO queries.",
    )
    parser.add_argument(
        "--msmarco_cache_dir",
        type=str,
        default=None,
        help="Optional HF cache directory for MSMARCO.",
    )
    parser.add_argument(
        "--msmarco_text_column",
        type=str,
        default=None,
        help="Override the query text column name.",
    )
    parser.add_argument(
        "--msmarco_shuffle_buffer_size",
        type=int,
        default=10000,
        help="Shuffle buffer size for streaming MSMARCO.",
    )
    parser.add_argument(
        "--msmarco_skip_samples",
        type=int,
        default=0,
        help="Skip a fixed number of MSMARCO samples.",
    )
    parser.add_argument(
        "--msmarco_streaming",
        action="store_true",
        help="Enable streaming MSMARCO dataset loading.",
    )
    parser.add_argument(
        "--no_msmarco_streaming",
        action="store_false",
        dest="msmarco_streaming",
        help="Disable streaming MSMARCO dataset loading.",
    )
    parser.set_defaults(msmarco_streaming=True)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Max token length for inputs.",
    )
    parser.add_argument(
        "--sample_size_per_batch",
        type=int,
        default=20000,
        help="Number of logits to sample per batch for histograms.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection: auto, cpu, or cuda.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the experiment script."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args: argparse.Namespace = _parse_args()

    device: torch.device = _resolve_device(args.device)
    msmarco_cache_dir: str | None = _normalize_optional_str(args.msmarco_cache_dir)
    msmarco_text_column: str | None = _normalize_optional_str(args.msmarco_text_column)
    msmarco_fallback: str | None = _normalize_optional_str(args.msmarco_fallback_subset)
    queries: list[str] = _load_msmarco_queries(
        hf_name=str(args.msmarco_name),
        query_subset=str(args.msmarco_query_subset),
        fallback_subset=msmarco_fallback,
        split=str(args.msmarco_split),
        cache_dir=msmarco_cache_dir,
        streaming=bool(args.msmarco_streaming),
        shuffle_buffer_size=int(args.msmarco_shuffle_buffer_size),
        skip_samples=int(args.msmarco_skip_samples),
        text_column=msmarco_text_column,
        sample_count=int(args.sample_count),
        seed=int(args.seed),
    )
    if not queries:
        raise RuntimeError("No MSMARCO queries available for the experiment.")

    output_dir: str = str(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stats_path: str = os.path.join(output_dir, "logit_stats.json")
    plot_path: str = os.path.join(output_dir, "logit_histogram.png")

    model_names: list[str] = [
        "bert-base-uncased",
        "distilbert-base-uncased",
    ]

    model_results: list[ModelResult] = []
    for model_name in model_names:
        LOGGER.info("Computing logits for %s on %s", model_name, device.type)
        result: ModelResult = _compute_model_logits_stats(
            model_name=model_name,
            texts=queries,
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            sample_size_per_batch=int(args.sample_size_per_batch),
            device=device,
            seed=int(args.seed),
        )
        model_results.append(result)

    payload: dict[str, Any] = {
        "settings": {
            "device": str(device),
            "sample_count": int(args.sample_count),
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "sample_size_per_batch": int(args.sample_size_per_batch),
            "seed": int(args.seed),
            "msmarco_name": str(args.msmarco_name),
            "msmarco_query_subset": str(args.msmarco_query_subset),
            "msmarco_fallback_subset": msmarco_fallback,
            "msmarco_split": str(args.msmarco_split),
            "msmarco_cache_dir": msmarco_cache_dir,
            "msmarco_text_column": msmarco_text_column,
            "msmarco_shuffle_buffer_size": int(args.msmarco_shuffle_buffer_size),
            "msmarco_skip_samples": int(args.msmarco_skip_samples),
            "msmarco_streaming": bool(args.msmarco_streaming),
        },
        "models": {},
    }
    for result in model_results:
        model_payload: dict[str, Any] = result.to_json()
        model_payload["stats"] = _add_quantiles(
            model_payload["stats"], result.sample_values
        )
        payload["models"][result.model_name] = model_payload

    _write_results(stats_path, payload)
    _plot_histogram(plot_path, model_results)

    LOGGER.info("Wrote stats to %s", stats_path)
    LOGGER.info("Wrote plot to %s", plot_path)


if __name__ == "__main__":
    main()
