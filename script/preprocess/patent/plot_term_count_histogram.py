#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any

import ijson
import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True)
class CountStats:
    counts: list[int]
    histogram: Counter[int]
    payload_mode_counts: Counter[str]
    source_bucket_counts: list[int]
    source_bucket_histogram: Counter[int]


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def inspect_term_payload(term_weights: Any) -> tuple[str, int, int]:
    if not isinstance(term_weights, dict):
        raise ValueError(f"Expected a dict payload, got: {type(term_weights).__name__}")
    if not term_weights:
        return "empty", 0, 0

    saw_flat_terms: bool = False
    saw_source_terms: bool = False
    total_term_count: int = 0
    source_bucket_count: int = 0

    value: Any
    for value in term_weights.values():
        if isinstance(value, dict):
            saw_source_terms = True
            source_bucket_count += 1
            nested_weight: Any
            for nested_weight in value.values():
                if not _is_numeric_value(nested_weight):
                    raise ValueError("Nested source-token payload values must be numeric weights.")
                total_term_count += 1
            continue
        if _is_numeric_value(value):
            saw_flat_terms = True
            total_term_count += 1
            continue
        raise ValueError(f"Unsupported term payload value type: {type(value).__name__}")

    if saw_flat_terms and saw_source_terms:
        raise ValueError("Mixed flat and source-token term payloads are not supported.")
    if saw_source_terms:
        return "source_token_terms", total_term_count, source_bucket_count
    return "flat_terms", total_term_count, 0


def resolve_file_payload_mode(payload_mode_counts: Counter[str]) -> str:
    active_modes = {
        mode
        for mode, count in payload_mode_counts.items()
        if count > 0 and mode != "empty"
    }
    if not active_modes:
        return "empty"
    if len(active_modes) == 1:
        return next(iter(active_modes))
    return "mixed"


def count_terms_per_doc(input_path: Path) -> CountStats:
    counts: list[int] = []
    histogram: Counter[int] = Counter()
    payload_mode_counts: Counter[str] = Counter()
    source_bucket_counts: list[int] = []
    source_bucket_histogram: Counter[int] = Counter()

    with input_path.open("rb") as handle:
        for _, term_weights in ijson.kvitems(handle, ""):
            payload_mode, term_count, source_bucket_count = inspect_term_payload(term_weights)
            counts.append(term_count)
            histogram[term_count] += 1
            payload_mode_counts[payload_mode] += 1
            source_bucket_counts.append(source_bucket_count)
            source_bucket_histogram[source_bucket_count] += 1

    return CountStats(
        counts=counts,
        histogram=histogram,
        payload_mode_counts=payload_mode_counts,
        source_bucket_counts=source_bucket_counts,
        source_bucket_histogram=source_bucket_histogram,
    )


def build_summary(input_path: Path, stats: CountStats) -> dict:
    if not stats.counts:
        raise ValueError(f"No documents found in {input_path}")

    counts_array = np.asarray(stats.counts, dtype=np.int32)
    most_common = [
        {"term_count": int(term_count), "document_count": int(document_count)}
        for term_count, document_count in stats.histogram.most_common(20)
    ]

    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "payload_mode": resolve_file_payload_mode(stats.payload_mode_counts),
        "payload_mode_counts": {
            str(mode): int(count)
            for mode, count in sorted(stats.payload_mode_counts.items())
        },
        "document_count": int(counts_array.size),
        "min_terms_per_doc": int(counts_array.min()),
        "max_terms_per_doc": int(counts_array.max()),
        "mean_terms_per_doc": float(counts_array.mean()),
        "median_terms_per_doc": float(np.median(counts_array)),
        "percentiles": {
            "p01": float(np.percentile(counts_array, 1)),
            "p05": float(np.percentile(counts_array, 5)),
            "p10": float(np.percentile(counts_array, 10)),
            "p25": float(np.percentile(counts_array, 25)),
            "p50": float(np.percentile(counts_array, 50)),
            "p75": float(np.percentile(counts_array, 75)),
            "p90": float(np.percentile(counts_array, 90)),
            "p95": float(np.percentile(counts_array, 95)),
            "p99": float(np.percentile(counts_array, 99)),
        },
        "term_count_histogram": {
            str(term_count): int(document_count)
            for term_count, document_count in sorted(stats.histogram.items())
        },
        "most_common": most_common,
    }

    if int(stats.payload_mode_counts.get("source_token_terms", 0)) > 0:
        source_bucket_array = np.asarray(stats.source_bucket_counts, dtype=np.int32)
        summary["source_bucket_stats"] = {
            "min_source_buckets_per_doc": int(source_bucket_array.min()),
            "max_source_buckets_per_doc": int(source_bucket_array.max()),
            "mean_source_buckets_per_doc": float(source_bucket_array.mean()),
            "median_source_buckets_per_doc": float(np.median(source_bucket_array)),
            "percentiles": {
                "p01": float(np.percentile(source_bucket_array, 1)),
                "p05": float(np.percentile(source_bucket_array, 5)),
                "p10": float(np.percentile(source_bucket_array, 10)),
                "p25": float(np.percentile(source_bucket_array, 25)),
                "p50": float(np.percentile(source_bucket_array, 50)),
                "p75": float(np.percentile(source_bucket_array, 75)),
                "p90": float(np.percentile(source_bucket_array, 90)),
                "p95": float(np.percentile(source_bucket_array, 95)),
                "p99": float(np.percentile(source_bucket_array, 99)),
            },
            "source_bucket_histogram": {
                str(bucket_count): int(document_count)
                for bucket_count, document_count in sorted(stats.source_bucket_histogram.items())
            },
        }

    return summary


def plot_histogram(
    counts: list[int],
    title: str,
    output_path: Path,
    bins: int,
    x_min: float | None,
    x_max: float | None,
) -> None:
    counts_array = np.asarray(counts, dtype=np.int32)
    min_count = int(counts_array.min())
    max_count = int(counts_array.max())

    if bins <= 0:
        raise ValueError("--bins must be positive")

    if min_count == max_count:
        edges = np.array([min_count - 0.5, max_count + 0.5], dtype=np.float64)
    else:
        edges = np.linspace(min_count - 0.5, max_count + 0.5, num=bins + 1)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.hist(counts_array, bins=edges, color="#1768ac", alpha=0.88, edgecolor="white")
    ax.set_ylabel("Document Count")
    ax.set_xlabel("Terms per Document")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title(title)
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a histogram of term counts per document from a SPLADE JSON export."
    )
    parser.add_argument("--input", required=True, help="Path to the merged JSON export.")
    parser.add_argument("--output-png", required=True, help="Path to the histogram PNG.")
    parser.add_argument(
        "--output-summary",
        required=True,
        help="Path to the JSON summary containing aggregate stats and exact counts.",
    )
    parser.add_argument(
        "--title",
        default="Patent SPLADE Terms",
        help="Title prefix to use on the histogram panels.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="Number of histogram bins to use for the rendered plot.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional minimum x-axis value to display.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional maximum x-axis value to display.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_png = Path(args.output_png)
    output_summary = Path(args.output_summary)

    stats = count_terms_per_doc(input_path)
    summary = build_summary(input_path, stats)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    plot_histogram(stats.counts, args.title, output_png, args.bins, args.x_min, args.x_max)
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
