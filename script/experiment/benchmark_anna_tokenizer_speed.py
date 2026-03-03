"""Benchmark ANNA tokenizer throughput (slow vs fast)."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

_DEFAULT_TEXTS: tuple[str, ...] = (
    "anna conversion validation",
    "Hello, world!",
    "lower UPPER 123",
    "punctuation: period. comma, semi; colon:",
    "CJK 中文 日本語",
    "accents cafe naive",
    "The quick brown fox jumps over the lazy dog.",
)


def _validate_hf_tokenizer_dir(model_dir: Path) -> None:
    required_files: tuple[str, ...] = (
        "vocab.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "anna_tokenizer.py",
    )
    missing: list[str] = [
        filename for filename in required_files if not (model_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing tokenizer artifacts in model dir "
            f"{model_dir}: {', '.join(missing)}"
        )


def _load_texts(
    texts_path: Path | None,
    *,
    max_texts: int,
    repeat: int,
) -> list[str]:
    if texts_path is not None:
        if not texts_path.is_file():
            raise FileNotFoundError(f"Texts file not found: {texts_path}")
        texts: list[str] = []
        with texts_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                texts.append(text)
                if len(texts) >= max_texts:
                    break
        if not texts:
            raise ValueError(f"No non-empty texts were loaded from: {texts_path}")
        return texts
    repeated: list[str] = []
    for _ in range(max(1, int(repeat))):
        repeated.extend(_DEFAULT_TEXTS)
        if len(repeated) >= max_texts:
            break
    return repeated[:max_texts]


def _benchmark(
    tokenizer: object,
    texts: list[str],
    *,
    batch_size: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> list[float]:
    # Use tokenizer() batching to mirror training/inference call paths.
    def run_once() -> float:
        start = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokenizer(
                batch,
                add_special_tokens=True,
                truncation=True,
                max_length=128,
                padding=False,
            )
        return time.perf_counter() - start

    for _ in range(max(0, warmup_runs)):
        _ = run_once()
    return [run_once() for _ in range(max(1, benchmark_runs))]


def _format_stats(name: str, runtimes: list[float], samples: int) -> str:
    mean_s = statistics.fmean(runtimes)
    p50_s = statistics.median(runtimes)
    p95_s = sorted(runtimes)[int(round(0.95 * (len(runtimes) - 1)))]
    qps = samples / mean_s if mean_s > 0 else 0.0
    return (
        f"{name:<18} mean={mean_s:>8.4f}s  "
        f"p50={p50_s:>8.4f}s  p95={p95_s:>8.4f}s  samples/s={qps:>10.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/model/anna_large_hf"),
        help="HF model/tokenizer directory containing anna_tokenizer.py + vocab.txt",
    )
    parser.add_argument(
        "--texts-path",
        type=Path,
        default=None,
        help="Optional newline-delimited text file for benchmark inputs",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=5000,
        help="Number of texts to benchmark",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=800,
        help="When --texts-path is omitted, repeat built-in samples this many times",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    _validate_hf_tokenizer_dir(model_dir)

    texts = _load_texts(
        args.texts_path,
        max_texts=int(args.max_texts),
        repeat=int(args.repeat),
    )
    if not texts:
        raise ValueError("No benchmark texts available")

    slow_tokenizer_cls = get_class_from_dynamic_module(
        "anna_tokenizer.AnnaTokenizer",
        str(model_dir),
        local_files_only=bool(args.local_files_only),
    )
    tokenizer_slow = slow_tokenizer_cls.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )
    tokenizer_fast = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_fast=True,
        local_files_only=bool(args.local_files_only),
    )
    if not bool(tokenizer_fast.is_fast):
        raise RuntimeError(
            "Fast tokenizer backend is not active. "
            "Build/install anna_fast_rs first."
        )

    # Quick parity gate before timing.
    for text in texts[: min(200, len(texts))]:
        slow_tokens = tokenizer_slow.tokenize(text)
        fast_tokens = tokenizer_fast.tokenize(text)
        if slow_tokens != fast_tokens:
            raise RuntimeError(f"Slow/Fast tokenization mismatch for text: {text!r}")

    slow_runtimes = _benchmark(
        tokenizer_slow,
        texts,
        batch_size=int(args.batch_size),
        warmup_runs=int(args.warmup_runs),
        benchmark_runs=int(args.benchmark_runs),
    )
    fast_runtimes = _benchmark(
        tokenizer_fast,
        texts,
        batch_size=int(args.batch_size),
        warmup_runs=int(args.warmup_runs),
        benchmark_runs=int(args.benchmark_runs),
    )

    samples = len(texts)
    slow_mean = statistics.fmean(slow_runtimes)
    fast_mean = statistics.fmean(fast_runtimes)
    speedup = slow_mean / fast_mean if fast_mean > 0 else 0.0

    print(f"Model dir: {model_dir}")
    print(f"Texts: {samples} | Batch size: {int(args.batch_size)}")
    print(_format_stats("slow (python)", slow_runtimes, samples))
    print(_format_stats("fast (rust)", fast_runtimes, samples))
    print(f"Speedup (slow/fast): {speedup:,.2f}x")


if __name__ == "__main__":
    main()
