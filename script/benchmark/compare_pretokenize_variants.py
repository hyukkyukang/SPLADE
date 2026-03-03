#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: list[str]


def _parse_last_step(metrics_path: Path) -> int | None:
    if not metrics_path.is_file():
        return None
    last_step: int | None = None
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_step: str = str(row.get("step") or "").strip()
            if not raw_step:
                continue
            try:
                parsed: int = int(float(raw_step))
            except ValueError:
                continue
            if last_step is None or parsed > last_step:
                last_step = parsed
    return last_step


def _recent_throughput(samples: list[tuple[float, int]]) -> float | None:
    if len(samples) < 2:
        return None
    end_time, end_step = samples[-1]
    for start_time, start_step in reversed(samples[:-1]):
        dt: float = end_time - start_time
        if dt >= 40.0 and end_step > start_step:
            return (end_step - start_step) / dt
    start_time, start_step = samples[0]
    dt = end_time - start_time
    if dt > 0 and end_step > start_step:
        return (end_step - start_step) / dt
    return None


def _metrics_path(base_dir: Path, model_name: str, tag: str) -> Path:
    return (
        base_dir
        / "log"
        / "train"
        / model_name
        / tag
        / "lightning_logs"
        / "version_0"
        / "metrics.csv"
    )


def _default_variants(cache_root: str) -> list[Variant]:
    return [
        Variant(
            name="sidecar_only",
            overrides=[
                "train_dataset.pretokenize.enabled=true",
                f"train_dataset.pretokenize.output_dir={cache_root}/sidecar_only",
                "train_dataset.pretokenize.storage_format=sidecar_only",
                "train_dataset.pretokenize.loading_mode=streaming",
                "train_dataset.pretokenize.streaming_numpy_sidecar=true",
                "train_dataset.pretokenize.overwrite=false",
            ],
        ),
        Variant(
            name="hybrid",
            overrides=[
                "train_dataset.pretokenize.enabled=true",
                f"train_dataset.pretokenize.output_dir={cache_root}/hybrid",
                "train_dataset.pretokenize.storage_format=hybrid",
                "train_dataset.pretokenize.loading_mode=streaming",
                "train_dataset.pretokenize.streaming_numpy_sidecar=true",
                "train_dataset.pretokenize.overwrite=false",
            ],
        ),
        Variant(
            name="no_pretok",
            overrides=["train_dataset.pretokenize.enabled=false"],
        ),
    ]


def _run_variant(
    *,
    base_dir: Path,
    variant: Variant,
    config_name: str,
    model_name: str,
    cuda_visible_devices: str,
    max_steps: int,
    target_steps: int,
    poll_seconds: int,
    max_runtime_seconds: int,
    max_no_step_seconds: int,
    extra_overrides: list[str],
) -> dict[str, Any]:
    timestamp: str = time.strftime("%Y%m%d_%H%M%S")
    tag: str = f"bench_{variant.name}_{timestamp}"

    base_overrides: list[str] = [
        f"tag={tag}",
        "training.num_devices=4",
        "training.strategy=ddp",
        f"training.max_steps={max_steps}",
        "training.val_check_interval=1.0",
        "training.limit_val_batches=0.0",
        "nanobeir.enabled=false",
        "training.torch_compile=false",
        "training.torch_compile_loss=false",
        "training.static_graph=false",
        "training.find_unused_parameters=true",
        "training.batch_size=4",
        "training.grad_accumulation=2",
        "training.num_workers=4",
        "training.log_every_n_steps=1",
        "training.mlflow.enabled=false",
        "train_dataset.hf_max_samples=100000",
        "val_dataset.hf_max_samples=256",
        "val_dataset.pretokenize.enabled=false",
    ]
    all_overrides: list[str] = (
        base_overrides + variant.overrides + list(extra_overrides)
    )
    command: str = " ".join(
        [
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}",
            f"python script/train.py --config-name {config_name}",
            *all_overrides,
        ]
    )
    cmd: list[str] = ["bash", "-lc", command]

    start_time: float = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    metrics_path: Path = _metrics_path(base_dir, model_name, tag)
    samples: list[tuple[float, int]] = []
    first_step_time: float | None = None
    last_step_seen_time: float | None = None
    next_poll_time: float = start_time + float(poll_seconds)

    run_log_path: Path = base_dir / f"benchmark_{variant.name}_{timestamp}.log"
    with run_log_path.open("w", encoding="utf-8") as run_log:
        while True:
            line: str = proc.stdout.readline() if proc.stdout is not None else ""
            if line:
                run_log.write(line)
                run_log.flush()

            now: float = time.time()
            if now >= next_poll_time:
                step: int | None = _parse_last_step(metrics_path)
                if step is not None:
                    samples.append((now, int(step)))
                    if first_step_time is None and int(step) > 0:
                        first_step_time = now
                    if int(step) > 0:
                        last_step_seen_time = now
                    recent_tput: float | None = _recent_throughput(samples)
                    tput_text: str = (
                        "NA" if recent_tput is None else f"{recent_tput:.4f}"
                    )
                    print(
                        f"[{variant.name}] step={step} recent_step_per_s={tput_text}",
                        flush=True,
                    )
                else:
                    print(f"[{variant.name}] waiting_for_metrics", flush=True)
                next_poll_time = now + float(poll_seconds)

            if (now - start_time) > float(max_runtime_seconds):
                print(
                    f"[{variant.name}] timeout={max_runtime_seconds}s; terminating",
                    flush=True,
                )
                proc.terminate()
                time.sleep(5)
                if proc.poll() is None:
                    proc.kill()
                break
            if (
                first_step_time is not None
                and last_step_seen_time is not None
                and (now - last_step_seen_time) > float(max_no_step_seconds)
            ):
                print(
                    f"[{variant.name}] no_progress={max_no_step_seconds}s; terminating",
                    flush=True,
                )
                proc.terminate()
                time.sleep(5)
                if proc.poll() is None:
                    proc.kill()
                break
            if proc.poll() is not None:
                if proc.stdout is not None:
                    for tail in proc.stdout:
                        run_log.write(tail)
                break

    end_time: float = time.time()
    final_step: int = _parse_last_step(metrics_path) or -1
    throughput_recent: float | None = _recent_throughput(samples)
    if throughput_recent is None and final_step > 0:
        elapsed: float = max(end_time - start_time, 1e-6)
        throughput_recent = final_step / elapsed

    startup_seconds: float | None = None
    if first_step_time is not None:
        startup_seconds = first_step_time - start_time

    eta_hours: float | None = None
    if throughput_recent is not None and throughput_recent > 0 and final_step >= 0:
        remaining: int = max(0, int(target_steps) - int(final_step))
        eta_hours = remaining / throughput_recent / 3600.0

    return {
        "variant": variant.name,
        "tag": tag,
        "returncode": int(proc.returncode or 0),
        "run_minutes": round((end_time - start_time) / 60.0, 3),
        "startup_seconds_to_first_step": None
        if startup_seconds is None
        else round(startup_seconds, 3),
        "final_step": int(final_step),
        "throughput_step_per_s_recent": None
        if throughput_recent is None
        else round(throughput_recent, 4),
        "projected_eta_hours_for_target_steps_recent": None
        if eta_hours is None
        else round(eta_hours, 2),
        "target_steps": int(target_steps),
        "run_dir": str((base_dir / "log" / "train" / model_name / tag).resolve()),
        "run_log": str(run_log_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sidecar_only/hybrid/no_pretok training throughput."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="train_embeddinggemma_splade_v2_pp",
        help="Hydra config name for script/train.py.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="splade_v2_pp_embeddinggemma_300m_lsr",
        help="Model name used in log/train/<model>/<tag>/... metrics path.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="0,1,2,3",
        help="CUDA_VISIBLE_DEVICES value for benchmark runs.",
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--target-steps", type=int, default=150000)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-runtime-seconds", type=int, default=1800)
    parser.add_argument("--max-no-step-seconds", type=int, default=360)
    parser.add_argument(
        "--cache-root",
        type=str,
        default="data/cache/pretokenized/bench_pretok_compare",
        help="Root folder for benchmark pretokenize caches.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark_results/pretokenize_compare_latest.json"),
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override (repeatable).",
    )
    args = parser.parse_args()

    base_dir: Path = args.base_dir.resolve()
    variants: list[Variant] = _default_variants(args.cache_root)
    records: list[dict[str, Any]] = []
    for variant in variants:
        print(f"=== Running {variant.name} ===", flush=True)
        record: dict[str, Any] = _run_variant(
            base_dir=base_dir,
            variant=variant,
            config_name=str(args.config_name),
            model_name=str(args.model_name),
            cuda_visible_devices=str(args.cuda_visible_devices),
            max_steps=int(args.max_steps),
            target_steps=int(args.target_steps),
            poll_seconds=int(args.poll_seconds),
            max_runtime_seconds=int(args.max_runtime_seconds),
            max_no_step_seconds=int(args.max_no_step_seconds),
            extra_overrides=list(args.extra_override),
        )
        records.append(record)
        print(json.dumps(record, ensure_ascii=True, indent=2), flush=True)

    output_path: Path = (
        args.output_json
        if args.output_json.is_absolute()
        else (base_dir / args.output_json).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(f"Wrote benchmark summary: {output_path}", flush=True)


if __name__ == "__main__":
    main()
