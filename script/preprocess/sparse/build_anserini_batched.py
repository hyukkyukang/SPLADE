#!/usr/bin/env python3
"""Build an Anserini impact index from sparse shards in bounded-storage batches.

This pipeline exports queries once, then overlaps document export for batch N+1
with Lucene indexing for batch N. Each batch export is parallelized across
multiple worker processes on disjoint shard ranges.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from src.index.sparse import ShardInfo, load_shard_manifest


@dataclass
class BatchSpec:
    batch_idx: int
    start_shard_index: int
    shard_limit: int
    doc_count: int


@dataclass
class ExportWorker:
    worker_idx: int
    start_shard_index: int
    shard_limit: int
    log_path: Path
    log_handle: TextIO
    process: subprocess.Popen[str]


@dataclass
class ExportJob:
    spec: BatchSpec
    batch_dir: Path
    workers: list[ExportWorker]
    started_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encode-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--query-output-dir", type=Path, required=True)
    parser.add_argument("--query-ids-path", type=Path, required=True)
    parser.add_argument("--query-id-column", type=str, default="query_id")
    parser.add_argument("--doc-top-k", type=int, required=True)
    parser.add_argument("--query-top-k", type=int, required=True)
    parser.add_argument("--quantization-factor", type=float, default=100.0)
    parser.add_argument("--batch-shards", type=int, default=8)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-buffer-mb", type=int, default=4096)
    parser.add_argument("--export-workers", type=int, default=8)
    parser.add_argument("--export-progress-every-docs", type=int, default=100000)
    parser.add_argument("--query-progress-every-docs", type=int, default=100000)
    parser.add_argument("--start-batch-index", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--skip-query-export", action="store_true")
    parser.add_argument(
        "--temp-root", type=Path, default=Path("/tmp/anserini_batch_work")
    )
    parser.add_argument(
        "--jar-path",
        type=Path,
        default=Path("tools/anserini/anserini-1.6.0-fatjar.jar"),
    )
    parser.add_argument("--overwrite-index", action="store_true")
    parser.add_argument("--keep-batches", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _partition_shards(start: int, limit: int, workers: int) -> list[tuple[int, int]]:
    worker_count = max(1, min(int(workers), int(limit)))
    base = limit // worker_count
    extra = limit % worker_count
    assignments: list[tuple[int, int]] = []
    cursor = start
    for worker_idx in range(worker_count):
        shard_count = base + (1 if worker_idx < extra else 0)
        if shard_count <= 0:
            continue
        assignments.append((cursor, shard_count))
        cursor += shard_count
    return assignments


def _build_batch_specs(shard_infos: list[ShardInfo], batch_shards: int) -> list[BatchSpec]:
    specs: list[BatchSpec] = []
    for batch_idx, start in enumerate(range(0, len(shard_infos), batch_shards)):
        selected = shard_infos[start : start + batch_shards]
        specs.append(
            BatchSpec(
                batch_idx=batch_idx,
                start_shard_index=start,
                shard_limit=len(selected),
                doc_count=sum(int(shard.doc_count) for shard in selected),
            )
        )
    return specs


def _launch_export_batch(args: argparse.Namespace, spec: BatchSpec, batch_dir: Path) -> ExportJob:
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    (batch_dir / "docs").mkdir(parents=True, exist_ok=True)

    workers: list[ExportWorker] = []
    assignments = _partition_shards(
        spec.start_shard_index,
        spec.shard_limit,
        int(args.export_workers),
    )
    print(
        json.dumps(
            {
                "event": "export_batch_start",
                "batch_idx": int(spec.batch_idx),
                "start_shard_index": int(spec.start_shard_index),
                "shard_limit": int(spec.shard_limit),
                "doc_count": int(spec.doc_count),
                "worker_count": len(assignments),
                "batch_dir": str(batch_dir),
            }
        ),
        flush=True,
    )
    for worker_idx, (worker_start, worker_limit) in enumerate(assignments):
        log_path = batch_dir / f"export_worker_{worker_idx:02d}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        cmd = [
            sys.executable,
            "script/preprocess/sparse/export_anserini_sparse.py",
            "--encode-dir",
            str(args.encode_dir),
            "--tokenizer",
            str(args.tokenizer),
            "--output-dir",
            str(batch_dir),
            "--export-docs",
            "--doc-top-k",
            str(int(args.doc_top_k)),
            "--query-top-k",
            str(int(args.query_top_k)),
            "--quantization-factor",
            str(float(args.quantization_factor)),
            "--start-shard-index",
            str(int(worker_start)),
            "--max-shards",
            str(int(worker_limit)),
            "--progress-every-docs",
            str(int(args.export_progress_every_docs)),
            "--skip-metadata",
        ]
        process = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers.append(
            ExportWorker(
                worker_idx=worker_idx,
                start_shard_index=worker_start,
                shard_limit=worker_limit,
                log_path=log_path,
                log_handle=log_handle,
                process=process,
            )
        )
    return ExportJob(spec=spec, batch_dir=batch_dir, workers=workers, started_at=time.time())


def _stop_export_job(job: ExportJob) -> None:
    for worker in job.workers:
        if worker.process.poll() is None:
            worker.process.terminate()
    for worker in job.workers:
        try:
            worker.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.process.kill()
            worker.process.wait()
        worker.log_handle.close()


def _wait_export_job(job: ExportJob) -> None:
    errors: list[tuple[ExportWorker, int]] = []
    for worker in job.workers:
        return_code = worker.process.wait()
        if return_code != 0:
            errors.append((worker, int(return_code)))
            break
    if errors:
        for worker in job.workers:
            if worker.process.poll() is None:
                worker.process.terminate()
        for worker in job.workers:
            if worker.process.poll() is None:
                try:
                    worker.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.process.kill()
                    worker.process.wait()
    for worker in job.workers:
        if worker.process.poll() is None:
            worker.process.wait()
        if not worker.log_handle.closed:
            worker.log_handle.close()
    if errors:
        first_worker, first_code = errors[0]
        raise RuntimeError(
            f"Export worker failed for batch {job.spec.batch_idx}: worker={first_worker.worker_idx} "
            f"start_shard_index={first_worker.start_shard_index} shard_limit={first_worker.shard_limit} "
            f"exit_code={first_code} log={first_worker.log_path}"
        )
    elapsed = time.time() - job.started_at
    print(
        json.dumps(
            {
                "event": "export_batch_done",
                "batch_idx": int(job.spec.batch_idx),
                "doc_count": int(job.spec.doc_count),
                "elapsed_sec": round(elapsed, 2),
                "batch_dir": str(job.batch_dir),
            }
        ),
        flush=True,
    )


def _run_index_batch(args: argparse.Namespace, spec: BatchSpec, batch_dir: Path) -> None:
    build_cmd = [
        sys.executable,
        "script/preprocess/sparse/build_anserini_index.py",
        "--input-dir",
        str(batch_dir / "docs"),
        "--index-dir",
        str(args.index_dir),
        "--jar-path",
        str(args.jar_path),
        "--threads",
        str(int(args.threads)),
        "--memory-buffer-mb",
        str(int(args.memory_buffer_mb)),
    ]
    if spec.batch_idx == 0:
        if args.overwrite_index:
            build_cmd.append("--overwrite")
    else:
        build_cmd.append("--append")
    started_at = time.time()
    print(
        json.dumps(
            {
                "event": "index_batch_start",
                "batch_idx": int(spec.batch_idx),
                "doc_count": int(spec.doc_count),
                "batch_dir": str(batch_dir),
            }
        ),
        flush=True,
    )
    _run(build_cmd, cwd=Path.cwd())
    elapsed = time.time() - started_at
    print(
        json.dumps(
            {
                "event": "index_batch_done",
                "batch_idx": int(spec.batch_idx),
                "doc_count": int(spec.doc_count),
                "elapsed_sec": round(elapsed, 2),
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    shard_infos, _ = load_shard_manifest(args.encode_dir)
    batch_shards = max(1, int(args.batch_shards))
    batch_specs_all = _build_batch_specs(shard_infos, batch_shards)
    batch_specs = batch_specs_all[max(0, int(args.start_batch_index)) :]
    if args.max_batches is not None:
        batch_specs = batch_specs[: max(0, int(args.max_batches))]
    total_batches = len(batch_specs)
    if total_batches == 0:
        raise RuntimeError(f"No shards found under {args.encode_dir}")

    args.query_output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_query_export:
        query_cmd = [
            sys.executable,
            "script/preprocess/sparse/export_anserini_sparse.py",
            "--encode-dir",
            str(args.encode_dir),
            "--tokenizer",
            str(args.tokenizer),
            "--output-dir",
            str(args.query_output_dir),
            "--export-queries",
            "--query-ids-path",
            str(args.query_ids_path),
            "--query-id-column",
            str(args.query_id_column),
            "--query-top-k",
            str(int(args.query_top_k)),
            "--doc-top-k",
            str(int(args.doc_top_k)),
            "--quantization-factor",
            str(float(args.quantization_factor)),
            "--progress-every-docs",
            str(int(args.query_progress_every_docs)),
        ]
        _run(query_cmd, cwd=Path.cwd())

    slot_dirs = [args.temp_root / "slot_0", args.temp_root / "slot_1"]
    prepared_job = _launch_export_batch(args, batch_specs[0], slot_dirs[0])
    _wait_export_job(prepared_job)

    for batch_idx, spec in enumerate(batch_specs):
        current_job = prepared_job
        next_job: ExportJob | None = None
        if batch_idx + 1 < total_batches:
            next_spec = batch_specs[batch_idx + 1]
            next_job = _launch_export_batch(
                args,
                next_spec,
                slot_dirs[(batch_idx + 1) % len(slot_dirs)],
            )

        try:
            _run_index_batch(args, current_job.spec, current_job.batch_dir)
        except Exception:
            if next_job is not None:
                _stop_export_job(next_job)
            raise

        if not args.keep_batches and current_job.batch_dir.exists():
            shutil.rmtree(current_job.batch_dir)

        if next_job is None:
            prepared_job = None
        else:
            _wait_export_job(next_job)
            prepared_job = next_job

    print(
        f"Completed batched Anserini build: index={args.index_dir} queries={args.query_output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
