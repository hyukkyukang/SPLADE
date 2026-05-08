"""Fan out the LENS MTEB 56-task suite across available GPUs.

Each GPU runs a subset of tasks via ``script/evaluate_lens_mteb.py`` with
``CUDA_VISIBLE_DEVICES`` pinned. Tasks are partitioned by estimated size so
the biggest (MSMARCO, DBPedia, NQ, FEVER, HotpotQA) land on separate GPUs.

Typical wall-clock on 8×A100-40GB: ~3–4 h per model for full 56 tasks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("evaluate_lens_mteb_parallel")


# --- Task list pulled from the official LENS repo (56 English tasks) ---
TASKS: List[str] = [
    # Retrieval (26)
    "NFCorpus", "SciFact", "Touche2020", "TRECCOVID", "ArguAna",
    "FiQA2018", "SCIDOCS", "QuoraRetrieval", "DBPedia", "ClimateFEVER",
    "FEVER", "HotpotQA", "MSMARCO",
    "CQADupstackAndroidRetrieval", "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval", "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval", "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval", "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval", "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval", "CQADupstackWordpressRetrieval",
    "NQ",
    # Classification (12)
    "AmazonCounterfactualClassification", "AmazonPolarityClassification",
    "AmazonReviewsClassification", "Banking77Classification",
    "EmotionClassification", "ImdbClassification",
    "MassiveIntentClassification", "MassiveScenarioClassification",
    "MTOPDomainClassification", "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    # Clustering (11)
    "ArxivClusteringP2P", "ArxivClusteringS2S",
    "BiorxivClusteringP2P", "BiorxivClusteringS2S",
    "MedrxivClusteringP2P", "MedrxivClusteringS2S",
    "RedditClustering", "RedditClusteringP2P",
    "StackExchangeClustering", "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
    # Reranking (4)
    "AskUbuntuDupQuestions", "MindSmallReranking",
    "SciDocsRR", "StackOverflowDupQuestions",
    # PairClassification (3)
    "SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus",
    # STS (10) + Summarization (1)
    "BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16",
    "STS17", "STS22", "STSBenchmark", "SummEval",
]


# Approximate cost weights (corpus size / sample count) drawn from MTEB docs.
# Larger number = runs longer; used for work-stealing assignment.
_TASK_COST: Dict[str, float] = {
    "MSMARCO": 100.0, "DBPedia": 80.0, "FEVER": 60.0, "HotpotQA": 55.0,
    "NQ": 40.0, "ClimateFEVER": 20.0, "TRECCOVID": 15.0, "QuoraRetrieval": 20.0,
    "FiQA2018": 10.0, "Touche2020": 8.0, "ArguAna": 8.0, "SciFact": 6.0,
    "SCIDOCS": 8.0, "NFCorpus": 3.0,
    # CQA subsets - small
    **{f"CQADupstack{sub}Retrieval": 5.0 for sub in [
        "Android", "English", "Gaming", "Gis", "Mathematica", "Physics",
        "Programmers", "Stats", "Tex", "Unix", "Webmasters", "Wordpress",
    ]},
    # Classification (moderate)
    "AmazonReviewsClassification": 8.0,
    "AmazonPolarityClassification": 10.0,
    "ToxicConversationsClassification": 8.0,
    "ImdbClassification": 6.0, "EmotionClassification": 4.0,
    "AmazonCounterfactualClassification": 4.0,
    "Banking77Classification": 4.0,
    "MassiveIntentClassification": 4.0, "MassiveScenarioClassification": 4.0,
    "MTOPDomainClassification": 4.0, "MTOPIntentClassification": 4.0,
    "TweetSentimentExtractionClassification": 4.0,
    # Clustering (moderate)
    "ArxivClusteringP2P": 10.0, "ArxivClusteringS2S": 6.0,
    "RedditClusteringP2P": 8.0, "RedditClustering": 6.0,
    "StackExchangeClusteringP2P": 6.0, "StackExchangeClustering": 5.0,
    "BiorxivClusteringP2P": 4.0, "BiorxivClusteringS2S": 3.0,
    "MedrxivClusteringP2P": 4.0, "MedrxivClusteringS2S": 3.0,
    "TwentyNewsgroupsClustering": 4.0,
    # Reranking (moderate)
    "MindSmallReranking": 6.0, "StackOverflowDupQuestions": 4.0,
    "AskUbuntuDupQuestions": 3.0, "SciDocsRR": 3.0,
    # PairClassification (small)
    "SprintDuplicateQuestions": 3.0,
    "TwitterSemEval2015": 3.0, "TwitterURLCorpus": 4.0,
    # STS and Summ (very small)
    "BIOSSES": 1.0, "SICK-R": 1.0, "STS12": 1.0, "STS13": 1.0, "STS14": 1.0,
    "STS15": 1.0, "STS16": 1.0, "STS17": 1.0, "STS22": 1.0,
    "STSBenchmark": 1.0, "SummEval": 1.0,
}


def _cost(task: str) -> float:
    return _TASK_COST.get(task, 5.0)


def _greedy_partition(tasks: List[str], num_workers: int) -> List[List[str]]:
    """LPT-like assignment: heaviest tasks first, always to the lightest bin."""
    ordered: List[str] = sorted(tasks, key=_cost, reverse=True)
    bins: List[List[str]] = [[] for _ in range(num_workers)]
    loads: List[float] = [0.0] * num_workers
    for task in ordered:
        min_idx: int = int(min(range(num_workers), key=loads.__getitem__))
        bins[min_idx].append(task)
        loads[min_idx] += _cost(task)
    return bins


def _resolve_gpu_ids(explicit: str | None) -> List[str]:
    if explicit:
        return [g.strip() for g in explicit.split(",") if g.strip()]
    visible: str | None = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [g.strip() for g in visible.split(",") if g.strip()]
    try:
        import torch  # local import to keep this script importable without CUDA
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return ["0"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--output_dir", default=None,
                   help="Default: $LENS_MTEB_RESULTS or ./mteb_results.")
    p.add_argument("--gpu_ids", default=None,
                   help="Comma-separated list; default = CUDA_VISIBLE_DEVICES "
                        "or all detected GPUs.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Subset of tasks to run; default = all 56.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the partition and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gpu_ids: List[str] = _resolve_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        raise RuntimeError("No GPUs available.")

    all_tasks: List[str] = args.tasks if args.tasks else TASKS
    logger.info("Partitioning %d tasks across %d GPU(s): %s",
                len(all_tasks), len(gpu_ids), gpu_ids)
    partitions: List[List[str]] = _greedy_partition(all_tasks, len(gpu_ids))
    for idx, (gpu, bin_tasks) in enumerate(zip(gpu_ids, partitions)):
        load = sum(_cost(t) for t in bin_tasks)
        logger.info("  GPU %s: cost≈%.1f, %d tasks: %s",
                    gpu, load, len(bin_tasks), ", ".join(bin_tasks))

    if args.dry_run:
        return

    output_root: str = (
        args.output_dir
        or os.environ.get("LENS_MTEB_RESULTS")
        or "./mteb_results"
    )
    worker_root_dir: Path = Path(output_root) / "__parallel_workers"
    worker_root_dir.mkdir(parents=True, exist_ok=True)

    script_path: Path = Path(__file__).resolve().parent / "evaluate_lens_mteb.py"
    repo_root: Path = Path(__file__).resolve().parent.parent
    worker_procs: List[Tuple[subprocess.Popen[str], str, List[str], Path]] = []
    for worker_idx, (gpu_id, bin_tasks) in enumerate(zip(gpu_ids, partitions)):
        if not bin_tasks:
            continue
        worker_out: Path = worker_root_dir / f"worker_{worker_idx}_gpu{gpu_id}"
        worker_out.mkdir(parents=True, exist_ok=True)
        log_path: Path = worker_out / "worker.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd: List[str] = [
            sys.executable,
            str(script_path),
            "--model_name_or_path", args.model_name_or_path,
            "--batch_size", str(args.batch_size),
            "--dtype", args.dtype,
            "--attn_implementation", args.attn_implementation,
            "--output_dir", str(worker_out),
            "--tasks", *bin_tasks,
        ]
        logger.info("Spawning worker %d on GPU %s -> %s",
                    worker_idx, gpu_id, log_path)
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log_path.open("w"),
            stderr=subprocess.STDOUT,
            text=True,
        )
        worker_procs.append((proc, gpu_id, bin_tasks, log_path))

    start_time: float = time.time()
    failures: List[Tuple[str, List[str], int, Path]] = []
    for proc, gpu_id, bin_tasks, log_path in worker_procs:
        rc: int = proc.wait()
        if rc != 0:
            failures.append((gpu_id, bin_tasks, rc, log_path))
        logger.info("Worker on GPU %s finished rc=%d (tasks=%d)",
                    gpu_id, rc, len(bin_tasks))

    duration: float = time.time() - start_time
    logger.info("All workers done in %.1f s", duration)
    if failures:
        for gpu_id, bin_tasks, rc, log_path in failures:
            logger.error("GPU %s rc=%d tasks=%s log=%s",
                         gpu_id, rc, bin_tasks, log_path)
        raise SystemExit(2)

    _aggregate_results(
        worker_root_dir=worker_root_dir,
        model_name_or_path=args.model_name_or_path,
        final_output_dir=Path(output_root),
    )


def _aggregate_results(
    *,
    worker_root_dir: Path,
    model_name_or_path: str,
    final_output_dir: Path,
) -> None:
    model_id: str = model_name_or_path.replace("/", "_").replace(":", "_")
    combined: Dict[str, Dict[str, float]] = {}
    for worker_dir in sorted(worker_root_dir.glob("worker_*")):
        nested = worker_dir / model_id
        if not nested.is_dir():
            continue
        for family_json in nested.glob("*_results.json"):
            family: str = family_json.stem.replace("_results", "")
            payload = json.loads(family_json.read_text())
            combined.setdefault(family, {}).update(payload.get("tasks", {}))

    final_dir: Path = final_output_dir / model_id
    final_dir.mkdir(parents=True, exist_ok=True)
    overall_avgs: List[float] = []
    summary: Dict[str, Dict[str, float]] = {}
    for family, tasks in sorted(combined.items()):
        avg = (sum(tasks.values()) / len(tasks)) if tasks else 0.0
        (final_dir / f"{family}_results.json").write_text(
            json.dumps({"tasks": tasks, "average": avg}, indent=2, sort_keys=True)
        )
        summary[family] = {"tasks": tasks, "average": avg}
        overall_avgs.append(avg)
    overall = (sum(overall_avgs) / len(overall_avgs)) if overall_avgs else 0.0
    summary["overall_average_over_families"] = overall
    (final_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    logger.info("Aggregated summary -> %s (overall avg over families: %.4f)",
                final_dir / "summary.json", overall)


if __name__ == "__main__":
    main()
