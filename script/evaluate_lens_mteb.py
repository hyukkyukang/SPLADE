"""MTEB English 56-task evaluation for LENS checkpoints.

Mirrors the official ``Yibin-Lei/LENS/eval/mteb_eval.py`` entrypoint:
- iterates the 7 task families (Classification, Clustering, PairClassification,
  Reranking, Retrieval, STS, Summarization);
- passes per-task instruction via :func:`src.utils.lens_mteb_instructions.task_to_instruction`;
- saves per-family average JSON under ``--output-dir/<model_id>/<family>_results.json``.

Supports two invocation modes:

- Bare CLI::
      python script/evaluate_lens_mteb.py --model_name_or_path yibinlei/LENS-d8000
- Task subset (used by the parallel runner)::
      python script/evaluate_lens_mteb.py --model_name_or_path yibinlei/LENS-d8000 \\
          --tasks MSMARCO NFCorpus --output_dir .../worker0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_lens_mteb")


TASK_LIST_CLASSIFICATION: List[str] = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING: List[str] = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION: List[str] = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING: List[str] = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL: List[str] = [
    "NFCorpus",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
    "ArguAna",
    "FiQA2018",
    "SCIDOCS",
    "QuoraRetrieval",
    "DBPedia",
    "ClimateFEVER",
    "FEVER",
    "HotpotQA",
    "MSMARCO",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "NQ",
]

TASK_LIST_STS: List[str] = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
]

TASK_LIST_SUMMARIZATION: List[str] = ["SummEval"]


ALL_TASK_FAMILIES: List[tuple[str, List[str]]] = [
    ("STS", TASK_LIST_STS),
    ("Summarization", TASK_LIST_SUMMARIZATION),
    ("PairClassification", TASK_LIST_PAIR_CLASSIFICATION),
    ("Reranking", TASK_LIST_RERANKING),
    ("Retrieval", TASK_LIST_RETRIEVAL),
    ("Clustering", TASK_LIST_CLUSTERING),
    ("Classification", TASK_LIST_CLASSIFICATION),
]


def _family_for_task(task_name: str) -> str:
    for family_name, task_list in ALL_TASK_FAMILIES:
        if task_name in task_list:
            return family_name
    return "Other"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True,
                   help="HF repo id or local dir, e.g. yibinlei/LENS-d8000.")
    p.add_argument("--lang", default="eng",
                   help="ISO 639-3 language code (MTEB 2.x).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--device", default=None,
                   help="Target device (e.g. cuda:0). Default: auto.")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Subset of task names to run; omit for full 56-task suite.")
    p.add_argument("--families", nargs="*", default=None,
                   help="Subset of families to run (STS, Retrieval, etc.).")
    p.add_argument("--output_dir", default="./mteb_results",
                   help="Root output directory.")
    p.add_argument("--normalize_embeddings", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=True,
                   help="Log per-task metrics to MLflow.")
    p.add_argument("--mlflow_tracking_uri",
                   default=os.environ.get("MLFLOW_TRACKING_URI",
                                          "https://mlflow.hyukkyu.com"))
    p.add_argument("--mlflow_experiment_name", default="Eval-LENS-MTEB")
    p.add_argument("--mlflow_run_name", default=None,
                   help="MLflow run name; default = <model>-<task_subset>-<ts>")
    return p.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32}[name]


def _resolve_tasks_to_run(args: argparse.Namespace) -> List[tuple[str, List[str]]]:
    if args.tasks:
        grouped: dict[str, List[str]] = defaultdict(list)
        for task in args.tasks:
            grouped[_family_for_task(task)].append(task)
        return [(family, tasks) for family, tasks in grouped.items()]
    if args.families:
        families = {f: list(tasks) for f, tasks in ALL_TASK_FAMILIES}
        return [(f, families[f]) for f in args.families if f in families]
    return ALL_TASK_FAMILIES


def _save_family_results(
    family: str,
    task_scores: dict[str, float],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    avg = (sum(task_scores.values()) / len(task_scores)) if task_scores else 0.0
    payload = {"tasks": task_scores, "average": avg}
    out_file = out_dir / f"{family}_results.json"
    with out_file.open("w") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
    logger.info("%s avg=%.4f (tasks=%d) -> %s", family, avg, len(task_scores), out_file)


def _maybe_init_mlflow(args: argparse.Namespace) -> "mlflow.ActiveRun | None":
    if not bool(getattr(args, "mlflow", False)):
        return None
    try:
        import mlflow  # noqa: F401
    except Exception:
        logger.warning("mlflow not installed; skipping MLflow logging.")
        return None
    import mlflow
    try:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        import time
        run_name: str = args.mlflow_run_name or (
            f"{args.model_name_or_path.split('/')[-1]}-"
            f"{'subset' if args.tasks else 'full'}-{int(time.time())}"
        )
        active_run = mlflow.start_run(run_name=run_name)
        # Log params (stringify lists for MLflow).
        params = {
            "model_name_or_path": args.model_name_or_path,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "normalize_embeddings": bool(args.normalize_embeddings),
            "tasks_subset": ",".join(args.tasks or []) or "all",
        }
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.set_tags({
            "benchmark": "lens_mteb_english",
            "eval_type": "phase0_lens_reproduction",
        })
        logger.info("MLflow run started: %s (experiment=%s)", run_name,
                    args.mlflow_experiment_name)
        return active_run
    except Exception as exc:
        logger.warning("MLflow init failed (%s); continuing without MLflow.", exc)
        return None


def _mlflow_log_task_score(task: str, score: float, family: str) -> None:
    try:
        import mlflow
        # MLflow metric names restrict chars. Keep alnum + underscore.
        safe_task = "".join(c if c.isalnum() else "_" for c in task)
        mlflow.log_metric(f"{family}/{safe_task}/main_score", float(score))
    except Exception:
        pass


def _mlflow_log_family_average(family: str, avg: float, n: int) -> None:
    try:
        import mlflow
        mlflow.log_metric(f"avg/{family}", float(avg))
        mlflow.log_metric(f"count/{family}", float(n))
    except Exception:
        pass


def _mlflow_log_overall(overall: float, n_families: int) -> None:
    try:
        import mlflow
        mlflow.log_metric("avg/overall_over_families", float(overall))
        mlflow.log_metric("count/families", float(n_families))
    except Exception:
        pass


def _mlflow_finish(run: "mlflow.ActiveRun | None", status: str = "FINISHED") -> None:
    if run is None:
        return
    try:
        import mlflow
        mlflow.end_run(status=status)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    from src.utils.lens_mteb_encoder import LENSMTEBEncoder

    import mteb  # noqa: E402

    mlflow_run = _maybe_init_mlflow(args)

    dtype = _resolve_dtype(args.dtype)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = LENSMTEBEncoder(
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        normalize_embeddings=args.normalize_embeddings,
        device=device,
    )

    model_id_raw: str = args.model_name_or_path.replace("/", "_").replace(":", "_")
    root_out_dir: Path = Path(args.output_dir) / model_id_raw
    root_out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing MTEB results under %s", root_out_dir)

    families_to_run: List[tuple[str, List[str]]] = _resolve_tasks_to_run(args)
    total_tasks: int = sum(len(task_list) for _, task_list in families_to_run)
    logger.info("Running %d tasks across %d families", total_tasks,
                len(families_to_run))

    overwrite_strategy: str = "always" if bool(args.overwrite) else "only-missing"
    overall_scores: dict[str, dict[str, float]] = {}
    for family, task_list in families_to_run:
        family_scores: dict[str, float] = {}
        for task in task_list:
            logger.info("[%s] Running task: %s", family, task)
            try:
                task_objs = list(mteb.get_tasks(
                    tasks=[task],
                    languages=[args.lang],
                ))
            except Exception as exc:
                logger.exception("get_tasks failed for %s: %s", task, exc)
                continue
            if not task_objs:
                logger.warning("No MTEB task object resolved for %r", task)
                continue
            eval_splits: Iterable[str] = (
                ["dev"] if task == "MSMARCO" else ["test"]
            )
            task_obj = task_objs[0]
            # Pin eval split via the task metadata when supported.
            try:
                if hasattr(task_obj.metadata, "eval_splits"):
                    object.__setattr__(
                        task_obj.metadata, "eval_splits", list(eval_splits)
                    )
            except Exception:
                pass
            encode_kwargs: dict[str, object] = {
                "batch_size": args.batch_size,
                "show_progress_bar": True,
            }
            try:
                result = mteb.evaluate(
                    model,
                    task_obj,
                    encode_kwargs=encode_kwargs,
                    overwrite_strategy=overwrite_strategy,
                    show_progress_bar=True,
                )
            except Exception as exc:
                logger.exception("Task %s failed: %s", task, exc)
                continue
            main_score: float = 0.0
            try:
                for task_result in result.task_results:
                    if str(task_result.task_name) == task:
                        main_score = float(task_result.main_score)
                        break
                else:
                    main_score = float(result.task_results[0].main_score)
            except Exception as exc:
                logger.exception("Could not parse score for task %s: %s", task, exc)
                continue
            family_scores[task] = main_score
            logger.info("[%s] %s main_score=%.4f", family, task, main_score)
            _mlflow_log_task_score(task, main_score, family)
        if family_scores:
            _save_family_results(family, family_scores, root_out_dir)
            overall_scores[family] = family_scores
            avg_fam = sum(family_scores.values()) / len(family_scores)
            _mlflow_log_family_average(family, avg_fam, len(family_scores))

    # Global summary.
    summary = {
        family: {
            "tasks": task_scores,
            "average": (sum(task_scores.values()) / len(task_scores))
                       if task_scores else 0.0,
        }
        for family, task_scores in overall_scores.items()
    }
    overall_avg = 0.0
    family_avgs: List[float] = []
    for payload in summary.values():
        family_avgs.append(payload["average"])
    if family_avgs:
        overall_avg = sum(family_avgs) / len(family_avgs)
    summary["overall_average_over_families"] = overall_avg
    with (root_out_dir / "summary.json").open("w") as fout:
        json.dump(summary, fout, indent=2, sort_keys=True)
    logger.info("Overall average over %d families: %.4f",
                len(overall_scores), overall_avg)
    _mlflow_log_overall(overall_avg, len(overall_scores))
    # Log summary JSON as artifact if we can.
    try:
        import mlflow
        mlflow.log_artifact(str(root_out_dir / "summary.json"))
    except Exception:
        pass
    _mlflow_finish(mlflow_run, status="FINISHED")


if __name__ == "__main__":
    main()
