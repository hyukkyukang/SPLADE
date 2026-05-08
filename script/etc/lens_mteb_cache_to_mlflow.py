"""Scrape MTEB's per-task result cache and upsert the metrics into one MLflow run.

Designed to run repeatedly while a LENS MTEB eval is in flight — each invocation
updates the same MLflow run so you can watch numbers populate live in the UI.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional


_FAMILY_OF_TASK: Dict[str, str] = {}


def _build_family_map() -> Dict[str, str]:
    if _FAMILY_OF_TASK:
        return _FAMILY_OF_TASK
    import sys
    sys.path.insert(0, "/home/user/SPLADE/script")
    from evaluate_lens_mteb import ALL_TASK_FAMILIES  # type: ignore
    for family, tasks in ALL_TASK_FAMILIES:
        for t in tasks:
            _FAMILY_OF_TASK[t] = family
    return _FAMILY_OF_TASK


def _extract_main_score(payload: dict) -> Optional[float]:
    scores = payload.get("scores", {})
    for split_name in ("test", "dev"):
        split = scores.get(split_name)
        if isinstance(split, list) and split:
            rec = split[0]
            if isinstance(rec, dict) and "main_score" in rec:
                try:
                    return float(rec["main_score"])
                except (TypeError, ValueError):
                    continue
        elif isinstance(split, dict) and "main_score" in split:
            try:
                return float(split["main_score"])
            except (TypeError, ValueError):
                continue
    return None


def _find_or_create_run(mlflow, experiment_name: str, run_name: str) -> str:
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment {experiment_name} missing after set_experiment.")
    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )
    if len(df) > 0:
        return str(df.iloc[0]["run_id"])
    with mlflow.start_run(run_name=run_name) as run:
        return run.info.run_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", default="/home/user/.cache/mteb/results")
    p.add_argument("--model_slug", required=True,
                   help="Cache subdir slug, e.g. yibinlei__LENS-d8000")
    p.add_argument("--model_name", required=True,
                   help="Pretty name, e.g. yibinlei/LENS-d8000")
    p.add_argument("--run_name", required=True)
    p.add_argument("--experiment_name", default="Eval-LENS-MTEB")
    p.add_argument("--tracking_uri",
                   default=os.environ.get("MLFLOW_TRACKING_URI",
                                          "https://mlflow.hyukkyu.com"))
    args = p.parse_args()

    import mlflow
    mlflow.set_tracking_uri(args.tracking_uri)
    run_id = _find_or_create_run(mlflow, args.experiment_name, args.run_name)
    family_of_task = _build_family_map()

    cache_dir = Path(args.cache_root) / args.model_slug
    if not cache_dir.exists():
        raise SystemExit(f"No cache dir at {cache_dir}")

    revs = sorted([p for p in cache_dir.iterdir() if p.is_dir()])
    if not revs:
        raise SystemExit(f"No revision subdirs under {cache_dir}")
    tasks_dir = revs[-1]  # latest revision

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({
            "benchmark": "lens_mteb_english",
            "model_name": args.model_name,
            "eval_source": "mteb_cache_scrape",
            "last_sync_ts": str(int(time.time())),
        })
        mlflow.log_param("model_name_or_path", args.model_name)

        family_scores: Dict[str, Dict[str, float]] = {}
        for jf in sorted(tasks_dir.glob("*.json")):
            if jf.stem == "model_meta":
                continue
            try:
                payload = json.loads(jf.read_text())
            except Exception:
                continue
            task = str(payload.get("task_name") or jf.stem)
            score = _extract_main_score(payload)
            if score is None:
                continue
            family = family_of_task.get(task, "Other")
            family_scores.setdefault(family, {})[task] = score
            safe = "".join(c if c.isalnum() else "_" for c in task)
            mlflow.log_metric(f"{family}/{safe}/main_score", float(score))
            print(f"  {family}/{task}: {score:.4f}")

        family_avgs = []
        for family, tasks in sorted(family_scores.items()):
            avg = sum(tasks.values()) / len(tasks)
            mlflow.log_metric(f"avg/{family}", float(avg))
            mlflow.log_metric(f"count/{family}", float(len(tasks)))
            family_avgs.append(avg)
            print(f"{family}: avg={avg:.4f} ({len(tasks)} tasks)")
        if family_avgs:
            overall = sum(family_avgs) / len(family_avgs)
            mlflow.log_metric("avg/overall_over_families", float(overall))
            mlflow.log_metric("count/families", float(len(family_scores)))
            print(f"Overall avg over {len(family_scores)} families: {overall:.4f}")

        print(f"MLflow run: {args.tracking_uri}/#/experiments/.../runs/{run_id}")


if __name__ == "__main__":
    main()
