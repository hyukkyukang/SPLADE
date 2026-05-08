"""Upload a LENS MTEB run's JSON results to MLflow retroactively.

Usage::

    python script/etc/lens_mteb_results_to_mlflow.py \\
        --results_dir /mnt/ex-disk-1/.../lens/mteb_results/yibinlei_LENS-d8000 \\
        --model_name yibinlei/LENS-d8000 \\
        --run_name phase0_d8000_sdpa_bs64

Reads ``summary.json`` and/or per-family ``*_results.json`` from ``--results_dir``
and logs all task-level metrics + per-family averages + overall average to a
single MLflow run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                   help="Directory holding summary.json and/or <Family>_results.json")
    p.add_argument("--model_name", required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument("--experiment_name", default="Eval-LENS-MTEB")
    p.add_argument("--tracking_uri",
                   default=os.environ.get("MLFLOW_TRACKING_URI",
                                          "https://mlflow.hyukkyu.com"))
    p.add_argument("--extra_tags", default=None,
                   help="JSON dict of extra tags, e.g. '{\"phase\": \"0\"}'")
    return p.parse_args()


def _load_combined(results_dir: Path) -> Dict[str, Dict[str, float]]:
    combined: Dict[str, Dict[str, float]] = {}
    summary_file = results_dir / "summary.json"
    if summary_file.is_file():
        payload = json.loads(summary_file.read_text())
        for family, bundle in payload.items():
            if family == "overall_average_over_families":
                continue
            if not isinstance(bundle, dict):
                continue
            tasks = bundle.get("tasks", {})
            if isinstance(tasks, dict) and tasks:
                combined[family] = {k: float(v) for k, v in tasks.items()}
    # Merge per-family JSONs too (override summary in case of inconsistency).
    for family_json in results_dir.glob("*_results.json"):
        family = family_json.stem.replace("_results", "")
        payload = json.loads(family_json.read_text())
        tasks = payload.get("tasks", {})
        if isinstance(tasks, dict) and tasks:
            combined[family] = {k: float(v) for k, v in tasks.items()}
    return combined


def main() -> None:
    args = parse_args()
    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or (
        f"{args.model_name.split('/')[-1]}-retro-{int(time.time())}"
    )

    results_dir = Path(args.results_dir).expanduser()
    combined = _load_combined(results_dir)
    if not combined:
        raise SystemExit(f"No results found in {results_dir}")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "benchmark": "lens_mteb_english",
            "eval_type": "phase0_lens_reproduction_retro",
            "model_name": args.model_name,
        })
        if args.extra_tags:
            try:
                mlflow.set_tags(json.loads(args.extra_tags))
            except Exception:
                pass
        mlflow.log_params({
            "model_name_or_path": args.model_name,
            "results_dir": str(results_dir),
        })

        family_avgs = []
        for family, tasks in sorted(combined.items()):
            for task, score in tasks.items():
                safe = "".join(c if c.isalnum() else "_" for c in task)
                mlflow.log_metric(f"{family}/{safe}/main_score", float(score))
            avg = sum(tasks.values()) / len(tasks)
            mlflow.log_metric(f"avg/{family}", float(avg))
            mlflow.log_metric(f"count/{family}", float(len(tasks)))
            family_avgs.append(avg)
            print(f"{family}: avg={avg:.4f}  tasks={len(tasks)}")

        overall = sum(family_avgs) / len(family_avgs)
        mlflow.log_metric("avg/overall_over_families", float(overall))
        mlflow.log_metric("count/families", float(len(combined)))
        print(f"Overall avg over {len(combined)} families: {overall:.4f}")

        # Artifacts
        if (results_dir / "summary.json").is_file():
            mlflow.log_artifact(str(results_dir / "summary.json"))
        for f in results_dir.glob("*_results.json"):
            mlflow.log_artifact(str(f))


if __name__ == "__main__":
    main()
