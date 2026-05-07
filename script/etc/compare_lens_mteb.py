"""Aggregate per-worker MTEB results and compare against the LENS paper.

The parallel evaluator (``script/evaluate_lens_mteb_parallel.py``) drops
each GPU's results under ``__parallel_workers/worker_N_gpuN/<model_subdir>/``
with one ``{Family}_results.json`` per task family plus a ``summary.json``.
This script:

1. Walks the per-worker dirs, collecting all per-task scores (deduping if a
   task happens to be present in multiple workers' outputs);
2. Aggregates them into family averages and an overall MTEB average using
   the same arithmetic the paper uses (mean over families weighted by
   #datasets; ``overall_average_over_families`` matches Table 1 in
   arXiv:2501.09749);
3. Prints a side-by-side comparison vs the LENS paper's reported numbers
   for LENS-4000 and LENS-8000 (Table 1).

Usage:
    python script/etc/compare_lens_mteb.py <results_dir>
e.g.:
    python script/etc/compare_lens_mteb.py \
      /mnt/ex-disk-1/hyukkyukang/SPLADE/lens/mteb_results/phase1_d4000_LR1e5_20260429_1005
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


# Paper-reported numbers from arXiv:2501.09749 Table 1 (MTEB 56-task suite).
PAPER_TABLE1 = {
    "LENS-4000": {
        "Retrieval":           (60.76, 15),
        "Reranking":           (60.86,  4),
        "Clustering":          (57.92, 11),
        "PairClassification":  (87.93,  3),
        "Classification":      (88.13, 12),
        "STS":                 (84.35, 10),
        "Summarization":       (31.56,  1),
        "Average":             (71.22, 56),
    },
    "LENS-8000": {
        "Retrieval":           (61.86, 15),
        "Reranking":           (60.91,  4),
        "Clustering":          (58.02, 11),
        "PairClassification":  (87.98,  3),
        "Classification":      (88.43, 12),
        "STS":                 (84.67, 10),
        "Summarization":       (29.54,  1),
        "Average":             (71.63, 56),
    },
    "BGE-en-ICL (zero-shot)": {
        "Retrieval":           (61.67, 15),
        "Reranking":           (59.66,  4),
        "Clustering":          (57.51, 11),
        "PairClassification":  (86.93,  3),
        "Classification":      (88.62, 12),
        "STS":                 (83.74, 10),
        "Summarization":       (30.75,  1),
        "Average":             (71.24, 56),
    },
}

# Mapping of MTEB task -> family. Families and counts mirror the paper
# (Retrieval 15, Reranking 4, Clustering 11, PairClassification 3,
# Classification 12, STS 10, Summarization 1; total 56).
TASK_FAMILY = {
    # Retrieval (15) — paper uses BEIR subset; the parallel runner runs the
    # 26 retrieval tasks but Table 1 averages over the 15-task BEIR subset.
    "Retrieval-BEIR-15": {
        "ArguAna", "ClimateFEVER", "CQADupstackRetrieval",  # CQA = avg of 12
        "DBPedia", "FEVER", "FiQA2018", "HotpotQA",
        "MSMARCO", "NFCorpus", "NQ", "QuoraRetrieval",
        "SCIDOCS", "SciFact", "Touche2020", "TRECCOVID",
    },
    "CQA-12": {
        "CQADupstackAndroidRetrieval", "CQADupstackEnglishRetrieval",
        "CQADupstackGamingRetrieval", "CQADupstackGisRetrieval",
        "CQADupstackMathematicaRetrieval", "CQADupstackPhysicsRetrieval",
        "CQADupstackProgrammersRetrieval", "CQADupstackStatsRetrieval",
        "CQADupstackTexRetrieval", "CQADupstackUnixRetrieval",
        "CQADupstackWebmastersRetrieval", "CQADupstackWordpressRetrieval",
    },
    "Reranking": {
        "AskUbuntuDupQuestions", "MindSmallReranking",
        "SciDocsRR", "StackOverflowDupQuestions",
    },
    "Clustering": {
        "ArxivClusteringP2P", "ArxivClusteringS2S",
        "BiorxivClusteringP2P", "BiorxivClusteringS2S",
        "MedrxivClusteringP2P", "MedrxivClusteringS2S",
        "RedditClustering", "RedditClusteringP2P",
        "StackExchangeClustering", "StackExchangeClusteringP2P",
        "TwentyNewsgroupsClustering",
    },
    "PairClassification": {
        "SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus",
    },
    "Classification": {
        "AmazonCounterfactualClassification", "AmazonPolarityClassification",
        "AmazonReviewsClassification", "Banking77Classification",
        "EmotionClassification", "ImdbClassification",
        "MassiveIntentClassification", "MassiveScenarioClassification",
        "MTOPDomainClassification", "MTOPIntentClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    },
    "STS": {
        "BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16",
        "STS17", "STS22", "STSBenchmark",
    },
    "Summarization": {"SummEval"},
}


def _flatten_per_task(results_dir: Path) -> dict[str, float]:
    """Walk every worker dir and collect {task_name: score}."""
    scores: dict[str, float] = {}
    workers_root = results_dir / "__parallel_workers"
    if not workers_root.is_dir():
        raise SystemExit(f"no __parallel_workers/ under {results_dir}")
    for worker_dir in sorted(workers_root.iterdir()):
        # results live one subdir deeper, named after the model path
        for model_dir in worker_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for fam_json in model_dir.glob("*_results.json"):
                try:
                    payload = json.load(fam_json.open())
                except (OSError, json.JSONDecodeError):
                    continue
                tasks = payload.get("tasks") or {}
                for name, score in tasks.items():
                    if isinstance(score, (int, float)):
                        # last-write-wins is fine; tasks should be assigned
                        # to a single worker by the partitioner.
                        scores[name] = float(score)
    return scores


def _aggregate(scores: dict[str, float]) -> dict:
    """Group flat task scores into families + the paper-style average."""
    family_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)

    # CQA-12 average is one entry inside the 15-task Retrieval-BEIR average,
    # so handle CQA first.
    cqa_present = [
        (t, scores[t]) for t in TASK_FAMILY["CQA-12"] if t in scores
    ]
    cqa_avg = mean(s for _, s in cqa_present) if cqa_present else None

    # Retrieval-BEIR uses CQA (single avg) + 14 standalone tasks for 15 total.
    beir_simple = [
        (t, scores[t]) for t in TASK_FAMILY["Retrieval-BEIR-15"]
        if t != "CQADupstackRetrieval" and t in scores
    ]
    if cqa_avg is not None:
        beir_simple.append(("CQADupstackRetrieval (avg of 12)", cqa_avg))
    family_scores["Retrieval"] = beir_simple

    for fam in (
        "Reranking", "Clustering", "PairClassification",
        "Classification", "STS", "Summarization",
    ):
        family_scores[fam] = [
            (t, scores[t]) for t in TASK_FAMILY[fam] if t in scores
        ]

    family_avg: dict[str, tuple[float | None, int, int]] = {}
    # value = (avg, present_count, expected_count)
    expected = {
        "Retrieval":          15,
        "Reranking":           4,
        "Clustering":         11,
        "PairClassification":  3,
        "Classification":     12,
        "STS":                10,
        "Summarization":       1,
    }
    for fam, present in family_scores.items():
        if present:
            family_avg[fam] = (
                mean(s for _, s in present), len(present), expected[fam]
            )
        else:
            family_avg[fam] = (None, 0, expected[fam])

    # Paper "overall" = mean of family averages (each family weighted equally).
    fam_values = [v for v, *_ in family_avg.values() if v is not None]
    overall = mean(fam_values) if fam_values else None
    return {
        "per_task": scores,
        "family_scores": family_scores,
        "family_avg": family_avg,
        "overall_average_over_families": overall,
        "cqa_avg": cqa_avg,
    }


def _fmt(v: float | None) -> str:
    return f"{v*100:.2f}" if (v is not None and v <= 1.0) else (
        f"{v:.2f}" if v is not None else "  n/a")


def _print_report(agg: dict, paper_target: str = "LENS-4000") -> None:
    paper = PAPER_TABLE1[paper_target]
    print(f"\n=== Comparison vs paper Table 1 ({paper_target}) ===\n")
    header = f"{'Family':<22} {'Ours':>8} {'Paper':>8} {'Δ':>8} {'#tasks (ours/paper)':>22}"
    print(header)
    print("-" * len(header))
    fam_order = ("Retrieval", "Reranking", "Clustering",
                 "PairClassification", "Classification", "STS", "Summarization")
    for fam in fam_order:
        ours, n_ours, n_expected = agg["family_avg"][fam]
        paper_score, paper_n = paper[fam]
        delta = (ours * 100 - paper_score) if ours is not None else None
        print(
            f"{fam:<22} {_fmt(ours):>8} {paper_score:>8.2f} "
            f"{(f'{delta:+.2f}' if delta is not None else '   n/a'):>8} "
            f"{f'{n_ours}/{n_expected}':>22}"
        )

    overall = agg["overall_average_over_families"]
    paper_overall, paper_n = paper["Average"]
    delta_overall = (overall * 100 - paper_overall) if overall is not None else None
    print("-" * len(header))
    print(
        f"{'Overall':<22} {_fmt(overall):>8} {paper_overall:>8.2f} "
        f"{(f'{delta_overall:+.2f}' if delta_overall is not None else '   n/a'):>8} "
        f"{'(family-avg)':>22}"
    )

    # Coverage warning
    missing_fams = [
        f for f, (v, n, exp) in agg["family_avg"].items() if n < exp
    ]
    if missing_fams:
        print(
            "\n[!] Family coverage is incomplete; numbers above use only the "
            "tasks that completed:")
        for f in missing_fams:
            v, n, exp = agg["family_avg"][f]
            present = {t for t, _ in agg["family_scores"][f]}
            expected_set = TASK_FAMILY[f] if f != "Retrieval" else set()
            if f == "Retrieval":
                expected_set = (
                    TASK_FAMILY["Retrieval-BEIR-15"]
                    | {"CQADupstackRetrieval (avg of 12)"}
                )
            missing = expected_set - present
            print(f"    {f}: {n}/{exp} present  missing={sorted(missing)[:6]}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    results_dir = Path(sys.argv[1]).resolve()
    paper_target = sys.argv[2] if len(sys.argv) > 2 else "LENS-4000"

    flat = _flatten_per_task(results_dir)
    print(f"Loaded {len(flat)} per-task scores from {results_dir}")

    agg = _aggregate(flat)

    print(f"\n=== Per-family breakdown (ours) ===")
    for fam, items in agg["family_scores"].items():
        if not items:
            continue
        avg, n, exp = agg["family_avg"][fam]
        print(f"\n{fam} ({n}/{exp}, avg={_fmt(avg)}):")
        for t, s in sorted(items):
            print(f"    {_fmt(s):>6}  {t}")

    _print_report(agg, paper_target=paper_target)

    # Also write the aggregate summary back to disk for downstream use.
    out = results_dir / "summary.json"
    payload = {
        "n_tasks": len(flat),
        "per_task": flat,
        "family_avg": {
            f: {"avg": v, "present": n, "expected": exp}
            for f, (v, n, exp) in agg["family_avg"].items()
        },
        "overall_average_over_families": agg["overall_average_over_families"],
        "compared_against": paper_target,
        "paper_table1": PAPER_TABLE1[paper_target],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
