#!/usr/bin/env python3
"""Re-analyze correlations by replacing normal ANNA with pruned-hangul ANNA.

Notes:
- Uses 2k distribution features from analysis_step1_4.
- Uses NanoMSMARCO proxy retrieval metrics for all compared models so targets
  are available for the pruned model as well.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _safe_div(numer: float, denom: float) -> float:
    if denom == 0.0:
        return math.nan
    return numer / denom


def _rankdata(values: list[float]) -> list[float]:
    """Average-tie rankdata (1-indexed)."""
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks: list[float] = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) == 0:
        return math.nan
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denom_x = math.sqrt(sum(v * v for v in dx))
    denom_y = math.sqrt(sum(v * v for v in dy))
    if denom_x == 0 or denom_y == 0:
        return math.nan
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / (denom_x * denom_y)


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _extract_pair_rows(analysis_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for pair in analysis_payload.get("pairs", []):
        pair_id = str(pair["id"])
        features = dict(pair.get("features", {}))
        if "nonzero_activation_retention" not in features:
            features["nonzero_activation_retention"] = _safe_div(
                float(features["finetuned_nonzero_activation_mean"]),
                float(features["backbone_nonzero_activation_mean"]),
            )
        rows[pair_id] = {
            "id": pair_id,
            "backbone": pair.get("backbone"),
            "finetuned": pair.get("finetuned"),
            "features": features,
            "proxy_ndcg10": pair.get("train_metrics", {}).get("nano_msmarco_ndcg10"),
            "proxy_mrr10": pair.get("train_metrics", {}).get("nano_msmarco_mrr10"),
            "full_ndcg10": pair.get("ndcg10"),
            "full_mrr10": pair.get("mrr10"),
        }
    return rows


def _corr_table(rows: list[dict[str, Any]], target_key: str) -> list[dict[str, float | str]]:
    target = [float(row[target_key]) for row in rows]
    feature_keys = sorted(set().union(*(row["features"].keys() for row in rows)))
    out: list[dict[str, float | str]] = []
    for key in feature_keys:
        if any(key not in row["features"] for row in rows):
            continue
        values = [float(row["features"][key]) for row in rows]
        pearson = _pearson(values, target)
        spearman = _spearman(values, target)
        if math.isnan(pearson) or math.isnan(spearman):
            continue
        out.append({"feature": key, "pearson": pearson, "spearman": spearman})
    out.sort(key=lambda item: abs(float(item["pearson"])), reverse=True)
    return out


def _fmt_float(value: float | None, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def _selected_feature_delta(
    old_corr: list[dict[str, float | str]],
    new_corr: list[dict[str, float | str]],
    feature_names: list[str],
) -> list[dict[str, float | str]]:
    old_map = {str(item["feature"]): item for item in old_corr}
    new_map = {str(item["feature"]): item for item in new_corr}
    out: list[dict[str, float | str]] = []
    for feature in feature_names:
        old_item = old_map.get(feature)
        new_item = new_map.get(feature)
        if old_item is None or new_item is None:
            continue
        old_pearson = float(old_item["pearson"])
        new_pearson = float(new_item["pearson"])
        out.append(
            {
                "feature": feature,
                "old_pearson": old_pearson,
                "new_pearson": new_pearson,
                "delta_pearson": new_pearson - old_pearson,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-analyze correlations with pruned-hangul ANNA.")
    parser.add_argument(
        "--analysis_json",
        type=Path,
        default=Path("script/experiment/output/analysis_step1_4/analysis_2k.json"),
    )
    parser.add_argument(
        "--pruned_addendum_json",
        type=Path,
        default=Path("script/experiment/output/analysis_pruned_hangul/analysis_pruned_hangul.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("script/experiment/output/analysis_pruned_substitution"),
    )
    args = parser.parse_args()

    analysis_payload = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    pruned_payload = json.loads(args.pruned_addendum_json.read_text(encoding="utf-8"))
    base_rows = _extract_pair_rows(analysis_payload)

    old_ids = ["splade_v2", "splade_v2_pp", "splade_v2_pp_anna", "splade_v2_large"]
    old_rows = [base_rows[row_id] for row_id in old_ids]

    pruned_features = dict(pruned_payload["pruned_hangul_pair"]["features"])
    if "nonzero_activation_retention" not in pruned_features:
        pruned_features["nonzero_activation_retention"] = _safe_div(
            float(pruned_features["finetuned_nonzero_activation_mean"]),
            float(pruned_features["backbone_nonzero_activation_mean"]),
        )

    pruned_row = {
        "id": "splade_v2_pp_anna_pruned",
        "backbone": pruned_payload["pruned_hangul_pair"]["backbone"],
        "finetuned": pruned_payload["pruned_hangul_pair"]["finetuned"],
        "features": pruned_features,
        "proxy_ndcg10": pruned_payload["pruned_hangul_pair"]["train_metrics"].get("nano_msmarco_ndcg10"),
        "proxy_mrr10": pruned_payload["pruned_hangul_pair"]["train_metrics"].get("nano_msmarco_mrr10"),
        "full_ndcg10": pruned_payload["pruned_hangul_pair"].get("ndcg10_full_eval"),
        "full_mrr10": pruned_payload["pruned_hangul_pair"].get("mrr10_full_eval"),
    }

    new_ids = ["splade_v2", "splade_v2_pp", "splade_v2_large"]
    new_rows = [base_rows[row_id] for row_id in new_ids] + [pruned_row]

    old_corr_ndcg = _corr_table(old_rows, "proxy_ndcg10")
    old_corr_mrr = _corr_table(old_rows, "proxy_mrr10")
    new_corr_ndcg = _corr_table(new_rows, "proxy_ndcg10")
    new_corr_mrr = _corr_table(new_rows, "proxy_mrr10")

    tracked_features = [
        "delta_survival_frac",
        "survival_retention",
        "l1_retention",
        "l2_retention",
        "delta_q99",
        "delta_nonzero_activation_mean",
        "nonzero_activation_retention",
    ]
    ndcg_shift = _selected_feature_delta(old_corr_ndcg, new_corr_ndcg, tracked_features)
    mrr_shift = _selected_feature_delta(old_corr_mrr, new_corr_mrr, tracked_features)

    output_payload: dict[str, Any] = {
        "inputs": {
            "analysis_json": str(args.analysis_json),
            "pruned_addendum_json": str(args.pruned_addendum_json),
            "note": "Uses NanoMSMARCO proxy metrics for all rows.",
        },
        "old_set": {
            "row_ids": old_ids,
            "rows": old_rows,
            "proxy_correlations": {
                "ndcg10": old_corr_ndcg,
                "mrr10": old_corr_mrr,
            },
        },
        "new_set_pruned_substitution": {
            "row_ids": [*new_ids, "splade_v2_pp_anna_pruned"],
            "rows": new_rows,
            "proxy_correlations": {
                "ndcg10": new_corr_ndcg,
                "mrr10": new_corr_mrr,
            },
        },
        "correlation_shift": {
            "proxy_ndcg10": ndcg_shift,
            "proxy_mrr10": mrr_shift,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "reanalyze_pruned_substitution.json"
    json_path.write_text(json.dumps(output_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    md_lines: list[str] = []
    md_lines.append("# Reanalysis With Pruned-Hangul ANNA (Proxy Retrieval Targets)")
    md_lines.append("")
    md_lines.append("## Setup")
    md_lines.append("- Replace `splade_v2_pp_anna` with `splade_v2_pp_anna_pruned` in the 4-model set.")
    md_lines.append("- Keep the same 2k distribution features.")
    md_lines.append(
        "- Use `NanoMSMARCO` proxy retrieval metrics (`ndcg@10`, `mrr@10`) for all models."
    )
    md_lines.append(
        "- Full MSMARCO dev-full metrics are still unavailable for the pruned-hangul checkpoint in this analysis."
    )
    md_lines.append("")

    md_lines.append("## Substituted 4-Model Table")
    md_lines.append("")
    md_lines.append(
        "| Model | Proxy nDCG@10 | Proxy MRR@10 | Survival Retention | L1 Retention | L2 Retention | Nonzero Retention | Delta q99 |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in new_rows:
        f = row["features"]
        md_lines.append(
            f"| `{row['id']}` | "
            f"{_fmt_float(float(row['proxy_ndcg10']))} | "
            f"{_fmt_float(float(row['proxy_mrr10']))} | "
            f"{_fmt_float(float(f['survival_retention']))} | "
            f"{_fmt_float(float(f['l1_retention']))} | "
            f"{_fmt_float(float(f['l2_retention']))} | "
            f"{_fmt_float(float(f['nonzero_activation_retention']))} | "
            f"{_fmt_float(float(f['delta_q99']))} |"
        )
    md_lines.append("")

    md_lines.append("## Top Correlations vs Proxy nDCG@10 (Pruned Substitution)")
    md_lines.append("")
    md_lines.append("| Feature | Pearson | Spearman |")
    md_lines.append("|---|---:|---:|")
    for item in new_corr_ndcg[:12]:
        md_lines.append(
            f"| `{item['feature']}` | "
            f"{_fmt_float(float(item['pearson']), precision=4)} | "
            f"{_fmt_float(float(item['spearman']), precision=4)} |"
        )
    md_lines.append("")

    md_lines.append("## Correlation Shift (Old -> New) vs Proxy nDCG@10")
    md_lines.append("")
    md_lines.append("| Feature | Old Pearson | New Pearson | Delta |")
    md_lines.append("|---|---:|---:|---:|")
    for item in ndcg_shift:
        md_lines.append(
            f"| `{item['feature']}` | "
            f"{_fmt_float(float(item['old_pearson']), precision=4)} | "
            f"{_fmt_float(float(item['new_pearson']), precision=4)} | "
            f"{_fmt_float(float(item['delta_pearson']), precision=4)} |"
        )
    md_lines.append("")

    md_lines.append("## Interpretation")
    md_lines.append(
        "- After substitution, the pruned ANNA run has the best proxy retrieval but also the lowest retention values."
    )
    md_lines.append(
        "- This breaks the earlier simple rule that \"higher retention => better retrieval\" when using this substituted set."
    )
    md_lines.append(
        "- Practical read: vocabulary-size normalization alone does not explain ANNA behavior; training regime and regularization schedule dominate."
    )

    md_path = args.output_dir / "reanalyze_pruned_substitution.md"
    md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
