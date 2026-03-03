#!/usr/bin/env python3
"""Addendum analysis for pruned-hangul ANNA backbone/SPLADE pair.

This script compares:
1) existing normal ANNA pair from analysis_step1_4/analysis_2k.json
2) pruned-hangul ANNA pair using freshly computed logit_stats.json files

Outputs:
- analysis_pruned_hangul.json
- analysis_pruned_hangul.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _safe_div(numer: float, denom: float) -> float:
    if denom == 0:
        return math.nan
    return numer / denom


def _build_features(backbone_payload: dict[str, Any], finetuned_payload: dict[str, Any]) -> dict[str, float]:
    b_stats = backbone_payload["stats"]
    f_stats = finetuned_payload["stats"]
    b_survival = backbone_payload["vocab_survival"]
    f_survival = finetuned_payload["vocab_survival"]

    features: dict[str, float] = {
        "backbone_mean_logit": float(b_stats["mean"]),
        "finetuned_mean_logit": float(f_stats["mean"]),
        "backbone_q95": float(b_stats["quantiles"]["0.95"]),
        "finetuned_q95": float(f_stats["quantiles"]["0.95"]),
        "backbone_q99": float(b_stats["quantiles"]["0.99"]),
        "finetuned_q99": float(f_stats["quantiles"]["0.99"]),
        "backbone_neg_frac": float(b_stats["negative_fraction"]),
        "finetuned_neg_frac": float(f_stats["negative_fraction"]),
        "backbone_survival_frac": float(b_survival["mean_fraction"]),
        "finetuned_survival_frac": float(f_survival["mean_fraction"]),
        "backbone_l1": float(backbone_payload["pooled_activation_l1_stats"]["mean"]),
        "finetuned_l1": float(finetuned_payload["pooled_activation_l1_stats"]["mean"]),
        "backbone_l2": float(backbone_payload["pooled_activation_l2_stats"]["mean"]),
        "finetuned_l2": float(finetuned_payload["pooled_activation_l2_stats"]["mean"]),
        "backbone_max_activation": float(backbone_payload["pooled_activation_max_stats"]["mean"]),
        "finetuned_max_activation": float(finetuned_payload["pooled_activation_max_stats"]["mean"]),
        "backbone_nonzero_activation_mean": float(
            backbone_payload["nonzero_pooled_activation_stats"]["mean"]
        ),
        "finetuned_nonzero_activation_mean": float(
            finetuned_payload["nonzero_pooled_activation_stats"]["mean"]
        ),
        "backbone_nonzero_activation_std": float(
            backbone_payload["nonzero_pooled_activation_stats"]["std"]
        ),
        "finetuned_nonzero_activation_std": float(
            finetuned_payload["nonzero_pooled_activation_stats"]["std"]
        ),
    }

    features["delta_mean_logit"] = features["finetuned_mean_logit"] - features["backbone_mean_logit"]
    features["delta_q95"] = features["finetuned_q95"] - features["backbone_q95"]
    features["delta_q99"] = features["finetuned_q99"] - features["backbone_q99"]
    features["delta_neg_frac"] = features["finetuned_neg_frac"] - features["backbone_neg_frac"]
    features["delta_survival_frac"] = (
        features["finetuned_survival_frac"] - features["backbone_survival_frac"]
    )
    features["survival_retention"] = _safe_div(
        features["finetuned_survival_frac"], features["backbone_survival_frac"]
    )
    features["delta_l1"] = features["finetuned_l1"] - features["backbone_l1"]
    features["l1_retention"] = _safe_div(features["finetuned_l1"], features["backbone_l1"])
    features["delta_l2"] = features["finetuned_l2"] - features["backbone_l2"]
    features["l2_retention"] = _safe_div(features["finetuned_l2"], features["backbone_l2"])
    features["delta_max_activation"] = (
        features["finetuned_max_activation"] - features["backbone_max_activation"]
    )
    features["delta_nonzero_activation_mean"] = (
        features["finetuned_nonzero_activation_mean"] - features["backbone_nonzero_activation_mean"]
    )
    features["delta_nonzero_activation_std"] = (
        features["finetuned_nonzero_activation_std"] - features["backbone_nonzero_activation_std"]
    )
    features["nonzero_activation_retention"] = _safe_div(
        features["finetuned_nonzero_activation_mean"], features["backbone_nonzero_activation_mean"]
    )
    return features


def _extract_best_step_from_checkpoints(checkpoint_dir: Path) -> tuple[int | None, float | None]:
    pattern = re.compile(r"stepstep=(\d+)-val_MRR_10val_MRR_10=([0-9.]+)\.ckpt$")
    best_step: int | None = None
    best_val: float | None = None

    for ckpt_path in checkpoint_dir.glob("*.ckpt"):
        match = pattern.search(ckpt_path.name)
        if match is None:
            continue
        step = int(match.group(1))
        score = float(match.group(2))
        if best_val is None or score > best_val:
            best_step = step
            best_val = score
    return best_step, best_val


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_float(value: float | None, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pruned-hangul ANNA retention vs normal ANNA.")
    parser.add_argument(
        "--analysis_json",
        type=Path,
        default=Path("script/experiment/output/analysis_step1_4/analysis_2k.json"),
        help="Existing analysis_2k.json containing normal ANNA pair.",
    )
    parser.add_argument(
        "--normal_pair_id",
        type=str,
        default="splade_v2_pp_anna",
        help="Pair id for the normal ANNA row in analysis_2k.json.",
    )
    parser.add_argument(
        "--pruned_backbone_stats",
        type=Path,
        default=Path("script/experiment/output/backbone_pruned_2k/logit_stats.json"),
        help="logit_stats.json for pruned-hangul ANNA backbone.",
    )
    parser.add_argument(
        "--pruned_finetuned_stats",
        type=Path,
        default=Path("script/experiment/output/splade_finetuned_pruned_2k/logit_stats.json"),
        help="logit_stats.json for pruned-hangul ANNA fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--pruned_backbone_model",
        type=str,
        default="data/model/trained_anna_base_hf_pruned_hangul",
        help="Model name key inside pruned backbone stats.",
    )
    parser.add_argument(
        "--pruned_finetuned_model",
        type=str,
        default=(
            "/home/user/SPLADE/log/train/splade_v2_pp_anna_trained_pruned_hangul/"
            "rerun_anna_pruned_mlflow_20260225_0115/checkpoints/best.ckpt"
        ),
        help="Model name key inside pruned fine-tuned stats.",
    )
    parser.add_argument(
        "--pruned_log_dir",
        type=Path,
        default=Path(
            "/home/user/SPLADE/log/train/splade_v2_pp_anna_trained_pruned_hangul/"
            "rerun_anna_pruned_mlflow_20260225_0115"
        ),
        help="Run directory for loading nanobeir_metrics_step*.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("script/experiment/output/analysis_pruned_hangul"),
        help="Output directory.",
    )
    args = parser.parse_args()

    analysis_payload = _load_json(args.analysis_json)
    normal_pair = None
    for pair in analysis_payload.get("pairs", []):
        if pair.get("id") == args.normal_pair_id:
            normal_pair = pair
            break
    if normal_pair is None:
        raise ValueError(f"Could not find pair id '{args.normal_pair_id}' in {args.analysis_json}")

    pruned_backbone_payload = _load_json(args.pruned_backbone_stats)["models"][args.pruned_backbone_model]
    pruned_finetuned_payload = _load_json(args.pruned_finetuned_stats)["models"][args.pruned_finetuned_model]
    pruned_features = _build_features(pruned_backbone_payload, pruned_finetuned_payload)

    best_step, best_val_mrr10 = _extract_best_step_from_checkpoints(args.pruned_log_dir / "checkpoints")
    train_metrics: dict[str, Any] = {
        "selected_step": best_step,
        "best_val_mrr10_from_checkpoint_name": best_val_mrr10,
    }
    if best_step is not None:
        metrics_path = args.pruned_log_dir / f"nanobeir_metrics_step{best_step}.json"
        if metrics_path.exists():
            metrics_payload = _load_json(metrics_path)
            train_metrics.update(
                {
                    "source_file": str(metrics_path),
                    "nano_msmarco_avg_flops": metrics_payload.get("NanoMSMARCO_avg_flops"),
                    "nano_msmarco_query_active_dims": metrics_payload.get("NanoMSMARCO_query_active_dims"),
                    "nano_msmarco_corpus_active_dims": metrics_payload.get("NanoMSMARCO_corpus_active_dims"),
                    "nanobeir_mean_avg_flops": metrics_payload.get("NanoBEIR_mean_avg_flops"),
                    "nanobeir_mean_query_active_dims": metrics_payload.get("NanoBEIR_mean_query_active_dims"),
                    "nanobeir_mean_corpus_active_dims": metrics_payload.get("NanoBEIR_mean_corpus_active_dims"),
                    "nano_msmarco_ndcg10": metrics_payload.get("NanoMSMARCO_dot_ndcg@10"),
                    "nano_msmarco_mrr10": metrics_payload.get("NanoMSMARCO_dot_mrr@10"),
                }
            )

    normal_vocab_size: float | None = None
    normal_backbone_payload: dict[str, Any] | None = None
    normal_backbone_stats_ref = analysis_payload.get("inputs", {}).get("backbone_stats")
    if isinstance(normal_backbone_stats_ref, str):
        normal_backbone_stats_path = Path(normal_backbone_stats_ref)
        if normal_backbone_stats_path.exists():
            normal_backbone_all = _load_json(normal_backbone_stats_path).get("models", {})
            normal_backbone_payload = normal_backbone_all.get(normal_pair["backbone"])
            if isinstance(normal_backbone_payload, dict):
                normal_vocab_size = float(normal_backbone_payload["vocab_survival"]["vocab_size"])

    normal_features = normal_pair["features"]
    if "nonzero_activation_retention" not in normal_features:
        normal_features["nonzero_activation_retention"] = _safe_div(
            float(normal_features["finetuned_nonzero_activation_mean"]),
            float(normal_features["backbone_nonzero_activation_mean"]),
        )
    if "backbone_nonzero_activation_std" not in normal_features and isinstance(normal_backbone_payload, dict):
        normal_features["backbone_nonzero_activation_std"] = float(
            normal_backbone_payload["nonzero_pooled_activation_stats"]["std"]
        )
    if "finetuned_nonzero_activation_std" not in normal_features:
        normal_finetuned_stats_ref = analysis_payload.get("inputs", {}).get("finetuned_stats")
        if isinstance(normal_finetuned_stats_ref, str) and Path(normal_finetuned_stats_ref).exists():
            normal_finetuned_all = _load_json(Path(normal_finetuned_stats_ref)).get("models", {})
            normal_finetuned_payload = normal_finetuned_all.get(normal_pair["finetuned"])
            if isinstance(normal_finetuned_payload, dict):
                normal_features["finetuned_nonzero_activation_std"] = float(
                    normal_finetuned_payload["nonzero_pooled_activation_stats"]["std"]
                )

    output_payload: dict[str, Any] = {
        "inputs": {
            "analysis_json": str(args.analysis_json),
            "pruned_backbone_stats": str(args.pruned_backbone_stats),
            "pruned_finetuned_stats": str(args.pruned_finetuned_stats),
            "pruned_backbone_model": args.pruned_backbone_model,
            "pruned_finetuned_model": args.pruned_finetuned_model,
            "pruned_log_dir": str(args.pruned_log_dir),
        },
        "normal_anna_pair": {
            "id": normal_pair["id"],
            "backbone": normal_pair["backbone"],
            "finetuned": normal_pair["finetuned"],
            "mrr10_full_eval": normal_pair.get("mrr10"),
            "ndcg10_full_eval": normal_pair.get("ndcg10"),
            "features": normal_features,
        },
        "pruned_hangul_pair": {
            "id": "splade_v2_pp_anna_pruned_hangul",
            "backbone": args.pruned_backbone_model,
            "finetuned": args.pruned_finetuned_model,
            "mrr10_full_eval": None,
            "ndcg10_full_eval": None,
            "features": pruned_features,
            "train_metrics": train_metrics,
            "meta": {
                "backbone_vocab_size": float(pruned_backbone_payload["vocab_survival"]["vocab_size"]),
                "finetuned_vocab_size": float(pruned_finetuned_payload["vocab_survival"]["vocab_size"]),
            },
        },
        "comparison": {
            "vocab_size_normal": normal_vocab_size,
            "vocab_size_pruned_hangul": float(pruned_backbone_payload["vocab_survival"]["vocab_size"]),
            "backbone_survival_frac_delta": float(pruned_features["backbone_survival_frac"])
            - float(normal_features["backbone_survival_frac"]),
            "finetuned_survival_frac_delta": float(pruned_features["finetuned_survival_frac"])
            - float(normal_features["finetuned_survival_frac"]),
            "survival_retention_delta": float(pruned_features["survival_retention"])
            - float(normal_features["survival_retention"]),
            "l1_retention_delta": float(pruned_features["l1_retention"])
            - float(normal_features["l1_retention"]),
            "l2_retention_delta": float(pruned_features["l2_retention"])
            - float(normal_features["l2_retention"]),
            "nonzero_activation_retention_delta": float(pruned_features["nonzero_activation_retention"])
            - float(normal_features["nonzero_activation_retention"]),
            "delta_nonzero_activation_std_delta": float(pruned_features["delta_nonzero_activation_std"])
            - float(normal_features.get("delta_nonzero_activation_std", 0.0)),
            "delta_q95_delta": float(pruned_features["delta_q95"]) - float(normal_features["delta_q95"]),
            "delta_q99_delta": float(pruned_features["delta_q99"]) - float(normal_features["delta_q99"]),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "analysis_pruned_hangul.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=True, indent=2)

    normal_vocab = normal_vocab_size
    pruned_vocab = float(pruned_backbone_payload["vocab_survival"]["vocab_size"])
    normal_f = normal_features
    pruned_f = pruned_features
    tm = train_metrics

    md_lines: list[str] = []
    md_lines.append("# Pruned-Hangul ANNA Addendum (2k Distribution Analysis)")
    md_lines.append("")
    md_lines.append("## Pair Mapping")
    md_lines.append(
        f"- Normal ANNA pair: `{normal_pair['backbone']}` -> `{normal_pair['finetuned']}`"
    )
    md_lines.append(
        f"- Pruned-hangul pair: `{args.pruned_backbone_model}` -> `{args.pruned_finetuned_model}`"
    )
    md_lines.append("")
    md_lines.append("## Retention Comparison")
    md_lines.append("")
    md_lines.append(
        "| Variant | Vocab Size | Backbone Survival | Fine-tuned Survival | Survival Retention | Backbone L1 | Fine-tuned L1 | L1 Retention | Backbone L2 | Fine-tuned L2 | L2 Retention | Backbone Nonzero Mean | Fine-tuned Nonzero Mean | Nonzero Retention | Backbone Nonzero Std | Fine-tuned Nonzero Std |"
    )
    md_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    md_lines.append(
        "| normal_anna | "
        f"{_format_float(normal_vocab, precision=0)} | "
        f"{normal_f['backbone_survival_frac']:.6f} | "
        f"{normal_f['finetuned_survival_frac']:.6f} | "
        f"{normal_f['survival_retention']:.6f} | "
        f"{normal_f['backbone_l1']:.3f} | "
        f"{normal_f['finetuned_l1']:.3f} | "
        f"{normal_f['l1_retention']:.6f} | "
        f"{normal_f['backbone_l2']:.3f} | "
        f"{normal_f['finetuned_l2']:.3f} | "
        f"{normal_f['l2_retention']:.6f} | "
        f"{normal_f['backbone_nonzero_activation_mean']:.6f} | "
        f"{normal_f['finetuned_nonzero_activation_mean']:.6f} | "
        f"{normal_f['nonzero_activation_retention']:.6f} | "
        f"{normal_f['backbone_nonzero_activation_std']:.6f} | "
        f"{normal_f['finetuned_nonzero_activation_std']:.6f} |"
    )
    md_lines.append(
        "| pruned_hangul_anna | "
        f"{pruned_vocab:.0f} | "
        f"{pruned_f['backbone_survival_frac']:.6f} | "
        f"{pruned_f['finetuned_survival_frac']:.6f} | "
        f"{pruned_f['survival_retention']:.6f} | "
        f"{pruned_f['backbone_l1']:.3f} | "
        f"{pruned_f['finetuned_l1']:.3f} | "
        f"{pruned_f['l1_retention']:.6f} | "
        f"{pruned_f['backbone_l2']:.3f} | "
        f"{pruned_f['finetuned_l2']:.3f} | "
        f"{pruned_f['l2_retention']:.6f} | "
        f"{pruned_f['backbone_nonzero_activation_mean']:.6f} | "
        f"{pruned_f['finetuned_nonzero_activation_mean']:.6f} | "
        f"{pruned_f['nonzero_activation_retention']:.6f} | "
        f"{pruned_f['backbone_nonzero_activation_std']:.6f} | "
        f"{pruned_f['finetuned_nonzero_activation_std']:.6f} |"
    )
    md_lines.append("")
    md_lines.append("## Logit-Shift Comparison")
    md_lines.append("")
    md_lines.append("| Variant | Delta q95 | Delta q99 | Delta NegFrac | Delta NonZero Activation Mean |")
    md_lines.append("|---|---:|---:|---:|---:|")
    md_lines.append(
        "| normal_anna | "
        f"{normal_f['delta_q95']:.6f} | "
        f"{normal_f['delta_q99']:.6f} | "
        f"{normal_f['delta_neg_frac']:.6f} | "
        f"{normal_f['delta_nonzero_activation_mean']:.6f} |"
    )
    md_lines.append(
        "| pruned_hangul_anna | "
        f"{pruned_f['delta_q95']:.6f} | "
        f"{pruned_f['delta_q99']:.6f} | "
        f"{pruned_f['delta_neg_frac']:.6f} | "
        f"{pruned_f['delta_nonzero_activation_mean']:.6f} |"
    )
    md_lines.append("")
    md_lines.append("## Run Metadata (Pruned-Hangul)")
    md_lines.append(
        f"- best checkpoint step inferred from checkpoint filename: `{tm.get('selected_step', 'n/a')}`"
    )
    md_lines.append(
        "- best checkpoint val_MRR@10 (from checkpoint filename): "
        f"`{_format_float(tm.get('best_val_mrr10_from_checkpoint_name'))}`"
    )
    md_lines.append(f"- NanoMSMARCO nDCG@10 at selected step: `{_format_float(tm.get('nano_msmarco_ndcg10'))}`")
    md_lines.append(f"- NanoMSMARCO MRR@10 at selected step: `{_format_float(tm.get('nano_msmarco_mrr10'))}`")
    md_lines.append(
        f"- NanoMSMARCO avg FLOPS at selected step: `{_format_float(tm.get('nano_msmarco_avg_flops'))}`"
    )
    md_lines.append(
        f"- NanoMSMARCO query active dims at selected step: `{_format_float(tm.get('nano_msmarco_query_active_dims'))}`"
    )
    md_lines.append(
        f"- NanoMSMARCO corpus active dims at selected step: `{_format_float(tm.get('nano_msmarco_corpus_active_dims'))}`"
    )
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append(
        "- This addendum uses the same 2k-query distribution settings as `analysis_step1_4` for comparability."
    )
    md_lines.append(
        "- Full MSMARCO dev-full test metrics for this pruned-hangul checkpoint are not included here."
    )

    md_path = args.output_dir / "analysis_pruned_hangul.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines).rstrip() + "\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
