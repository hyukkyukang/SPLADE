import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    name: str
    path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze step-wise activation collapse trajectories for SPLADE runs "
            "using Lightning metrics.csv and nanobeir_metrics_step*.json."
        )
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help=(
            "Run specs in 'name=/abs/path' or '/abs/path' form. "
            "If omitted, uses discovered defaults."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="script/experiment/output/activation_collapse",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--drop_ratio",
        type=float,
        default=0.70,
        help="Collapse onset threshold ratio vs running peak.",
    )
    parser.add_argument(
        "--persist_points",
        type=int,
        default=3,
        help="Required consecutive evaluation points below threshold.",
    )
    parser.add_argument(
        "--smoothing_points",
        type=int,
        default=3,
        help="Rolling median window for smoothed signals.",
    )
    parser.add_argument(
        "--window_steps",
        type=int,
        default=10000,
        help="Event window size (+/- steps) around detected onset.",
    )
    parser.add_argument(
        "--onset_signal",
        type=str,
        default="auto",
        choices=("auto", "query_dims", "flops"),
        help="Signal used for onset detection.",
    )
    parser.add_argument(
        "--min_onset_step",
        type=int,
        default=10000,
        help="Ignore onset candidates earlier than this step (warmup guard).",
    )
    parser.add_argument(
        "--snapshot_offsets",
        type=str,
        default="-10000,-5000,0,5000,10000",
        help="Comma-separated step offsets for checkpoint snapshot planning.",
    )
    return parser.parse_args()


def _parse_run_spec(value: str) -> RunSpec:
    if "=" in value:
        name, path = value.split("=", 1)
        run_name: str = name.strip()
        run_path: Path = Path(path.strip()).expanduser().resolve()
    else:
        run_path = Path(value.strip()).expanduser().resolve()
        run_name = run_path.name
    if not run_name:
        raise ValueError(f"Invalid run spec: {value}")
    if not run_path.exists():
        raise FileNotFoundError(f"Run path not found: {run_path}")
    return RunSpec(name=run_name, path=run_path)


def _discover_default_runs() -> list[RunSpec]:
    candidates: list[tuple[str, str]] = [
        ("co_condenser", "/home/user/SPLADE/log/train/splade_v2_pp"),
        ("anna_baseline", "/home/user/SPLADE/log/train/splade_v2_pp_anna"),
        (
            "anna_high_reg",
            "/home/user/SPLADE/log/train/splade_v2_pp_anna/splade_v2_pp_anna_regnorm_q6d3",
        ),
        (
            "anna_high_reg_sched",
            "/home/user/SPLADE/log/train/splade_v2_pp_anna/splade_v2_pp_anna_regnorm_q12d6",
        ),
    ]
    runs: list[RunSpec] = []
    for name, path_text in candidates:
        path: Path = Path(path_text).expanduser().resolve()
        if path.exists():
            runs.append(RunSpec(name=name, path=path))
    return runs


def _load_metrics_csv(path: Path) -> pd.DataFrame:
    frame: pd.DataFrame = pd.read_csv(path)
    if "step" not in frame.columns:
        raise ValueError(f"'step' column missing in metrics csv: {path}")
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame = frame.dropna(subset=["step"]).copy()
    frame["step"] = frame["step"].astype(int)
    return frame


def _score_metrics_csv(path: Path) -> tuple[int, int, float]:
    try:
        frame: pd.DataFrame = _load_metrics_csv(path)
    except Exception:
        return (-1, -1, -1.0)
    if frame.empty:
        return (-1, -1, -1.0)
    step_max: int = int(frame["step"].max())
    val_non_na: int = 0
    for column in (
        "val_nDCG_10",
        "val_MRR_10",
        "val_nanobeir_NanoMSMARCO_dot_ndcg@10",
        "val_nanobeir_NanoMSMARCO_query_active_dims",
    ):
        if column in frame.columns:
            val_non_na += int(frame[column].notna().sum())
    return (step_max, val_non_na, path.stat().st_mtime)


def _find_best_metrics_csv(run_dir: Path) -> Path | None:
    metric_files: list[Path] = []
    lightning_logs_dir: Path = run_dir / "lightning_logs"
    if lightning_logs_dir.is_dir():
        metric_files.extend(sorted(lightning_logs_dir.rglob("metrics.csv")))
    direct_metrics: Path = run_dir / "metrics.csv"
    if direct_metrics.is_file():
        metric_files.append(direct_metrics)
    if not metric_files:
        return None
    scored: list[tuple[tuple[int, int, float], Path]] = []
    for file_path in metric_files:
        scored.append((_score_metrics_csv(file_path), file_path))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _last_non_null(series: pd.Series) -> float | int | str | None:
    dropped: pd.Series = series.dropna()
    if dropped.empty:
        return None
    return dropped.iloc[-1]


def _collapse_metrics_by_step(frame: pd.DataFrame) -> pd.DataFrame:
    grouped: pd.DataFrame = (
        frame.sort_values("step")
        .groupby("step", as_index=False)
        .agg(_last_non_null)
        .sort_values("step")
        .reset_index(drop=True)
    )
    return grouped


def _load_nanobeir_step_json(run_dir: Path) -> pd.DataFrame:
    files: list[Path] = sorted(run_dir.glob("nanobeir_metrics_step*.json"))
    if not files:
        return pd.DataFrame(columns=["step"])
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"step(\d+)")
    for file_path in files:
        match = pattern.search(file_path.name)
        if match is None:
            continue
        step: int = int(match.group(1))
        with file_path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
        row: dict[str, Any] = {"step": step, "nanobeir_file": str(file_path)}
        row.update(payload)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["step"])
    frame = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return frame


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out: pd.DataFrame = frame.copy()

    def pick(*columns: str) -> pd.Series:
        picked: pd.Series = pd.Series([np.nan] * len(out), index=out.index, dtype=float)
        for column in columns:
            if column not in out.columns:
                continue
            values: pd.Series = pd.to_numeric(out[column], errors="coerce")
            picked = picked.combine_first(values)
        return picked

    out["retrieval_ndcg10"] = pick(
        "val_nanobeir_NanoMSMARCO_dot_ndcg@10",
        "NanoMSMARCO_dot_ndcg@10",
        "val_nDCG_10",
    )
    out["retrieval_mrr10"] = pick(
        "val_nanobeir_NanoMSMARCO_dot_mrr@10",
        "NanoMSMARCO_dot_mrr@10",
        "val_MRR_10",
    )
    out["query_active_dims"] = pick(
        "val_nanobeir_NanoMSMARCO_query_active_dims",
        "NanoMSMARCO_query_active_dims",
    )
    out["corpus_active_dims"] = pick(
        "val_nanobeir_NanoMSMARCO_corpus_active_dims",
        "NanoMSMARCO_corpus_active_dims",
    )
    out["avg_flops"] = pick(
        "val_nanobeir_NanoMSMARCO_avg_flops",
        "NanoMSMARCO_avg_flops",
    )
    for column in (
        "train_q_reg_step",
        "train_d_reg_step",
        "train_in_batch_loss_step",
        "train_loss_step",
        "train_reg_query_lambda_step",
        "train_reg_doc_lambda_step",
        "train_reg_lambda_scale",
        "lr-Adam",
    ):
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).median()


def _detect_onset_from_series(
    steps: pd.Series,
    values: pd.Series,
    *,
    drop_ratio: float,
    persist_points: int,
    min_onset_step: int,
) -> tuple[int | None, pd.Series]:
    clean: pd.DataFrame = pd.DataFrame({"step": steps, "value": values}).dropna()
    if clean.empty:
        return None, pd.Series([np.nan] * len(steps), index=steps.index)

    running_peak: float = -float("inf")
    ratio_by_step: dict[int, float] = {}
    for _, row in clean.iterrows():
        step_value: int = int(row["step"])
        value: float = float(row["value"])
        running_peak = max(running_peak, value)
        ratio: float = value / running_peak if running_peak > 0 else 1.0
        ratio_by_step[step_value] = ratio

    ratio_series: pd.Series = pd.Series(np.nan, index=steps.index, dtype=float)
    for idx, step_value in enumerate(steps.tolist()):
        if pd.isna(step_value):
            continue
        ratio_series.iloc[idx] = ratio_by_step.get(int(step_value), np.nan)

    threshold_mask: np.ndarray = clean["value"].to_numpy(dtype=float)
    peaks: np.ndarray = np.maximum.accumulate(threshold_mask)
    ratios: np.ndarray = np.divide(
        threshold_mask,
        peaks,
        out=np.ones_like(threshold_mask),
        where=peaks > 0,
    )
    below: np.ndarray = ratios < float(drop_ratio)
    onset_step: int | None = None
    eligible: np.ndarray = clean["step"].to_numpy(dtype=int) >= int(min_onset_step)
    candidate_mask: np.ndarray = below & eligible
    if persist_points <= 1:
        indices = np.where(candidate_mask)[0]
        if len(indices) > 0:
            onset_step = int(clean.iloc[int(indices[0])]["step"])
    else:
        for start in range(0, len(candidate_mask) - persist_points + 1):
            window_below: np.ndarray = candidate_mask[start : start + persist_points]
            if not bool(np.all(window_below)):
                continue
            if not bool(
                np.all(
                    clean["step"].to_numpy(dtype=int)[start : start + persist_points]
                    >= int(min_onset_step)
                )
            ):
                continue
            if bool(np.all(window_below)):
                onset_step = int(clean.iloc[start]["step"])
                break
    return onset_step, ratio_series


def _choose_onset(
    frame: pd.DataFrame,
    *,
    onset_signal: str,
    drop_ratio: float,
    persist_points: int,
    min_onset_step: int,
) -> tuple[int | None, str, pd.Series, pd.Series]:
    query_onset, query_ratio = _detect_onset_from_series(
        frame["step"],
        frame["query_active_dims_smooth"],
        drop_ratio=drop_ratio,
        persist_points=persist_points,
        min_onset_step=min_onset_step,
    )
    flops_onset, flops_ratio = _detect_onset_from_series(
        frame["step"],
        frame["avg_flops_smooth"],
        drop_ratio=drop_ratio,
        persist_points=persist_points,
        min_onset_step=min_onset_step,
    )
    if onset_signal == "query_dims":
        return query_onset, "query_dims", query_ratio, flops_ratio
    if onset_signal == "flops":
        return flops_onset, "flops", query_ratio, flops_ratio
    candidates: list[tuple[int, str]] = []
    if query_onset is not None:
        candidates.append((query_onset, "query_dims"))
    if flops_onset is not None:
        candidates.append((flops_onset, "flops"))
    if not candidates:
        return None, "none", query_ratio, flops_ratio
    candidates.sort(key=lambda item: item[0])
    return candidates[0][0], candidates[0][1], query_ratio, flops_ratio


def _window_slice(frame: pd.DataFrame, *, start: int, end: int) -> pd.DataFrame:
    return frame[(frame["step"] >= start) & (frame["step"] <= end)].copy()


def _metric_delta(
    frame: pd.DataFrame,
    *,
    onset_step: int,
    window_steps: int,
    metric: str,
) -> dict[str, float | None]:
    pre = _window_slice(frame, start=onset_step - window_steps, end=onset_step - 1)
    post = _window_slice(frame, start=onset_step + 1, end=onset_step + window_steps)
    pre_values: pd.Series = pd.to_numeric(pre.get(metric), errors="coerce").dropna()
    post_values: pd.Series = pd.to_numeric(post.get(metric), errors="coerce").dropna()
    pre_med: float | None = (
        float(pre_values.median()) if not pre_values.empty else None
    )
    post_med: float | None = (
        float(post_values.median()) if not post_values.empty else None
    )
    if pre_med is None or post_med is None:
        return {
            "pre_median": pre_med,
            "post_median": post_med,
            "delta": None,
            "relative_delta": None,
        }
    delta: float = post_med - pre_med
    rel: float | None = None
    if abs(pre_med) > 1e-12:
        rel = (post_med / pre_med) - 1.0
    return {
        "pre_median": pre_med,
        "post_median": post_med,
        "delta": delta,
        "relative_delta": rel,
    }


def _build_event_window_attribution(
    frame: pd.DataFrame,
    *,
    onset_step: int | None,
    window_steps: int,
) -> dict[str, Any]:
    metrics: list[str] = [
        "query_active_dims_smooth",
        "avg_flops_smooth",
        "retrieval_ndcg10_smooth",
        "retrieval_mrr10_smooth",
        "train_q_reg_step",
        "train_d_reg_step",
        "train_reg_query_lambda_step",
        "train_reg_doc_lambda_step",
        "train_reg_lambda_scale",
        "train_in_batch_loss_step",
        "lr-Adam",
    ]
    if onset_step is None:
        return {"onset_step": None, "metrics": {}, "trigger_candidate": None}
    metric_deltas: dict[str, Any] = {}
    for metric in metrics:
        if metric not in frame.columns:
            continue
        metric_deltas[metric] = _metric_delta(
            frame, onset_step=onset_step, window_steps=window_steps, metric=metric
        )
    trigger_metrics: list[str] = [
        "train_reg_query_lambda_step",
        "train_reg_doc_lambda_step",
        "train_reg_lambda_scale",
        "train_q_reg_step",
        "train_d_reg_step",
        "train_in_batch_loss_step",
        "lr-Adam",
    ]
    ranked: list[tuple[float, str]] = []
    for metric in trigger_metrics:
        payload = metric_deltas.get(metric)
        if payload is None:
            continue
        relative_delta = payload.get("relative_delta")
        absolute_delta = payload.get("delta")
        score: float | None = None
        if relative_delta is not None and math.isfinite(float(relative_delta)):
            score = abs(float(relative_delta))
        elif absolute_delta is not None and math.isfinite(float(absolute_delta)):
            score = abs(float(absolute_delta))
        if score is None:
            continue
        ranked.append((score, metric))
    ranked.sort(reverse=True)
    trigger_candidate: str | None = ranked[0][1] if ranked else None
    return {
        "onset_step": int(onset_step),
        "metrics": metric_deltas,
        "trigger_candidate": trigger_candidate,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _lagged_correlation(
    frame: pd.DataFrame,
    *,
    target: str,
    predictors: list[str],
    lags: list[int],
) -> dict[str, Any]:
    if target not in frame.columns:
        return {}
    eval_frame: pd.DataFrame = frame.dropna(subset=[target]).sort_values("step").copy()
    results: dict[str, Any] = {}
    for predictor in predictors:
        if predictor not in eval_frame.columns:
            continue
        predictor_values: np.ndarray = pd.to_numeric(
            eval_frame[predictor], errors="coerce"
        ).to_numpy(dtype=float)
        target_values: np.ndarray = pd.to_numeric(
            eval_frame[target], errors="coerce"
        ).to_numpy(dtype=float)
        payload: dict[str, Any] = {}
        for lag in lags:
            if lag < 0 or len(eval_frame) <= lag + 2:
                payload[f"lag_{lag}"] = None
                continue
            x = predictor_values[: len(predictor_values) - lag]
            y = target_values[lag:]
            valid_mask = np.isfinite(x) & np.isfinite(y)
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            payload[f"lag_{lag}"] = _pearson(x_valid, y_valid)
        results[predictor] = payload
    return results


def _parse_checkpoint_step(path: Path) -> int | None:
    pattern = re.compile(r"step(?:step=|=)(\d+)")
    match = pattern.search(path.name)
    if match is None:
        try:
            import torch

            payload: Any = torch.load(str(path), map_location="cpu")
            if isinstance(payload, dict):
                global_step: Any = payload.get("global_step")
                if isinstance(global_step, (int, float)):
                    return int(global_step)
        except Exception:
            return None
        return None
    return int(match.group(1))


def _build_snapshot_plan(
    run_dir: Path, *, onset_step: int | None, offsets: list[int]
) -> list[dict[str, Any]]:
    if onset_step is None:
        return []
    checkpoint_files: list[Path] = []
    for child in sorted(run_dir.iterdir()):
        if child.is_file() and child.suffix == ".ckpt":
            checkpoint_files.append(child)
        if child.is_dir() and child.name.startswith("checkpoints"):
            checkpoint_files.extend(sorted(child.glob("*.ckpt")))
    parsed: list[tuple[int, Path]] = []
    for file_path in checkpoint_files:
        step = _parse_checkpoint_step(file_path)
        if step is None:
            continue
        parsed.append((step, file_path))
    if not parsed:
        return []
    parsed.sort(key=lambda item: item[0])
    plan_rows: list[dict[str, Any]] = []
    for offset in offsets:
        target_step: int = int(onset_step + offset)
        nearest: tuple[int, Path] = min(parsed, key=lambda item: abs(item[0] - target_step))
        nearest_step, nearest_path = nearest
        plan_rows.append(
            {
                "onset_step": int(onset_step),
                "offset": int(offset),
                "target_step": int(target_step),
                "selected_checkpoint_step": int(nearest_step),
                "selected_checkpoint_path": str(nearest_path),
                "distance": int(abs(nearest_step - target_step)),
            }
        )
    return plan_rows


def _plot_run_timeline(
    frame: pd.DataFrame,
    *,
    run_name: str,
    onset_step: int | None,
    onset_signal: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    x = frame["step"].to_numpy(dtype=float)

    axes[0].plot(x, frame["retrieval_ndcg10"], label="retrieval_ndcg10", alpha=0.35)
    axes[0].plot(
        x,
        frame["retrieval_ndcg10_smooth"],
        label="retrieval_ndcg10_smooth",
        linewidth=2.0,
    )
    axes[0].plot(x, frame["retrieval_mrr10"], label="retrieval_mrr10", alpha=0.35)
    axes[0].plot(
        x,
        frame["retrieval_mrr10_smooth"],
        label="retrieval_mrr10_smooth",
        linewidth=2.0,
    )
    axes[0].set_ylabel("Retrieval")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.2)

    axes[1].plot(x, frame["query_active_dims"], label="query_active_dims", alpha=0.35)
    axes[1].plot(
        x,
        frame["query_active_dims_smooth"],
        label="query_active_dims_smooth",
        linewidth=2.0,
    )
    axes[1].plot(x, frame["avg_flops"], label="avg_flops", alpha=0.35)
    axes[1].plot(
        x,
        frame["avg_flops_smooth"],
        label="avg_flops_smooth",
        linewidth=2.0,
    )
    axes[1].set_ylabel("Dims/FLOPS")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.2)

    axes[2].plot(x, frame["train_q_reg_step"], label="train_q_reg_step", linewidth=1.5)
    axes[2].plot(x, frame["train_d_reg_step"], label="train_d_reg_step", linewidth=1.5)
    axes[2].plot(
        x,
        frame["train_in_batch_loss_step"],
        label="train_in_batch_loss_step",
        linewidth=1.5,
    )
    axes[2].set_ylabel("Reg/Loss")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.2)

    axes[3].plot(
        x,
        frame["train_reg_query_lambda_step"],
        label="train_reg_query_lambda_step",
        linewidth=1.5,
    )
    axes[3].plot(
        x,
        frame["train_reg_doc_lambda_step"],
        label="train_reg_doc_lambda_step",
        linewidth=1.5,
    )
    axes[3].plot(x, frame["train_reg_lambda_scale"], label="train_reg_lambda_scale")
    axes[3].plot(x, frame["lr-Adam"], label="lr-Adam")
    axes[3].set_ylabel("Lambda/LR")
    axes[3].set_xlabel("Step")
    axes[3].legend(loc="best")
    axes[3].grid(alpha=0.2)

    if onset_step is not None:
        for axis in axes:
            axis.axvline(
                float(onset_step),
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )
    title_suffix: str = (
        f"onset={onset_step} ({onset_signal})" if onset_step is not None else "no onset"
    )
    fig.suptitle(f"{run_name} Activation Collapse Timeline - {title_suffix}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _analyze_run(
    run: RunSpec,
    *,
    drop_ratio: float,
    persist_points: int,
    smoothing_points: int,
    onset_signal: str,
    window_steps: int,
    min_onset_step: int,
    snapshot_offsets: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    metrics_csv: Path | None = _find_best_metrics_csv(run.path)
    if metrics_csv is None:
        raise FileNotFoundError(f"No metrics.csv found for run: {run.path}")
    raw_metrics: pd.DataFrame = _load_metrics_csv(metrics_csv)
    step_metrics: pd.DataFrame = _collapse_metrics_by_step(raw_metrics)
    nanobeir: pd.DataFrame = _load_nanobeir_step_json(run.path)
    merged: pd.DataFrame = step_metrics.merge(
        nanobeir, on="step", how="outer", suffixes=("", "_nanobeir")
    )
    merged = merged.sort_values("step").reset_index(drop=True)
    merged = _canonicalize_columns(merged)

    for signal in ("retrieval_ndcg10", "retrieval_mrr10", "query_active_dims", "avg_flops"):
        merged[f"{signal}_smooth"] = _smooth_series(merged[signal], smoothing_points)

    onset_step, onset_source, query_ratio, flops_ratio = _choose_onset(
        merged,
        onset_signal=onset_signal,
        drop_ratio=drop_ratio,
        persist_points=persist_points,
        min_onset_step=min_onset_step,
    )
    merged["query_ratio_vs_peak"] = query_ratio
    merged["flops_ratio_vs_peak"] = flops_ratio

    attribution: dict[str, Any] = _build_event_window_attribution(
        merged, onset_step=onset_step, window_steps=window_steps
    )
    lagged: dict[str, Any] = _lagged_correlation(
        merged,
        target="retrieval_ndcg10",
        predictors=[
            "query_active_dims_smooth",
            "avg_flops_smooth",
            "train_q_reg_step",
            "train_d_reg_step",
            "train_reg_query_lambda_step",
            "train_reg_doc_lambda_step",
            "train_reg_lambda_scale",
            "train_in_batch_loss_step",
            "lr-Adam",
        ],
        lags=[1, 2],
    )
    snapshot_plan: list[dict[str, Any]] = _build_snapshot_plan(
        run.path, onset_step=onset_step, offsets=snapshot_offsets
    )

    merged_dir: Path = output_dir / "merged"
    plot_dir: Path = output_dir / "plots"
    attribution_dir: Path = output_dir / "event_windows"
    snapshot_dir: Path = output_dir / "snapshot_plans"
    merged_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    attribution_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    merged_path: Path = merged_dir / f"{run.name}.csv"
    merged.to_csv(merged_path, index=False)

    plot_path: Path = plot_dir / f"{run.name}.png"
    _plot_run_timeline(
        merged,
        run_name=run.name,
        onset_step=onset_step,
        onset_signal=onset_source,
        output_path=plot_path,
    )

    event_payload: dict[str, Any] = {
        "run_name": run.name,
        "run_path": str(run.path),
        "metrics_csv": str(metrics_csv),
        "onset_step": onset_step,
        "onset_signal": onset_source,
        "drop_ratio": drop_ratio,
        "persist_points": persist_points,
        "window_steps": window_steps,
        "attribution": attribution,
        "lagged_ndcg10": lagged,
    }
    event_path: Path = attribution_dir / f"{run.name}.json"
    with event_path.open("w", encoding="utf-8") as handle:
        json.dump(event_payload, handle, indent=2)

    snapshot_path: Path = snapshot_dir / f"{run.name}.json"
    with snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot_plan, handle, indent=2)

    retrieval_peak: float | None = None
    if merged["retrieval_ndcg10"].notna().any():
        retrieval_peak = float(pd.to_numeric(merged["retrieval_ndcg10"], errors="coerce").max())

    return {
        "run_name": run.name,
        "run_path": str(run.path),
        "metrics_csv": str(metrics_csv),
        "onset_step": onset_step,
        "onset_signal": onset_source,
        "trigger_candidate": attribution.get("trigger_candidate"),
        "retrieval_ndcg10_peak": retrieval_peak,
        "merged_csv": str(merged_path),
        "plot_path": str(plot_path),
        "event_window_json": str(event_path),
        "snapshot_plan_json": str(snapshot_path),
    }


def _write_summary(outputs: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path: Path = output_dir / "summary.json"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2)

    summary_frame: pd.DataFrame = pd.DataFrame(outputs)
    summary_csv_path: Path = output_dir / "onset_summary.csv"
    summary_frame.to_csv(summary_csv_path, index=False)

    markdown_lines: list[str] = []
    markdown_lines.append("# Activation Collapse Summary")
    markdown_lines.append("")
    markdown_lines.append(
        "| run | onset_step | onset_signal | trigger_candidate | peak_ndcg10 |"
    )
    markdown_lines.append("|---|---:|---|---|---:|")
    for row in outputs:
        peak_text = ""
        if row.get("retrieval_ndcg10_peak") is not None:
            peak_text = f"{float(row['retrieval_ndcg10_peak']):.6f}"
        onset_text = "" if row.get("onset_step") is None else str(row["onset_step"])
        markdown_lines.append(
            f"| {row['run_name']} | {onset_text} | {row.get('onset_signal', '')} | "
            f"{row.get('trigger_candidate', '') or ''} | {peak_text} |"
        )
    markdown_path: Path = output_dir / "summary.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir: Path = Path(str(args.output_dir)).expanduser().resolve()
    if args.runs:
        run_specs: list[RunSpec] = [_parse_run_spec(value) for value in args.runs]
    else:
        run_specs = _discover_default_runs()
    if not run_specs:
        raise RuntimeError("No runs provided/discovered. Use --runs name=/path ...")

    snapshot_offsets: list[int] = []
    for token in str(args.snapshot_offsets).split(","):
        token_clean = token.strip()
        if not token_clean:
            continue
        snapshot_offsets.append(int(token_clean))

    results: list[dict[str, Any]] = []
    for run in run_specs:
        result: dict[str, Any] = _analyze_run(
            run,
            drop_ratio=float(args.drop_ratio),
            persist_points=int(args.persist_points),
            smoothing_points=int(args.smoothing_points),
            onset_signal=str(args.onset_signal),
            window_steps=int(args.window_steps),
            min_onset_step=int(args.min_onset_step),
            snapshot_offsets=snapshot_offsets,
            output_dir=output_dir,
        )
        results.append(result)
    _write_summary(results, output_dir)
    print(f"Wrote activation-collapse analysis to {output_dir}")


if __name__ == "__main__":
    main()
