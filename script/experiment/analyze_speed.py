import json
import logging
from itertools import product
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import RetrievalSpeedDataModule
from src.model.pl_module import RetrievalSpeedLightningModule
from src.utils import log_if_rank_zero
from src.utils.logging import get_logger
from src.utils.model_utils import apply_checkpoint_model_config
from src.utils.script_setup import (
    configure_script_environment,
    initialize_run,
    resolve_model_source,
    resolve_trainer_settings,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=True,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


def _resolve_summary_path(cfg: DictConfig, trainer: L.Trainer) -> Path:
    output_dir: Path = Path(str(cfg.speed.output_dir))
    filename: str = str(cfg.speed.summary_filename)
    summary_path: Path = output_dir / filename
    if summary_path.exists():
        return summary_path
    rank: int = int(trainer.global_rank)
    suffix: str = summary_path.suffix
    stem: str = summary_path.stem
    if suffix:
        return summary_path.with_name(f"{stem}.rank{rank}{suffix}")
    return summary_path.with_name(f"{summary_path.name}.rank{rank}")


def _load_summary(cfg: DictConfig, trainer: L.Trainer) -> dict[str, Any] | None:
    summary_path: Path = _resolve_summary_path(cfg, trainer)
    if not summary_path.exists():
        log_if_rank_zero(
            logger, f"Speed summary not found at {summary_path}", level="warning"
        )
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_run_tag(
    base_tag: object | None,
    batch_size: int,
    scoring_method: str,
    scoring_workers: int | None,
) -> str:
    tag_prefix: str = "" if base_tag is None else str(base_tag).strip()
    workers_label: str = "auto" if scoring_workers is None else str(scoring_workers)
    combo_tag: str = f"bs{batch_size}_method-{scoring_method}_workers-{workers_label}"
    if tag_prefix:
        return f"{tag_prefix}_{combo_tag}"
    return combo_tag


def _collect_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    settings: dict[str, Any] = summary.get("settings", {})
    batch_size_value: Any = settings.get("batch_size", 0)
    scoring_workers_value: Any = settings.get("scoring_workers")
    batch_size: int = int(batch_size_value) if batch_size_value is not None else 0
    scoring_workers: int = (
        0 if scoring_workers_value is None else int(scoring_workers_value)
    )
    base_row: dict[str, Any] = {
        "batch_size": batch_size,
        "scoring_method": str(settings.get("scoring_method", "")),
        "scoring_workers": scoring_workers,
    }
    modes: dict[str, Any] = summary.get("modes", {})
    for mode, stats in modes.items():
        row: dict[str, Any] = {
            **base_row,
            "mode": mode,
            "queries": int(stats.get("queries", 0)),
            "throughput_qps": float(stats.get("throughput_qps", 0.0)),
        }
        for metric in ("encode_ms", "search_ms", "total_ms", "sparsify_ms", "score_ms"):
            metric_stats: Any = stats.get(metric)
            if not isinstance(metric_stats, dict):
                continue
            row[f"{metric}_mean"] = float(metric_stats.get("mean_ms", 0.0))
            row[f"{metric}_median"] = float(metric_stats.get("median_ms", 0.0))
            row[f"{metric}_p95"] = float(metric_stats.get("p95_ms", 0.0))
            row[f"{metric}_p99"] = float(metric_stats.get("p99_ms", 0.0))
        rows.append(row)
    return rows


def _format_speed_summary_long_table(
    table: pd.DataFrame,
    base_cols: list[str],
    metric_prefixes: tuple[str, ...],
) -> str:
    stat_cols: dict[str, str] = {
        "mean": "mean_ms",
        "median": "median_ms",
        "p95": "p95_ms",
        "p99": "p99_ms",
    }
    long_rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        base_row = {col: row[col] for col in base_cols if col in table.columns}
        for metric in metric_prefixes:
            metric_row: dict[str, Any] = {}
            for stat_key, stat_label in stat_cols.items():
                column_name = f"{metric}_{stat_key}"
                if column_name in table.columns and pd.notna(row[column_name]):
                    metric_row[stat_label] = row[column_name]
            if metric_row:
                metric_row = {
                    **base_row,
                    "metric": metric.replace("_ms", ""),
                    **metric_row,
                }
                long_rows.append(metric_row)
    if not long_rows:
        return "No speed summary data available."
    long_table = pd.DataFrame(long_rows)
    sort_cols = [col for col in base_cols + ["metric"] if col in long_table.columns]
    if sort_cols:
        long_table = long_table.sort_values(sort_cols)
    ordered_cols = [
        col
        for col in base_cols + ["metric"] + list(stat_cols.values())
        if col in long_table.columns
    ]
    remaining_cols = [col for col in long_table.columns if col not in ordered_cols]
    long_table = long_table[ordered_cols + remaining_cols]
    return long_table.to_string(
        index=False,
        float_format=lambda value: f"{value:,.3f}",
        na_rep="",
    )


def _format_speed_summary_table(
    summary_data: dict[str, Any] | list[dict[str, Any]],
    summary_format: str,
) -> str:
    summaries: list[dict[str, Any]] = (
        [summary_data] if isinstance(summary_data, dict) else list(summary_data)
    )
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.extend(_collect_summary_rows(summary))
    if not rows:
        return "No speed summary data available."
    table = pd.DataFrame(rows)
    sort_cols = [
        col
        for col in ("batch_size", "scoring_method", "scoring_workers", "mode")
        if col in table.columns
    ]
    if sort_cols:
        table = table.sort_values(sort_cols)
    ordered_cols: list[str] = [
        col
        for col in (
            "batch_size",
            "scoring_method",
            "scoring_workers",
            "mode",
            "queries",
            "throughput_qps",
        )
        if col in table.columns
    ]
    metric_prefixes: tuple[str, ...] = (
        "encode_ms",
        "search_ms",
        "total_ms",
        "sparsify_ms",
        "score_ms",
    )
    metric_cols: list[str] = [
        col
        for metric in metric_prefixes
        for col in (
            f"{metric}_mean",
            f"{metric}_median",
            f"{metric}_p95",
            f"{metric}_p99",
        )
        if col in table.columns
    ]
    remaining_cols = [
        col for col in table.columns if col not in ordered_cols + metric_cols
    ]
    table = table[ordered_cols + metric_cols + remaining_cols]
    normalized_format = str(summary_format).strip().lower()
    if normalized_format == "wide":
        return table.to_string(
            index=False,
            float_format=lambda value: f"{value:,.3f}",
            na_rep="",
        )
    return _format_speed_summary_long_table(table, ordered_cols, metric_prefixes)


def _run_speed_once(run_cfg: DictConfig) -> dict[str, Any] | None:
    initialize_run(run_cfg, logger=logger, suppress_lightning_tips=True)
    speed_module = RetrievalSpeedLightningModule(cfg=run_cfg)
    data_module = RetrievalSpeedDataModule(cfg=run_cfg)
    speed_module.eval()

    trainer_kwargs, precision = resolve_trainer_settings(run_cfg.testing)
    trainer: L.Trainer = L.Trainer(
        precision=precision,
        default_root_dir=run_cfg.log_dir,
        logger=False,
        **trainer_kwargs,
    )
    trainer.test(model=speed_module, datamodule=data_module)
    log_if_rank_zero(logger, "Speed benchmark complete")
    if trainer.is_global_zero:
        return _load_summary(run_cfg, trainer)
    return None


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="analyze_speed")
def main(cfg: DictConfig) -> None:
    base_cfg = resolve_model_source(cfg, logger=logger, set_nanobeir_flag=False)
    base_cfg = apply_checkpoint_model_config(
        base_cfg,
        checkpoint_path=base_cfg.testing.checkpoint_path,
        logger=logger,
    )

    batch_sizes = [int(value) for value in base_cfg.speed.batch_sizes]
    if not batch_sizes:
        raise ValueError("speed.batch_sizes must contain at least one value.")
    if any(value <= 0 for value in batch_sizes):
        raise ValueError("speed.batch_sizes must be positive integers.")
    scoring_methods = [str(value).lower() for value in base_cfg.testing.scoring_methods]
    scoring_workers_list = [
        None if value is None else int(value)
        for value in base_cfg.testing.scoring_workers_list
    ]

    summaries: list[dict[str, Any]] = []
    total_runs = len(batch_sizes) * len(scoring_methods) * len(scoring_workers_list)
    for run_idx, (batch_size, scoring_method, scoring_workers) in enumerate(
        product(batch_sizes, scoring_methods, scoring_workers_list), start=1
    ):
        workers_label = "auto" if scoring_workers is None else str(scoring_workers)
        log_if_rank_zero(
            logger,
            f"Running speed config ({run_idx}/{total_runs}):\n"
            f"  batch_size: {batch_size}\n"
            f"  scoring_method: {scoring_method}\n"
            f"  scoring_workers: {workers_label}",
        )
        run_tag: str = _build_run_tag(
            base_cfg.tag, batch_size, scoring_method, scoring_workers
        )
        run_cfg = OmegaConf.merge(
            base_cfg,
            {
                "speed": {"batch_sizes": [batch_size]},
                "testing": {
                    "scoring_method": scoring_method,
                    "scoring_workers": scoring_workers,
                },
                "tag": run_tag,
            },
        )
        summary = _run_speed_once(run_cfg)
        if summary is not None:
            summaries.append(summary)
        log_if_rank_zero(logger, f"Completed speed config ({run_idx}/{total_runs})")

    if summaries:
        table = _format_speed_summary_table(
            summaries,
            summary_format=str(base_cfg.speed.summary_format),
        )
        log_if_rank_zero(logger, f"\nCombined speed summary:\n{table}")
    else:
        log_if_rank_zero(logger, "No speed summaries collected.", level="warning")


if __name__ == "__main__":
    main()
