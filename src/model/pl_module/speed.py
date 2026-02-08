import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig

from src.search.retrieval import IndexedRetrievalHelper
from src.model.pl_module.utils import (
    build_splade_model_with_checkpoint,
    resolve_cudagraph_mark_step,
    validate_torch_compile_mode,
)
from src.model.retriever.sparse.neural.splade import SpladeModel
from src.utils.logging import log_if_rank_zero

logger: logging.Logger = logging.getLogger("RetrievalSpeedLightningModule")


def _append_rank_suffix(path: Path, rank: int) -> Path:
    suffix: str = path.suffix
    stem: str = path.stem
    if suffix:
        return path.with_name(f"{stem}.rank{rank}{suffix}")
    return path.with_name(f"{path.name}.rank{rank}")


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    values_np: np.ndarray = np.asarray(values, dtype=np.float64)
    percentiles: np.ndarray = np.percentile(values_np, [50, 95, 99])
    return {
        "count": float(values_np.size),
        "mean_ms": float(values_np.mean()),
        "median_ms": float(percentiles[0]),
        "p95_ms": float(percentiles[1]),
        "p99_ms": float(percentiles[2]),
    }


class RetrievalSpeedLightningModule(L.LightningModule):
    """LightningModule for SPLADE speed benchmarking."""

    # --- Special methods ---
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.automatic_optimization: bool = False
        self.cfg: DictConfig = cfg
        self.save_hyperparameters(cfg)

        self.model: SpladeModel = self._load_model()
        self._torch_compile_mark_step: Callable[[], None] | None = None
        self._setup_torch_compile()

        self._retrieval_helper = IndexedRetrievalHelper(
            cfg=cfg, logger=logger, index_context="speed"
        )

        speed_cfg: DictConfig = self.cfg.speed
        self._warmup_steps: int = int(speed_cfg.warmup_steps)
        self._breakdown_search_time: bool = bool(speed_cfg.breakdown_search_time)
        self._output_dir: Path = Path(str(speed_cfg.output_dir))
        self._summary_filename: str = str(speed_cfg.summary_filename)
        self._per_query_filename: str = str(speed_cfg.per_query_filename)
        self._merge_ranks: bool = bool(speed_cfg.merge_ranks)

        self._mode_names: dict[int, str] = {0: "per_query", 1: "batch"}
        self._timings: dict[str, dict[str, list[float]]] = {}
        self._per_query_records: list[dict[str, Any]] = []
        self._mode_query_counts: dict[str, int] = {}
        self._mode_total_time_s: dict[str, float] = {}
        self._use_cuda: bool = False

    # --- Protected methods ---
    def _load_model(self) -> SpladeModel:
        checkpoint_path: str | None = self.cfg.testing.checkpoint_path
        return build_splade_model_with_checkpoint(
            cfg=self.cfg,
            use_cpu=bool(self.cfg.testing.use_cpu),
            checkpoint_path=checkpoint_path,
            logger=logger,
        )

    def _setup_torch_compile(self) -> dict[str, Any]:
        compile_enabled: bool = bool(self.cfg.testing.torch_compile)
        compile_available: bool = hasattr(torch, "compile")
        self._torch_compile_mark_step = None
        if compile_enabled and not compile_available:
            log_if_rank_zero(
                logger,
                "torch.compile is not available in this PyTorch build; continuing "
                "without compilation.",
                level="warning",
            )
            return {}
        if not compile_enabled or not compile_available:
            return {}
        compile_mode_value: Any = self.cfg.testing.torch_compile_mode
        compile_mode, compile_mode_kwargs = validate_torch_compile_mode(
            compile_mode_value
        )
        if compile_mode in {"reduce-overhead", "max-autotune"}:
            self._torch_compile_mark_step = resolve_cudagraph_mark_step()
        query_wrapper: torch.nn.Module = self.model._query_encoder_wrapper
        query_encoder = torch.compile(query_wrapper, **compile_mode_kwargs)
        self.model._query_encoder_fn = query_encoder
        return compile_mode_kwargs

    def _sync_device(self) -> None:
        if not self._use_cuda:
            return
        torch.cuda.synchronize()

    def _resolve_mode(self, dataloader_idx: int) -> str:
        return self._mode_names.get(dataloader_idx, f"loader_{dataloader_idx}")

    def _resolve_batch_size(self) -> int:
        batch_sizes = [int(value) for value in self.cfg.speed.batch_sizes]
        if not batch_sizes:
            raise ValueError("speed.batch_sizes must contain at least one value.")
        if any(value <= 0 for value in batch_sizes):
            raise ValueError("speed.batch_sizes must be positive integers.")
        return batch_sizes[0]

    def _process_batch(
        self, batch: Dict[str, Any], mode: str, *, record_metrics: bool
    ) -> None:
        qids: List[str] = batch["qid"]
        query_input_ids: torch.Tensor = batch["query_input_ids"].to(self.device)
        query_attention_mask: torch.Tensor = batch["query_attention_mask"].to(
            self.device
        )
        batch_size: int = int(query_input_ids.shape[0])

        self._sync_device()
        encode_start: float = time.perf_counter()
        query_reps: torch.Tensor = self._retrieval_helper.encode_queries(
            self.model,
            query_input_ids,
            query_attention_mask,
            self._torch_compile_mark_step,
        )
        self._sync_device()
        encode_end: float = time.perf_counter()

        sparsify_ms: float | None = None
        score_ms: float | None = None
        if self._breakdown_search_time:
            sparsify_start: float = time.perf_counter()
            q_indices_list, q_values_list = self._retrieval_helper._sparsify_queries(
                query_reps
            )
            self._sync_device()
            sparsify_end: float = time.perf_counter()

            score_start: float = time.perf_counter()
            scored = self._retrieval_helper._score_batch(q_indices_list, q_values_list)
            _ = self._format_scored_results(scored)
            self._sync_device()
            score_end: float = time.perf_counter()

            sparsify_ms = (sparsify_end - sparsify_start) * 1000.0
            score_ms = (score_end - score_start) * 1000.0
            search_end: float = score_end
        else:
            search_start: float = time.perf_counter()
            _ = self._retrieval_helper.score_queries(query_reps)
            self._sync_device()
            search_end = time.perf_counter()

        encode_ms: float = (encode_end - encode_start) * 1000.0
        if (
            self._breakdown_search_time
            and sparsify_ms is not None
            and score_ms is not None
        ):
            search_ms: float = sparsify_ms + score_ms
        else:
            search_ms = (search_end - search_start) * 1000.0
        total_ms: float = (search_end - encode_start) * 1000.0

        if not record_metrics:
            return

        per_query_encode_ms: float = encode_ms / float(max(batch_size, 1))
        per_query_search_ms: float = search_ms / float(max(batch_size, 1))
        per_query_total_ms: float = total_ms / float(max(batch_size, 1))
        per_query_sparsify_ms: float | None = (
            None if sparsify_ms is None else sparsify_ms / float(max(batch_size, 1))
        )
        per_query_score_ms: float | None = (
            None if score_ms is None else score_ms / float(max(batch_size, 1))
        )

        self._mode_query_counts[mode] += int(batch_size)
        self._mode_total_time_s[mode] += total_ms / 1000.0

        rank: int = int(self.trainer.global_rank)
        repeat_count: int = len(qids)
        self._timings[mode]["encode_ms"].extend([per_query_encode_ms] * repeat_count)
        self._timings[mode]["search_ms"].extend([per_query_search_ms] * repeat_count)
        self._timings[mode]["total_ms"].extend([per_query_total_ms] * repeat_count)
        if per_query_sparsify_ms is not None:
            self._timings[mode]["sparsify_ms"].extend(
                [per_query_sparsify_ms] * repeat_count
            )
        if per_query_score_ms is not None:
            self._timings[mode]["score_ms"].extend([per_query_score_ms] * repeat_count)
        for qid in qids:
            record: dict[str, Any] = {
                "mode": mode,
                "qid": qid,
                "encode_ms": per_query_encode_ms,
                "search_ms": per_query_search_ms,
                "total_ms": per_query_total_ms,
                "batch_size": batch_size,
                "rank": rank,
            }
            if per_query_sparsify_ms is not None:
                record["sparsify_ms"] = per_query_sparsify_ms
            if per_query_score_ms is not None:
                record["score_ms"] = per_query_score_ms
            self._per_query_records.append(record)

    def _resolve_warmup_dataloaders(self) -> list[Any]:
        trainer: L.Trainer | None = self.trainer
        if trainer is None:
            return []
        datamodule: L.LightningDataModule | None = trainer.datamodule
        if datamodule is None:
            return []
        dataloaders: Any = datamodule.test_dataloader()
        if dataloaders is None:
            return []
        if isinstance(dataloaders, list):
            return dataloaders
        if isinstance(dataloaders, tuple):
            return list(dataloaders)
        return [dataloaders]

    def _run_warmup(self) -> None:
        warmup_steps: int = int(self._warmup_steps)
        if warmup_steps <= 0:
            return
        dataloaders: list[Any] = self._resolve_warmup_dataloaders()
        if not dataloaders:
            log_if_rank_zero(
                logger,
                "Warmup skipped because no test dataloaders were found.",
                level="warning",
            )
            return
        log_if_rank_zero(logger, f"Running {warmup_steps} warmup steps per mode.")
        for dataloader_idx, dataloader in enumerate(dataloaders):
            mode: str = self._resolve_mode(dataloader_idx)
            try:
                if len(dataloader) == 0:
                    log_if_rank_zero(
                        logger,
                        f"Warmup skipped for {mode}: empty dataloader.",
                        level="warning",
                    )
                    continue
            except TypeError:
                pass
            steps_remaining: int = warmup_steps
            data_iter = iter(dataloader)
            while steps_remaining > 0:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        log_if_rank_zero(
                            logger,
                            f"Warmup stopped for {mode}: dataloader exhausted.",
                            level="warning",
                        )
                        break
                with torch.inference_mode():
                    self._process_batch(batch, mode, record_metrics=False)
                steps_remaining -= 1

    def _format_scored_results(
        self, scored: list[tuple[np.ndarray, np.ndarray]]
    ) -> list[tuple[list[str], list[float]]]:
        doc_ids: list[str] = self._retrieval_helper.doc_ids
        results: list[tuple[list[str], list[float]]] = []
        for top_docs, top_scores in scored:
            selected_doc_ids: list[str] = [
                doc_ids[int(doc_idx)] for doc_idx in top_docs.tolist()
            ]
            selected_scores: list[float] = [
                float(score) for score in top_scores.tolist()
            ]
            results.append((selected_doc_ids, selected_scores))
        return results

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def _resolve_output_path(self, filename: str, rank: int) -> Path:
        base_path: Path = self._output_dir / filename
        if int(self.trainer.world_size) <= 1:
            return base_path
        return _append_rank_suffix(base_path, rank)

    # --- Public methods ---
    def on_test_start(self) -> None:
        self.model.eval()
        self._retrieval_helper.setup()

        self._use_cuda = bool(self.device.type == "cuda")
        self._timings = {
            mode: {
                "encode_ms": [],
                "search_ms": [],
                "total_ms": [],
                "sparsify_ms": [],
                "score_ms": [],
            }
            for mode in self._mode_names.values()
        }
        self._per_query_records = []
        self._mode_query_counts = {mode: 0 for mode in self._mode_names.values()}
        self._mode_total_time_s = {mode: 0.0 for mode in self._mode_names.values()}
        self._run_warmup()

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _ = batch_idx
        mode: str = self._resolve_mode(dataloader_idx)
        self._process_batch(batch, mode, record_metrics=True)

    def on_test_end(self) -> None:
        self._retrieval_helper.shutdown()

        world_size: int = int(self.trainer.world_size)
        rank: int = int(self.trainer.global_rank)

        summary: dict[str, Any] = {
            "settings": {
                "device": str(self.device),
                "sample_queries": int(self.cfg.speed.sample_queries),
                "per_query_batch_size": 1,
                "batch_size": self._resolve_batch_size(),
                "warmup_steps": int(self.cfg.speed.warmup_steps),
                "breakdown_search_time": bool(self.cfg.speed.breakdown_search_time),
                "torch_compile": bool(self.cfg.testing.torch_compile),
                "torch_compile_mode": str(self.cfg.testing.torch_compile_mode),
                "gpu_sparsify": bool(self.cfg.testing.gpu_sparsify),
                "scoring_method": str(self.cfg.testing.scoring_method),
                "scoring_backend": str(self.cfg.testing.scoring_backend),
                "scoring_workers": int(
                    0
                    if self.cfg.testing.scoring_workers is None
                    else self.cfg.testing.scoring_workers
                ),
                "world_size": world_size,
                "rank": rank,
            },
            "modes": {},
        }

        for mode, timing in self._timings.items():
            query_count: int = self._mode_query_counts.get(mode, 0)
            total_time_s: float = self._mode_total_time_s.get(mode, 0.0)
            throughput: float = (
                float(query_count) / total_time_s if total_time_s > 0 else 0.0
            )
            mode_summary: dict[str, Any] = {
                "queries": int(query_count),
                "throughput_qps": throughput,
                "encode_ms": _summarize(timing["encode_ms"]),
                "search_ms": _summarize(timing["search_ms"]),
                "total_ms": _summarize(timing["total_ms"]),
            }
            if self._breakdown_search_time:
                mode_summary["sparsify_ms"] = _summarize(timing["sparsify_ms"])
                mode_summary["score_ms"] = _summarize(timing["score_ms"])
            summary["modes"][mode] = mode_summary

        summary_path: Path = self._resolve_output_path(self._summary_filename, rank)
        per_query_path: Path = self._resolve_output_path(self._per_query_filename, rank)
        self._write_json(summary_path, summary)
        self._write_jsonl(per_query_path, self._per_query_records)
        # Print new line for cleaner prints
        print("")
        log_if_rank_zero(
            logger,
            f"Wrote speed summary to {summary_path} and per-query timings to {per_query_path}",
        )

        if self._merge_ranks and world_size > 1:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if self.trainer.is_global_zero:
                merged_summary = self._merge_rank_summaries(world_size)
                merged_path: Path = self._output_dir / self._summary_filename
                self._write_json(merged_path, merged_summary)
                log_if_rank_zero(logger, f"Wrote merged speed summary to {merged_path}")

    def _merge_rank_summaries(self, world_size: int) -> dict[str, Any]:
        combined_records: list[dict[str, Any]] = []
        for rank in range(world_size):
            per_query_path: Path = self._resolve_output_path(
                self._per_query_filename, rank
            )
            if not per_query_path.exists():
                continue
            with per_query_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    combined_records.append(json.loads(line))

        combined_timings: dict[str, dict[str, list[float]]] = {}
        combined_counts: dict[str, int] = {}
        combined_total_time_s: dict[str, float] = {}
        for record in combined_records:
            mode: str = str(record.get("mode"))
            timing = combined_timings.setdefault(
                mode,
                {
                    "encode_ms": [],
                    "search_ms": [],
                    "total_ms": [],
                    "sparsify_ms": [],
                    "score_ms": [],
                },
            )
            timing["encode_ms"].append(float(record["encode_ms"]))
            timing["search_ms"].append(float(record["search_ms"]))
            timing["total_ms"].append(float(record["total_ms"]))
            if "sparsify_ms" in record:
                timing["sparsify_ms"].append(float(record["sparsify_ms"]))
            if "score_ms" in record:
                timing["score_ms"].append(float(record["score_ms"]))
            combined_counts[mode] = combined_counts.get(mode, 0) + 1
            combined_total_time_s[mode] = (
                combined_total_time_s.get(mode, 0.0)
                + float(record["total_ms"]) / 1000.0
            )

        merged_summary: dict[str, Any] = {
            "settings": {
                "device": str(self.device),
                "sample_queries": int(self.cfg.speed.sample_queries),
                "per_query_batch_size": 1,
                "batch_size": self._resolve_batch_size(),
                "warmup_steps": int(self.cfg.speed.warmup_steps),
                "breakdown_search_time": bool(self.cfg.speed.breakdown_search_time),
                "torch_compile": bool(self.cfg.testing.torch_compile),
                "torch_compile_mode": str(self.cfg.testing.torch_compile_mode),
                "gpu_sparsify": bool(self.cfg.testing.gpu_sparsify),
                "scoring_method": str(self.cfg.testing.scoring_method),
                "scoring_backend": str(self.cfg.testing.scoring_backend),
                "scoring_workers": int(
                    0
                    if self.cfg.testing.scoring_workers is None
                    else self.cfg.testing.scoring_workers
                ),
                "world_size": int(world_size),
                "rank": 0,
            },
            "modes": {},
        }

        for mode, timing in combined_timings.items():
            query_count: int = combined_counts.get(mode, 0)
            total_time_s: float = combined_total_time_s.get(mode, 0.0)
            throughput: float = (
                float(query_count) / total_time_s if total_time_s > 0 else 0.0
            )
            mode_summary: dict[str, Any] = {
                "queries": int(query_count),
                "throughput_qps": throughput,
                "encode_ms": _summarize(timing["encode_ms"]),
                "search_ms": _summarize(timing["search_ms"]),
                "total_ms": _summarize(timing["total_ms"]),
            }
            if self._breakdown_search_time:
                mode_summary["sparsify_ms"] = _summarize(timing["sparsify_ms"])
                mode_summary["score_ms"] = _summarize(timing["score_ms"])
            merged_summary["modes"][mode] = mode_summary
        return merged_summary
