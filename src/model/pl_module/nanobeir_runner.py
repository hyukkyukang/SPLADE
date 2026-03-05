import gc
import json
import logging
import os
from typing import Any, Callable, cast

import torch
from omegaconf import DictConfig
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

from src.utils.logging import log_if_rank_zero
from src.utils.sparse_encoder import (
    DocOnlySparseEncoderAdapter,
    SparseEncoderCache,
    build_doc_only_sparse_encoder_adapter,
    build_sparse_encoder_cache,
    resolve_nanobeir_compatibility,
    update_sparse_encoder_cache,
)


class NanoBEIREvaluationRunner:
    """Encapsulate NanoBEIR evaluation runtime state and execution."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        logger: logging.Logger,
        doc_only_enabled: bool,
    ) -> None:
        self.cfg: DictConfig = cfg
        self.logger: logging.Logger = logger
        self.doc_only_enabled: bool = bool(doc_only_enabled)

        nanobeir_cfg: DictConfig = cfg.nanobeir
        self.enabled: bool = bool(nanobeir_cfg.enabled)
        self.run_every_n_val: int = int(nanobeir_cfg.run_every_n_val)
        self.batch_size: int = int(nanobeir_cfg.batch_size)
        self.save_json: bool = bool(nanobeir_cfg.save_json)
        self.dataset_names: list[str] = [str(name) for name in nanobeir_cfg.datasets]
        self.use_cpu: bool = bool(nanobeir_cfg.use_cpu)

        self._val_counter: int = 0
        self._cache: SparseEncoderCache | None = None
        self._cache_device: torch.device | None = None
        self._doc_only_encoder: DocOnlySparseEncoderAdapter | None = None
        self._doc_only_device: torch.device | None = None
        self._evaluator: SparseNanoBEIREvaluator | None = None
        self._evaluator_datasets: list[str] = []
        self._evaluator_batch_size: int = int(self.batch_size)
        self._force_adapter_fallback: bool = False

        if self.enabled and not self.doc_only_enabled:
            compatible: bool
            reason: str | None
            compatible, reason = resolve_nanobeir_compatibility(self.cfg)
            if not compatible:
                self._force_adapter_fallback = True
                log_if_rank_zero(
                    self.logger,
                    "NanoBEIR SparseEncoder path disabled; using direct SPLADE "
                    f"adapter path instead. Reason: {reason}",
                    level="warning",
                )

        if self.enabled and not self.dataset_names:
            log_if_rank_zero(
                self.logger,
                "NanoBEIR evaluation disabled because nanobeir.datasets is empty.",
                level="warning",
            )
            self.enabled = False

    def should_run_eval(self, *, sanity_checking: bool) -> bool:
        if not self.enabled:
            return False
        if sanity_checking:
            return False
        if not self.dataset_names:
            return False
        run_every_val: int = int(self.run_every_n_val)
        if run_every_val <= 0:
            run_every_val = 1
        self._val_counter += 1
        return self._val_counter % run_every_val == 0

    def barrier(self, strategy: Any) -> None:
        barrier_fn: Any | None = (
            strategy.barrier if hasattr(strategy, "barrier") else None
        )
        if callable(barrier_fn):
            barrier_fn()
            return
        distributed_available: bool = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if not distributed_available:
            return
        torch.distributed.barrier()

    def reset_runtime_state(self) -> None:
        self._cache = None
        self._cache_device = None
        self._doc_only_encoder = None
        self._doc_only_device = None
        self._evaluator = None
        self._evaluator_datasets = []

    def cleanup_after_failure(self) -> None:
        self.reset_runtime_state()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def offload_cache_to_cpu(self) -> None:
        """Release NanoBEIR's extra GPU model copy between validations."""
        cache: SparseEncoderCache | None = self._cache
        cache_device: torch.device | None = self._cache_device
        if cache is None or cache_device is None:
            return
        if cache_device.type != "cuda":
            return
        try:
            cache.sparse_encoder.to(torch.device("cpu"))
            self._cache_device = torch.device("cpu")
        except Exception as exc:
            log_if_rank_zero(
                self.logger,
                "Failed to offload NanoBEIR cache to CPU; resetting cache. "
                f"Error: {exc}",
                level="warning",
            )
            self.reset_runtime_state()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _resolve_device(self, training_device: torch.device) -> torch.device:
        if self.doc_only_enabled or self._force_adapter_fallback:
            if self.use_cpu:
                log_if_rank_zero(
                    self.logger,
                    "NanoBEIR use_cpu ignored for direct SPLADE adapter path; "
                    "using training device.",
                    level="warning",
                )
            return training_device
        if self.use_cpu:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def resolve_device(self, training_device: torch.device) -> torch.device:
        return self._resolve_device(training_device)

    def run_eval(
        self,
        *,
        eval_model: torch.nn.Module,
        training_device: torch.device,
        global_step: int,
        log_dir: str,
        log_dict_fn: Callable[[dict[str, float]], None],
        masked_lm_incompatibility_predicate: Callable[[Exception], bool],
    ) -> None:
        device: torch.device = self._resolve_device(training_device)
        sparse_encoder: SparseEncoder | DocOnlySparseEncoderAdapter
        use_adapter_path: bool = bool(
            self.doc_only_enabled or self._force_adapter_fallback
        )
        if use_adapter_path:
            doc_cache: DocOnlySparseEncoderAdapter | None = self._doc_only_encoder
            doc_cache_device: torch.device | None = self._doc_only_device
            if doc_cache is None or doc_cache_device != device:
                doc_cache = build_doc_only_sparse_encoder_adapter(
                    cfg=self.cfg,
                    model=eval_model,
                    device=device,
                    batch_size=self.batch_size,
                )
                self._doc_only_encoder = doc_cache
                self._doc_only_device = device
            sparse_encoder = doc_cache
        else:
            cache: SparseEncoderCache | None = self._cache
            cache_device: torch.device | None = self._cache_device
            try:
                if cache is None or cache_device != device:
                    cache = build_sparse_encoder_cache(
                        cfg=self.cfg, model=eval_model, device=device
                    )
                    self._cache = cache
                    self._cache_device = device
                else:
                    update_sparse_encoder_cache(
                        cache=cache, model=eval_model, device=device
                    )
                sparse_encoder = cache.sparse_encoder
            except Exception as exc:
                if not masked_lm_incompatibility_predicate(exc):
                    raise
                self._force_adapter_fallback = True
                self._cache = None
                self._cache_device = None
                log_if_rank_zero(
                    self.logger,
                    "NanoBEIR SparseEncoder MLM path is incompatible with this "
                    "backbone type; falling back to the direct SPLADE adapter "
                    f"path for subsequent validations. Root cause: {exc}",
                    level="warning",
                )
                adapter_device: torch.device = training_device
                doc_cache = build_doc_only_sparse_encoder_adapter(
                    cfg=self.cfg,
                    model=eval_model,
                    device=adapter_device,
                    batch_size=self.batch_size,
                )
                self._doc_only_encoder = doc_cache
                self._doc_only_device = adapter_device
                sparse_encoder = doc_cache

        evaluator: SparseNanoBEIREvaluator
        if (
            self._evaluator is None
            or self._evaluator_datasets != self.dataset_names
            or self._evaluator_batch_size != self.batch_size
        ):
            evaluator = SparseNanoBEIREvaluator(
                dataset_names=self.dataset_names,
                batch_size=self.batch_size,
            )
            self._evaluator = evaluator
            self._evaluator_datasets = list(self.dataset_names)
            self._evaluator_batch_size = int(self.batch_size)
        else:
            evaluator = self._evaluator

        with torch.no_grad():
            results: dict[str, Any] = evaluator(cast(SparseEncoder, sparse_encoder))

        nanobeir_sparsity_stats: dict[str, float] = {}
        raw_sparsity_stats: Any = getattr(evaluator, "sparsity_stats", None)
        if isinstance(raw_sparsity_stats, dict):
            for metric_key in (
                "query_active_dims",
                "query_sparsity_ratio",
                "corpus_active_dims",
                "corpus_sparsity_ratio",
                "avg_flops",
            ):
                metric_value_any: Any = raw_sparsity_stats.get(metric_key)
                try:
                    nanobeir_sparsity_stats[metric_key] = float(metric_value_any)
                except (TypeError, ValueError):
                    continue

        metric_name: str
        metric_value: Any
        logged_metrics: dict[str, float] = {}
        for metric_name, metric_value in results.items():
            log_if_rank_zero(self.logger, f"NanoBEIR {metric_name}: {metric_value}")
            try:
                metric_float: float = float(metric_value)
            except (TypeError, ValueError):
                continue
            logged_metrics[f"val_nanobeir_{metric_name}"] = metric_float
        for metric_name, metric_value in nanobeir_sparsity_stats.items():
            logged_metrics[f"val_nanobeir_{metric_name}"] = metric_value
        if "query_active_dims" in nanobeir_sparsity_stats:
            logged_metrics["val_nanobeir_query_active_logits"] = nanobeir_sparsity_stats[
                "query_active_dims"
            ]
        if "corpus_active_dims" in nanobeir_sparsity_stats:
            corpus_active_dims: float = nanobeir_sparsity_stats["corpus_active_dims"]
            logged_metrics["val_nanobeir_doc_active_dims"] = corpus_active_dims
            logged_metrics["val_nanobeir_doc_active_logits"] = corpus_active_dims
        if "corpus_sparsity_ratio" in nanobeir_sparsity_stats:
            logged_metrics["val_nanobeir_doc_sparsity_ratio"] = nanobeir_sparsity_stats[
                "corpus_sparsity_ratio"
            ]
        if logged_metrics:
            log_dict_fn(logged_metrics)
        if self.save_json:
            output_path: str = os.path.join(
                log_dir, f"nanobeir_metrics_step{int(global_step)}.json"
            )
            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(results, json_file, indent=2)
            log_if_rank_zero(self.logger, f"Saved NanoBEIR metrics to {output_path}")
