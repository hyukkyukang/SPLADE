from typing import Any

import torch
from omegaconf import DictConfig

from src.metric.retrieval import RetrievalMetrics, resolve_k_list
from src.metric.validation_retrieval import ValidationRetrievalMetrics


class ValidationMetricsAccumulator:
    """Accumulate per-query validation metrics for reranking-style evaluation."""

    def __init__(self, *, dataset_name: str, metrics_cfg: DictConfig) -> None:
        self.metrics_cfg = metrics_cfg
        self.enabled: bool = bool(metrics_cfg.enabled)
        self._query_offset: int = 0
        backend_value: Any = metrics_cfg.get("backend", "custom")
        self.backend: str = str(backend_value).strip().lower()
        if self.backend not in ("custom", "torchmetrics"):
            raise ValueError(
                "training.validation_metrics.backend must be one of "
                "['custom', 'torchmetrics'], got: "
                f"{backend_value!r}"
            )
        self.tie_break_seed: int = int(metrics_cfg.get("tie_break_seed", 0))
        self._metric_collection: RetrievalMetrics | ValidationRetrievalMetrics | None = None
        if self.enabled:
            k_list: list[int] = resolve_k_list(metrics_cfg.k_list)
            if self.backend == "torchmetrics":
                self._metric_collection = RetrievalMetrics(
                    dataset_name=dataset_name,
                    k_list=k_list,
                    sync_on_compute=False,
                )
            else:
                self._metric_collection = ValidationRetrievalMetrics(
                    k_list=k_list,
                    tie_break_seed=self.tie_break_seed,
                )

    @property
    def has_collection(self) -> bool:
        return self._metric_collection is not None

    def _build_tie_break_values(
        self,
        *,
        global_query_idx: int,
        local_doc_indexes: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Build deterministic pseudo-random keys to break score ties."""
        query_component: torch.Tensor = torch.full(
            local_doc_indexes.shape,
            int(global_query_idx + self.tie_break_seed),
            dtype=torch.long,
            device=device,
        )
        # 64-bit integer mixing for deterministic, backend-consistent ordering.
        mixed: torch.Tensor = (
            local_doc_indexes * 6364136223846793005
            + query_component * 1442695040888963407
            + 0x9E3779B97F4A7C15
        )
        # Keep positive values for lexsort secondary key.
        return mixed & 0x7FFFFFFFFFFFFFFF

    def _build_flat_tie_break_values(
        self,
        *,
        global_query_indexes: torch.Tensor,
        local_doc_indexes: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        query_component: torch.Tensor = global_query_indexes.to(
            dtype=torch.long, device=device
        ) + int(self.tie_break_seed)
        mixed: torch.Tensor = (
            local_doc_indexes.to(dtype=torch.long, device=device)
            * 6364136223846793005
            + query_component * 1442695040888963407
            + 0x9E3779B97F4A7C15
        )
        return mixed & 0x7FFFFFFFFFFFFFFF

    @staticmethod
    def _build_flat_local_doc_indexes(
        counts: torch.Tensor, *, device: torch.device
    ) -> torch.Tensor:
        counts_long: torch.Tensor = counts.to(dtype=torch.long, device=device)
        total_docs: int = int(counts_long.sum().item())
        if total_docs <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        segment_starts: torch.Tensor = counts_long.cumsum(0) - counts_long
        flat_positions: torch.Tensor = torch.arange(
            total_docs, dtype=torch.long, device=device
        )
        repeated_starts: torch.Tensor = segment_starts.repeat_interleave(counts_long)
        return flat_positions - repeated_starts

    def on_validation_start(self, device: torch.device) -> None:
        if self._metric_collection is None:
            return
        self._query_offset = 0
        self._metric_collection.reset()
        # Ensure metric buffers live on the same device as predictions.
        self._metric_collection.to(device)

    def append_batch(
        self,
        *,
        pairwise_scores: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        world_size: int,
        global_rank: int,
    ) -> None:
        if self._metric_collection is None:
            return
        metric_device: torch.device = pairwise_scores.device
        batch_size: int = int(pairwise_scores.shape[0])
        base_offset: int = self._query_offset
        self._query_offset += batch_size

        valid_query_mask: torch.Tensor = doc_mask.any(dim=1)
        if not bool(valid_query_mask.any()):
            return

        valid_doc_mask: torch.Tensor = doc_mask[valid_query_mask]
        valid_pos_mask: torch.Tensor = pos_mask[valid_query_mask]
        valid_pairwise_scores: torch.Tensor = pairwise_scores[valid_query_mask]
        valid_counts: torch.Tensor = valid_doc_mask.sum(dim=1, dtype=torch.long)
        query_positions: torch.Tensor = torch.arange(
            batch_size, dtype=torch.long, device=metric_device
        )[valid_query_mask]
        global_query_indexes: torch.Tensor = (
            global_rank + world_size * (base_offset + query_positions)
        ).to(dtype=torch.long)
        flat_scores: torch.Tensor = (
            valid_pairwise_scores[valid_doc_mask].float().detach().to(metric_device)
        )
        flat_targets: torch.Tensor = (
            valid_pos_mask[valid_doc_mask].float().detach().to(metric_device)
        )
        flat_indexes: torch.Tensor = global_query_indexes.repeat_interleave(valid_counts)

        if self.backend == "torchmetrics":
            metric_collection = self._metric_collection
            if not isinstance(metric_collection, RetrievalMetrics):
                raise TypeError("Expected RetrievalMetrics for torchmetrics backend.")
            metric_collection.append(flat_scores, flat_targets, flat_indexes)
            return

        metric_collection_custom = self._metric_collection
        if not isinstance(metric_collection_custom, ValidationRetrievalMetrics):
            raise TypeError("Expected ValidationRetrievalMetrics for custom backend.")
        flat_local_doc_indexes: torch.Tensor = self._build_flat_local_doc_indexes(
            valid_counts,
            device=metric_device,
        )
        flat_tie_break: torch.Tensor = self._build_flat_tie_break_values(
            global_query_indexes=flat_indexes,
            local_doc_indexes=flat_local_doc_indexes,
            device=metric_device,
        )
        metric_collection_custom.append(
            flat_scores,
            flat_targets,
            flat_indexes,
            flat_tie_break,
        )

    def finalize_epoch(
        self,
        *,
        world_size: int,
        all_gather_fn: Any | None,
    ) -> tuple[bool, dict[str, torch.Tensor]]:
        if self._metric_collection is None:
            return False, {}
        has_data: bool = self._metric_collection.gather(
            world_size=world_size,
            all_gather_fn=all_gather_fn,
        )
        if not has_data:
            return False, {}
        metrics: dict[str, torch.Tensor] = self._metric_collection.compute()
        filtered_metrics: dict[str, torch.Tensor] = {
            f"val_{name}": value
            for name, value in metrics.items()
            if name.startswith(("nDCG_", "MRR_", "Recall_"))
        }
        self._metric_collection.reset()
        return True, filtered_metrics
