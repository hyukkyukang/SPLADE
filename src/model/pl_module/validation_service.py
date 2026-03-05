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
        q_reps: torch.Tensor,
        doc_reps: torch.Tensor,
        pos_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        world_size: int,
        global_rank: int,
    ) -> None:
        if self._metric_collection is None:
            return
        # Validation metrics use each query's own candidate pool only.
        pairwise_scores: torch.Tensor = torch.bmm(
            doc_reps.float(), q_reps.float().unsqueeze(2)
        ).squeeze(2)

        batch_size: int = int(pairwise_scores.shape[0])
        base_offset: int = self._query_offset
        self._query_offset += batch_size

        for i in range(batch_size):
            valid_mask: torch.Tensor = doc_mask[i]
            if not valid_mask.any():
                continue
            metric_device: torch.device = q_reps.device
            scores: torch.Tensor = (
                pairwise_scores[i][valid_mask]
                .float()
                .detach()
                .to(metric_device)
            )
            targets: torch.Tensor = (
                pos_mask[i][valid_mask]
                .float()
                .detach()
                .to(metric_device)
            )
            global_query_idx: int = global_rank + world_size * (base_offset + i)
            indexes: torch.Tensor = torch.full(
                (scores.shape[0],),
                global_query_idx,
                dtype=torch.long,
                device=scores.device,
            )
            if self.backend == "torchmetrics":
                metric_collection = self._metric_collection
                if not isinstance(metric_collection, RetrievalMetrics):
                    raise TypeError("Expected RetrievalMetrics for torchmetrics backend.")
                metric_collection.append(scores, targets, indexes)
                continue

            metric_collection_custom = self._metric_collection
            if not isinstance(metric_collection_custom, ValidationRetrievalMetrics):
                raise TypeError(
                    "Expected ValidationRetrievalMetrics for custom backend."
                )
            local_doc_indexes: torch.Tensor = torch.arange(
                scores.shape[0], dtype=torch.long, device=scores.device
            )
            tie_break: torch.Tensor = self._build_tie_break_values(
                global_query_idx=global_query_idx,
                local_doc_indexes=local_doc_indexes,
                device=scores.device,
            )
            metric_collection_custom.append(scores, targets, indexes, tie_break)

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
