from typing import Any

import torch
from omegaconf import DictConfig

from src.metric.retrieval import RetrievalMetrics, resolve_k_list


class ValidationMetricsAccumulator:
    """Accumulate per-query validation metrics for reranking-style evaluation."""

    def __init__(self, *, dataset_name: str, metrics_cfg: DictConfig) -> None:
        self.metrics_cfg = metrics_cfg
        self.enabled: bool = bool(metrics_cfg.enabled)
        self._query_offset: int = 0
        self._metric_collection: RetrievalMetrics | None = None
        if self.enabled:
            k_list: list[int] = resolve_k_list(metrics_cfg.k_list)
            self._metric_collection = RetrievalMetrics(
                dataset_name=dataset_name,
                k_list=k_list,
                sync_on_compute=False,
            )

    @property
    def has_collection(self) -> bool:
        return self._metric_collection is not None

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
            self._metric_collection.append(scores, targets, indexes)

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
