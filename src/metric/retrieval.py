from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalRecall,
)

DEFAULT_K_LIST: list[int] = [1, 5, 10, 50, 100]


def resolve_k_list(k_list: Sequence[int] | None) -> list[int]:
    """Resolve k_list from explicit values or defaults."""
    if k_list is not None and len(k_list) > 0:
        return list(dict.fromkeys(int(k) for k in k_list))
    return list(DEFAULT_K_LIST)


class RetrievalMetrics(MetricCollection):
    """
    Retrieval metrics collection using torchmetrics.

    This mirrors GenZ's approach: accumulate raw tensors during steps, gather once,
    then compute all metrics at epoch end for efficient distributed evaluation.
    """

    def __init__(
        self,
        dataset_name: str = "",
        k_list: Optional[List[int]] = None,
        sync_on_compute: bool = False,
    ) -> None:
        k_list_final: List[int] = k_list if k_list is not None else list(DEFAULT_K_LIST)

        metrics: Dict[str, torch.nn.Module] = self._build_metrics(
            k_list=k_list_final,
            sync_on_compute=sync_on_compute,
        )
        prefix: str = f"{dataset_name}_" if dataset_name else ""
        super().__init__(metrics, prefix=prefix)

        # Track device even when no data is appended on a rank.
        self.register_buffer("_device_ref", torch.tensor(0), persistent=False)

        # Accumulate per-batch data for a single all_gather at epoch end.
        self._accumulated_preds: List[torch.Tensor] = []
        self._accumulated_targets: List[torch.Tensor] = []
        self._accumulated_indexes: List[torch.Tensor] = []

    @property
    def has_accumulated_data(self) -> bool:
        """Return True when at least one batch has been appended."""
        return len(self._accumulated_preds) > 0

    def _build_metrics(
        self,
        k_list: List[int],
        sync_on_compute: bool = False,
    ) -> Dict[str, torch.nn.Module]:
        metrics: Dict[str, torch.nn.Module] = {}
        for k in k_list:
            # sync_on_compute=False: disable automatic distributed sync for performance.
            metrics[f"nDCG_{k}"] = RetrievalNormalizedDCG(
                top_k=k, sync_on_compute=sync_on_compute
            )
            metrics[f"MRR_{k}"] = RetrievalMRR(top_k=k, sync_on_compute=sync_on_compute)
            metrics[f"MAP_{k}"] = RetrievalMAP(top_k=k, sync_on_compute=sync_on_compute)
            metrics[f"Recall_{k}"] = RetrievalRecall(
                top_k=k, sync_on_compute=sync_on_compute
            )
            metrics[f"Success_{k}"] = RetrievalHitRate(
                top_k=k, sync_on_compute=sync_on_compute
            )
        return metrics

    def append(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor
    ) -> None:
        """Accumulate tensors for a single all_gather at epoch end."""
        self._accumulated_preds.append(preds)
        self._accumulated_targets.append(target)
        self._accumulated_indexes.append(indexes)

    def gather(
        self,
        world_size: int = 1,
        all_gather_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> bool:
        """
        Gather accumulated data across ranks and update metrics once.

        All ranks must call this method in distributed settings.
        """
        if not self.has_accumulated_data:
            local_preds: torch.Tensor = torch.empty(
                0, device=self._device_ref.device, dtype=torch.float32
            )
            local_targets: torch.Tensor = torch.empty(
                0, device=self._device_ref.device, dtype=torch.float32
            )
            local_indexes: torch.Tensor = torch.empty(
                0, device=self._device_ref.device, dtype=torch.long
            )
        else:
            local_preds: torch.Tensor = torch.cat(self._accumulated_preds).float()
            local_targets: torch.Tensor = torch.cat(self._accumulated_targets).float()
            local_indexes: torch.Tensor = torch.cat(self._accumulated_indexes).long()

        if world_size > 1 and all_gather_fn is None:
            raise ValueError("all_gather_fn is required when world_size > 1.")

        if world_size > 1 and all_gather_fn is not None:
            size_tensor: torch.Tensor = torch.tensor(
                [local_preds.numel()],
                device=local_preds.device,
                dtype=torch.long,
            )
            all_sizes: torch.Tensor = all_gather_fn(size_tensor).flatten()
            max_size: int = int(all_sizes.max().item())

            if max_size == 0:
                return False

            if local_preds.numel() < max_size:
                pad_len: int = max_size - local_preds.numel()
                local_preds = torch.cat(
                    [
                        local_preds,
                        torch.zeros(
                            pad_len, device=local_preds.device, dtype=local_preds.dtype
                        ),
                    ]
                )
                local_targets = torch.cat(
                    [
                        local_targets,
                        torch.zeros(
                            pad_len,
                            device=local_targets.device,
                            dtype=local_targets.dtype,
                        ),
                    ]
                )
                local_indexes = torch.cat(
                    [
                        local_indexes,
                        torch.zeros(
                            pad_len,
                            device=local_indexes.device,
                            dtype=local_indexes.dtype,
                        ),
                    ]
                )

            gathered_preds: torch.Tensor = all_gather_fn(local_preds)
            gathered_targets: torch.Tensor = all_gather_fn(local_targets)
            gathered_indexes: torch.Tensor = all_gather_fn(local_indexes)

            preds_list: list[torch.Tensor] = []
            targets_list: list[torch.Tensor] = []
            indexes_list: list[torch.Tensor] = []
            for rank_idx, size in enumerate(all_sizes.tolist()):
                size_int: int = int(size)
                if size_int <= 0:
                    continue
                preds_list.append(gathered_preds[rank_idx, :size_int])
                targets_list.append(gathered_targets[rank_idx, :size_int])
                indexes_list.append(gathered_indexes[rank_idx, :size_int])

            if not preds_list:
                return False

            all_preds: torch.Tensor = torch.cat(preds_list)
            all_targets: torch.Tensor = torch.cat(targets_list)
            all_indexes: torch.Tensor = torch.cat(indexes_list)
        else:
            if local_preds.numel() == 0:
                return False
            all_preds: torch.Tensor = local_preds
            all_targets: torch.Tensor = local_targets
            all_indexes: torch.Tensor = local_indexes

        self.update(all_preds, all_targets, all_indexes)
        return True

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        indexes: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Update all metrics with appropriate label types."""
        _ = args, kwargs
        binary_metric_names: List[str] = ["MRR", "MAP", "Recall", "Success"]
        target_bool: torch.Tensor = target.bool()

        for name, metric in self.items():
            requires_binary: bool = any(
                metric_name in name for metric_name in binary_metric_names
            )
            if requires_binary:
                metric.update(preds, target_bool, indexes)
            else:
                metric.update(preds, target, indexes)

    def reset(self) -> None:
        """Reset metrics and clear accumulated data."""
        super().reset()
        self._accumulated_preds = []
        self._accumulated_targets = []
        self._accumulated_indexes = []
