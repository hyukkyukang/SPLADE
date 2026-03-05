from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn


class ValidationRetrievalMetrics(nn.Module):
    """Deterministic retrieval metrics for training-time validation.

    This backend avoids torchmetrics retrieval tie-pathologies and does not
    mask labels based on score sign (e.g., preds > 0).
    """

    def __init__(
        self,
        *,
        k_list: Sequence[int],
        tie_break_seed: int = 0,
    ) -> None:
        super().__init__()
        self.k_list: list[int] = list(dict.fromkeys(int(k) for k in k_list if int(k) > 0))
        if not self.k_list:
            raise ValueError("k_list must contain at least one positive integer.")
        self.tie_break_seed: int = int(tie_break_seed)

        # Track device even when this rank has no accumulated samples.
        self.register_buffer("_device_ref", torch.tensor(0), persistent=False)

        self._accumulated_preds: list[torch.Tensor] = []
        self._accumulated_targets: list[torch.Tensor] = []
        self._accumulated_indexes: list[torch.Tensor] = []
        self._accumulated_tie_break: list[torch.Tensor] = []

    @property
    def has_accumulated_data(self) -> bool:
        return len(self._accumulated_preds) > 0

    def reset(self) -> None:
        self._accumulated_preds = []
        self._accumulated_targets = []
        self._accumulated_indexes = []
        self._accumulated_tie_break = []

    def append(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        indexes: torch.Tensor,
        tie_break: torch.Tensor,
    ) -> None:
        self._accumulated_preds.append(preds)
        self._accumulated_targets.append(target)
        self._accumulated_indexes.append(indexes)
        self._accumulated_tie_break.append(tie_break)

    def gather(
        self,
        world_size: int = 1,
        all_gather_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> bool:
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
            local_tie_break: torch.Tensor = torch.empty(
                0, device=self._device_ref.device, dtype=torch.long
            )
        else:
            local_preds = torch.cat(self._accumulated_preds).float()
            local_targets = torch.cat(self._accumulated_targets).float()
            local_indexes = torch.cat(self._accumulated_indexes).long()
            local_tie_break = torch.cat(self._accumulated_tie_break).long()

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
                            pad_len,
                            device=local_preds.device,
                            dtype=local_preds.dtype,
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
                local_tie_break = torch.cat(
                    [
                        local_tie_break,
                        torch.zeros(
                            pad_len,
                            device=local_tie_break.device,
                            dtype=local_tie_break.dtype,
                        ),
                    ]
                )

            gathered_preds: torch.Tensor = all_gather_fn(local_preds)
            gathered_targets: torch.Tensor = all_gather_fn(local_targets)
            gathered_indexes: torch.Tensor = all_gather_fn(local_indexes)
            gathered_tie_break: torch.Tensor = all_gather_fn(local_tie_break)

            preds_list: list[torch.Tensor] = []
            targets_list: list[torch.Tensor] = []
            indexes_list: list[torch.Tensor] = []
            tie_break_list: list[torch.Tensor] = []
            rank_idx: int
            size: int
            for rank_idx, size in enumerate(all_sizes.tolist()):
                size_int: int = int(size)
                if size_int <= 0:
                    continue
                preds_list.append(gathered_preds[rank_idx, :size_int])
                targets_list.append(gathered_targets[rank_idx, :size_int])
                indexes_list.append(gathered_indexes[rank_idx, :size_int])
                tie_break_list.append(gathered_tie_break[rank_idx, :size_int])

            if not preds_list:
                return False
            all_preds: torch.Tensor = torch.cat(preds_list)
            all_targets: torch.Tensor = torch.cat(targets_list)
            all_indexes: torch.Tensor = torch.cat(indexes_list)
            all_tie_break: torch.Tensor = torch.cat(tie_break_list)
        else:
            if local_preds.numel() == 0:
                return False
            all_preds = local_preds
            all_targets = local_targets
            all_indexes = local_indexes
            all_tie_break = local_tie_break

        self._accumulated_preds = [all_preds]
        self._accumulated_targets = [all_targets]
        self._accumulated_indexes = [all_indexes]
        self._accumulated_tie_break = [all_tie_break]
        return True

    def compute(self) -> dict[str, torch.Tensor]:
        if not self.has_accumulated_data:
            return {}
        preds: torch.Tensor = torch.cat(self._accumulated_preds).float()
        targets: torch.Tensor = torch.cat(self._accumulated_targets).float()
        indexes: torch.Tensor = torch.cat(self._accumulated_indexes).long()
        tie_break: torch.Tensor = torch.cat(self._accumulated_tie_break).long()

        preds_np: np.ndarray = preds.detach().cpu().numpy()
        targets_np: np.ndarray = targets.detach().cpu().numpy()
        indexes_np: np.ndarray = indexes.detach().cpu().numpy()
        tie_break_np: np.ndarray = tie_break.detach().cpu().numpy()

        if preds_np.size == 0:
            return {}

        order_by_query: np.ndarray = np.argsort(indexes_np, kind="mergesort")
        preds_np = preds_np[order_by_query]
        targets_np = targets_np[order_by_query]
        indexes_np = indexes_np[order_by_query]
        tie_break_np = tie_break_np[order_by_query]

        metric_sums: dict[str, float] = {}
        query_count: int = 0

        query_ids, counts = np.unique(indexes_np, return_counts=True)
        _ = query_ids
        start: int = 0
        for count in counts:
            end: int = start + int(count)
            q_scores: np.ndarray = preds_np[start:end]
            q_targets: np.ndarray = targets_np[start:end]
            q_tie_break: np.ndarray = tie_break_np[start:end]
            start = end

            # Primary: score descending. Secondary: deterministic tie-break key.
            rank_order: np.ndarray = np.lexsort((q_tie_break, -q_scores))
            ranked_targets: np.ndarray = q_targets[rank_order] > 0.0

            num_pos: int = int(ranked_targets.sum())
            query_count += 1

            for k in self.k_list:
                metric_key_prefix: str = str(k)
                top_targets: np.ndarray = ranked_targets[:k]
                metric_sums.setdefault(f"MRR_{metric_key_prefix}", 0.0)
                metric_sums.setdefault(f"Recall_{metric_key_prefix}", 0.0)
                metric_sums.setdefault(f"nDCG_{metric_key_prefix}", 0.0)

                first_pos: np.ndarray = np.flatnonzero(top_targets)
                rr_k: float = 0.0
                if first_pos.size > 0:
                    rr_k = 1.0 / float(int(first_pos[0]) + 1)
                metric_sums[f"MRR_{metric_key_prefix}"] += rr_k

                recall_k: float = 0.0
                if num_pos > 0:
                    recall_k = float(top_targets.sum()) / float(num_pos)
                metric_sums[f"Recall_{metric_key_prefix}"] += recall_k

                ndcg_k: float = 0.0
                if num_pos > 0:
                    gains: np.ndarray = top_targets.astype(np.float64)
                    discounts: np.ndarray = 1.0 / np.log2(
                        np.arange(2, gains.shape[0] + 2, dtype=np.float64)
                    )
                    dcg: float = float((gains * discounts).sum())
                    ideal_len: int = min(num_pos, int(k))
                    if ideal_len > 0:
                        ideal_discounts: np.ndarray = 1.0 / np.log2(
                            np.arange(2, ideal_len + 2, dtype=np.float64)
                        )
                        idcg: float = float(ideal_discounts.sum())
                        if idcg > 0.0:
                            ndcg_k = dcg / idcg
                metric_sums[f"nDCG_{metric_key_prefix}"] += ndcg_k

        if query_count <= 0:
            return {}

        device: torch.device = self._device_ref.device
        metrics: dict[str, torch.Tensor] = {}
        name: str
        total: float
        for name, total in metric_sums.items():
            metrics[name] = torch.tensor(
                total / float(query_count),
                device=device,
                dtype=torch.float32,
            )
        return metrics

