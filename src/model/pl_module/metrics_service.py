import lightning as L
import torch

_SPARSE_TRAINING_METRIC_NAMES: tuple[tuple[str, str], ...] = (
    ("distill_loss", "train_distill_loss"),
    ("pairwise_loss", "train_pairwise_loss"),
    ("sigmoid_pos_loss", "train_sigmoid_pos_loss"),
    ("sigmoid_neg_loss", "train_sigmoid_neg_loss"),
    ("sigmoid_logit_scale", "train_sigmoid_logit_scale"),
    ("sigmoid_bias", "train_sigmoid_bias"),
    ("sigmoid_pos_score_mean", "train_sigmoid_pos_score_mean"),
    ("sigmoid_neg_score_mean", "train_sigmoid_neg_score_mean"),
    ("sigmoid_pos_margin_mean", "train_sigmoid_pos_margin_mean"),
    ("sigmoid_neg_margin_mean", "train_sigmoid_neg_margin_mean"),
    ("in_batch_loss", "train_in_batch_loss"),
    ("q_reg", "train_q_reg"),
    ("d_reg", "train_d_reg"),
    ("reg_query_lambda", "train_reg_query_lambda"),
    ("reg_doc_lambda", "train_reg_doc_lambda"),
    ("q_rep_magnitude", "train_q_rep_magnitude"),
    ("doc_rep_magnitude", "train_doc_rep_magnitude"),
)

_STEP_ONLY_DIAGNOSTIC_METRIC_NAMES: tuple[str, ...] = (
    "reg_lambda_scale",
    "reg_q_contrib",
    "reg_d_contrib",
    "reg_total_contrib",
    "pre_reg_loss",
    "reg_total_frac_loss",
    "reg_total_frac_pre_reg",
    "q_active_dims",
    "doc_active_dims",
    "q_sparsity_ratio",
    "doc_sparsity_ratio",
    "q_flops_proxy_sum_equiv",
    "d_flops_proxy_sum_equiv",
    "q_flops_proxy_mean_equiv",
    "d_flops_proxy_mean_equiv",
)


class TrainingMetricsService:
    """Own logging policy for training-step metrics."""

    def __init__(
        self,
        *,
        step_only_metric_log_interval: int = 100,
        validation_diagnostics_enabled: bool = True,
        validation_diagnostics_log_interval: int = 1,
    ) -> None:
        self.step_only_metric_log_interval: int = max(
            int(step_only_metric_log_interval), 1
        )
        self.validation_diagnostics_enabled: bool = bool(
            validation_diagnostics_enabled
        )
        self.validation_diagnostics_log_interval: int = max(
            int(validation_diagnostics_log_interval), 1
        )

    def _log_sparse_training_metric(
        self,
        module: L.LightningModule,
        *,
        name: str,
        value: torch.Tensor,
        should_log_step_only: bool,
    ) -> None:
        if not should_log_step_only:
            return
        module.log(
            name,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

    def should_compute_step_only_metrics(self, module: L.LightningModule) -> bool:
        interval: int = self.step_only_metric_log_interval
        if interval <= 1:
            return True
        global_step: int = int(getattr(module, "global_step", 0))
        return global_step % interval == 0

    def should_compute_validation_diagnostics(self, *, batch_idx: int) -> bool:
        if not self.validation_diagnostics_enabled:
            return False
        interval: int = self.validation_diagnostics_log_interval
        if interval <= 1:
            return True
        return int(batch_idx) % interval == 0

    def _log_sparse_training_metric_if_present(
        self,
        module: L.LightningModule,
        *,
        metrics: dict[str, torch.Tensor],
        metric_name: str,
        log_name: str,
    ) -> None:
        value: torch.Tensor | None = metrics.get(metric_name)
        if value is None:
            return
        self._log_sparse_training_metric(
            module,
            name=log_name,
            value=value.detach(),
            should_log_step_only=True,
        )

    def log_training_metrics(
        self, module: L.LightningModule, metrics: dict[str, torch.Tensor]
    ) -> None:
        module.log(
            "train_loss",
            metrics["loss"].detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if not self.should_compute_step_only_metrics(module):
            return

        metric_name: str
        log_name: str
        for metric_name, log_name in _SPARSE_TRAINING_METRIC_NAMES:
            self._log_sparse_training_metric_if_present(
                module,
                metrics=metrics,
                metric_name=metric_name,
                log_name=log_name,
            )

        name: str
        value: torch.Tensor
        for name, value in metrics.items():
            if not name.startswith("distill_") or name == "distill_loss":
                continue
            self._log_sparse_training_metric(
                module,
                name=f"train_{name}",
                value=value.detach(),
                should_log_step_only=True,
            )

        for metric_name in _STEP_ONLY_DIAGNOSTIC_METRIC_NAMES:
            self._log_sparse_training_metric_if_present(
                module,
                metrics=metrics,
                metric_name=metric_name,
                log_name=f"train_{metric_name}",
            )
