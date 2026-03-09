import lightning as L
import torch


class TrainingMetricsService:
    """Own logging policy for training-step metrics."""

    def __init__(
        self,
        *,
        step_only_metric_log_interval: int = 1,
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

    def log_training_metrics(
        self, module: L.LightningModule, metrics: dict[str, torch.Tensor]
    ) -> None:
        detached_metrics: dict[str, torch.Tensor] = {
            name: value.detach() for name, value in metrics.items()
        }
        should_log_step_only: bool = self.should_compute_step_only_metrics(module)
        module.log(
            "train_loss",
            detached_metrics["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if "distill_loss" in detached_metrics:
            module.log(
                "train_distill_loss",
                detached_metrics["distill_loss"],
                on_step=True,
                on_epoch=True,
            )
        for name, value in detached_metrics.items():
            if not name.startswith("distill_") or name == "distill_loss":
                continue
            module.log(
                f"train_{name}",
                value,
                on_step=True,
                on_epoch=True,
            )
        if "pairwise_loss" in detached_metrics:
            module.log(
                "train_pairwise_loss",
                detached_metrics["pairwise_loss"],
                on_step=True,
                on_epoch=True,
            )
        if "sigmoid_pos_loss" in detached_metrics:
            module.log(
                "train_sigmoid_pos_loss",
                detached_metrics["sigmoid_pos_loss"],
                on_step=True,
                on_epoch=True,
            )
        if "sigmoid_neg_loss" in detached_metrics:
            module.log(
                "train_sigmoid_neg_loss",
                detached_metrics["sigmoid_neg_loss"],
                on_step=True,
                on_epoch=True,
            )
        if "sigmoid_logit_scale" in detached_metrics:
            module.log(
                "train_sigmoid_logit_scale",
                detached_metrics["sigmoid_logit_scale"],
                on_step=True,
                on_epoch=True,
            )
        if "sigmoid_bias" in detached_metrics:
            module.log(
                "train_sigmoid_bias",
                detached_metrics["sigmoid_bias"],
                on_step=True,
                on_epoch=True,
            )
        if "sigmoid_pos_score_mean" in detached_metrics:
            module.log(
                "train_sigmoid_pos_score_mean",
                detached_metrics["sigmoid_pos_score_mean"],
                on_step=True,
                on_epoch=True,
            )
        if "sigmoid_neg_score_mean" in detached_metrics:
            module.log(
                "train_sigmoid_neg_score_mean",
                detached_metrics["sigmoid_neg_score_mean"],
                on_step=True,
                on_epoch=True,
            )
        if "sigmoid_pos_margin_mean" in detached_metrics:
            module.log(
                "train_sigmoid_pos_margin_mean",
                detached_metrics["sigmoid_pos_margin_mean"],
                on_step=True,
                on_epoch=True,
            )
        if "sigmoid_neg_margin_mean" in detached_metrics:
            module.log(
                "train_sigmoid_neg_margin_mean",
                detached_metrics["sigmoid_neg_margin_mean"],
                on_step=True,
                on_epoch=True,
            )
        if "in_batch_loss" in detached_metrics:
            module.log(
                "train_in_batch_loss",
                detached_metrics["in_batch_loss"],
                on_step=True,
                on_epoch=True,
            )
        if "q_reg" in detached_metrics:
            module.log(
                "train_q_reg",
                detached_metrics["q_reg"],
                on_step=True,
                on_epoch=True,
            )
        if "d_reg" in detached_metrics:
            module.log(
                "train_d_reg",
                detached_metrics["d_reg"],
                on_step=True,
                on_epoch=True,
            )
        if "reg_query_lambda" in detached_metrics:
            module.log(
                "train_reg_query_lambda",
                detached_metrics["reg_query_lambda"],
                on_step=True,
                on_epoch=True,
            )
        if "reg_doc_lambda" in detached_metrics:
            module.log(
                "train_reg_doc_lambda",
                detached_metrics["reg_doc_lambda"],
                on_step=True,
                on_epoch=True,
            )
        if "q_rep_magnitude" in detached_metrics:
            module.log(
                "train_q_rep_magnitude",
                detached_metrics["q_rep_magnitude"],
                on_step=True,
                on_epoch=True,
            )
        if "doc_rep_magnitude" in detached_metrics:
            module.log(
                "train_doc_rep_magnitude",
                detached_metrics["doc_rep_magnitude"],
                on_step=True,
                on_epoch=True,
            )
        for name in (
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
        ):
            if not should_log_step_only:
                continue
            if name not in detached_metrics:
                continue
            module.log(
                f"train_{name}",
                detached_metrics[name],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
