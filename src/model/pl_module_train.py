from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
from torch import nn

from src.metric.beir_evaluator import BEIREvaluator
from src.model.losses import (
    distillation_loss,
    multi_positive_contrastive_loss,
    regularization_loss,
)
from src.model.splade import SpladeModel
from src.tokenization.tokenizer import build_tokenizer
from src.utils import is_rank_zero, log_if_rank_zero


class SPLADETrainingModule(L.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        dtype = torch.float16 if cfg.model.dtype == "float16" else None
        self.model = SpladeModel(
            model_name=cfg.model.huggingface_name,
            query_pooling=cfg.model.query_pooling,
            doc_pooling=cfg.model.doc_pooling,
            sparse_activation=cfg.model.sparse_activation,
            attn_implementation=cfg.model.attn_implementation,
            dtype=dtype,
            normalize=cfg.model.normalize,
        )

        self.temperature = float(cfg.training.temperature)
        self.distill_cfg = cfg.training.distill
        self.reg_cfg = cfg.training.regularization
        self.loss_cfg = cfg.training.loss
        self.beir_cfg = cfg.training.beir_eval
        self.beir_evaluator: BEIREvaluator | None = None
        self._beir_tokenizer = None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        q = self.model.encode_queries(batch["query_input_ids"], batch["query_attention_mask"])
        d = self.model.encode_docs(batch["doc_input_ids"], batch["doc_attention_mask"])
        return {"q": q, "d": d}

    def _compute_scores(
        self, q_reps: torch.Tensor, d_reps: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("bv,bnv->bn", q_reps, d_reps)

    def _training_step_shared(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> dict[str, torch.Tensor]:
        q_reps = self.model.encode_queries(
            batch["query_input_ids"], batch["query_attention_mask"]
        )
        doc_input_ids = batch["doc_input_ids"]
        doc_attention_mask = batch["doc_attention_mask"]

        bsz, doc_count, seq_len = doc_input_ids.shape
        flat_docs = doc_input_ids.view(bsz * doc_count, seq_len)
        flat_masks = doc_attention_mask.view(bsz * doc_count, seq_len)

        flat_doc_reps = self.model.encode_docs(flat_docs, flat_masks)
        doc_reps = flat_doc_reps.view(bsz, doc_count, -1)

        scores = self._compute_scores(q_reps, doc_reps)

        pos_mask = batch["pos_mask"]
        doc_mask = batch["doc_mask"]

        loss = multi_positive_contrastive_loss(
            scores, pos_mask, doc_mask, temperature=self.temperature
        )

        metrics: dict[str, torch.Tensor] = {"loss": loss}

        if self.distill_cfg.enabled:
            teacher_scores = batch["teacher_scores"]
            distill = distillation_loss(
                scores,
                teacher_scores,
                doc_mask,
                loss_type=self.distill_cfg.loss,
            )
            loss = loss + self.distill_cfg.weight * distill
            metrics["distill_loss"] = distill

        if self.reg_cfg.query_weight > 0:
            q_reg = regularization_loss(
                q_reps, self.reg_cfg.type, self.reg_cfg.paper_faithful
            )
            loss = loss + self.reg_cfg.query_weight * q_reg
            metrics["q_reg"] = q_reg

        if self.reg_cfg.doc_weight > 0:
            flat_doc_mask = doc_mask.view(-1)
            if flat_doc_mask.any():
                doc_reg = regularization_loss(
                    flat_doc_reps[flat_doc_mask],
                    self.reg_cfg.type,
                    self.reg_cfg.paper_faithful,
                )
                loss = loss + self.reg_cfg.doc_weight * doc_reg
                metrics["d_reg"] = doc_reg

        metrics["loss"] = loss

        if stage != "train":
            metrics.update(self._compute_mrr(scores, pos_mask, doc_mask, k=10))

        return metrics

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        metrics = self._training_step_shared(batch, stage="train")
        self.log("train_loss", metrics["loss"], on_step=True, on_epoch=True, prog_bar=True)
        if "distill_loss" in metrics:
            self.log("train_distill_loss", metrics["distill_loss"], on_step=True, on_epoch=True)
        if "q_reg" in metrics:
            self.log("train_q_reg", metrics["q_reg"], on_step=True, on_epoch=True)
        if "d_reg" in metrics:
            self.log("train_d_reg", metrics["d_reg"], on_step=True, on_epoch=True)
        return metrics["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        metrics = self._training_step_shared(batch, stage="val")
        batch_size = batch["query_input_ids"].shape[0]
        self.log(
            "val_loss",
            metrics["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        if "mrr10" in metrics:
            self.log(
                "val_mrr10",
                metrics["mrr10"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

        if self.cfg.training.scheduler == "linear":
            from transformers import get_linear_schedule_with_warmup

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.training.warmup_steps,
                num_training_steps=self.cfg.training.max_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer

    def on_fit_start(self) -> None:
        if self.beir_cfg.enabled:
            self._beir_tokenizer = build_tokenizer(self.cfg.model.huggingface_name)
            self.beir_evaluator = BEIREvaluator(
                model=self.model,
                tokenizer=self._beir_tokenizer,
                max_query_length=self.cfg.train_dataset.max_query_length,
                max_doc_length=self.cfg.train_dataset.max_doc_length,
                batch_size=self.cfg.training.eval_batch_size,
                device=self.device,
            )

    def on_validation_epoch_end(self) -> None:
        if not self.beir_cfg.enabled or self.beir_evaluator is None:
            return
        if not is_rank_zero():
            return
        for dataset_name in self.beir_cfg.datasets:
            if self.beir_cfg.use_hf:
                metrics = self.beir_evaluator.evaluate_hf(
                    hf_name=f"BeIR/{dataset_name}",
                    split=self.beir_cfg.split,
                    metrics=self.beir_cfg.metrics,
                    top_k=100,
                    sample_size=self.beir_cfg.sample_size,
                    max_docs=self.beir_cfg.max_docs,
                    cache_dir=self.beir_cfg.hf_cache_dir,
                )
            else:
                corpus_path = f"{self.beir_cfg.data_dir}/{dataset_name}/corpus.jsonl"
                queries_path = f"{self.beir_cfg.data_dir}/{dataset_name}/queries.jsonl"
                qrels_path = f"{self.beir_cfg.data_dir}/{dataset_name}/qrels/{self.beir_cfg.split}.tsv"
                metrics = self.beir_evaluator.evaluate(
                    corpus_path=corpus_path,
                    queries_path=queries_path,
                    qrels_path=qrels_path,
                    metrics=self.beir_cfg.metrics,
                    top_k=100,
                    sample_size=self.beir_cfg.sample_size,
                    max_docs=self.beir_cfg.max_docs,
                )
            for metric_name, value in metrics.items():
                key = f"beir_{dataset_name}_{metric_name}"
                self.log(key, value, on_epoch=True, prog_bar=False, sync_dist=False)

    @staticmethod
    def _compute_mrr(
        scores: torch.Tensor, pos_mask: torch.Tensor, doc_mask: torch.Tensor, k: int
    ) -> dict[str, torch.Tensor]:
        scores = scores.masked_fill(~doc_mask, -1e9)
        topk = torch.topk(scores, k=min(k, scores.size(1)), dim=1).indices
        mrrs = []
        for i in range(scores.size(0)):
            rank = 0.0
            for rank_idx, doc_idx in enumerate(topk[i], start=1):
                if pos_mask[i, doc_idx]:
                    rank = 1.0 / rank_idx
                    break
            mrrs.append(rank)
        mrr_tensor = torch.tensor(mrrs, device=scores.device, dtype=scores.dtype)
        return {"mrr10": mrr_tensor.mean()}
