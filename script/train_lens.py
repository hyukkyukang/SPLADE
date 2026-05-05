"""Entrypoint for paper-faithful LENS training on bge-full-data.

Kept separate from ``script/train.py`` because the LENS training step needs
the custom ``LENSTrainingModule`` / ``LENSTrainDataModule`` plumbing. Invoke
via::

    bash script/etc/launch_lens.sh train_lens_official \\
        training.tag=lens_d4000_run01 \\
        checkpoint.dirpath=${LENS_CHECKPOINTS}/lens_d4000_run01
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, MLFlowLogger
from lightning.pytorch.profilers import Profiler, PyTorchProfiler
from omegaconf import DictConfig, OmegaConf

from config.path import ABS_CONFIG_DIR
from src.data.pl_module.lens_train import LENSTrainDataModule
from src.model.pl_module.lens_encoder import LENSBiEncoder
from src.model.pl_module.lens_training_module import LENSTrainingModule
from src.utils.compact_head import swap_clustered_head_from_artifact
from src.utils.lens_official_loader import (
    load_official_lens as load_official_lens_weights,
)
from src.utils.logging import get_logger
from src.utils.peft import apply_peft_adapter, resolve_peft_settings
from src.utils.script_setup import (
    configure_default_entrypoint_environment,
    initialize_run,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_default_entrypoint_environment(
    load_env=True,
    set_matmul_precision=True,
)


def _build_encoder(cfg: DictConfig) -> LENSBiEncoder:
    """Build the encoder on CPU; Lightning's strategy moves it to the per-rank GPU."""
    model_cfg: DictConfig = cfg.model
    source: str = str(model_cfg.huggingface_name)
    dtype: torch.dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(str(model_cfg.get("dtype", "bfloat16")), torch.bfloat16)

    if bool(model_cfg.get("load_official_lens", False)):
        # Load weights on CPU; let Lightning move to per-rank GPU later.
        backbone, _ = load_official_lens_weights(
            source,
            dtype=dtype,
            device=None,
            attn_implementation=str(model_cfg.get("attn_implementation", "sdpa")),
            strict_official_tokenizer=True,
            local_files_only=True,
        )
    else:
        from src.utils.transformers import build_pretrained_model
        backbone = build_pretrained_model(
            source,
            model_class_name=str(model_cfg.get("huggingface_model_class",
                                              "MistralBiForCausalLM")),
            attn_implementation=str(model_cfg.get("attn_implementation", "sdpa")),
            dtype=dtype,
            tie_word_embeddings=bool(model_cfg.get("tie_word_embeddings", False)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        # Mirror the official LENS finetune flow (finetune/load_model.py:82-95):
        # if the artifact dir ships a clustered head, swap it into model.lm_head
        # BEFORE LoRA wrapping. Without this, training learns a 32003-dim sparse
        # output even though the cluster artifact ships a 4000-dim head, and the
        # Phase 0 baseline (which uses the 4000-dim head end-to-end) cannot be
        # reproduced.
        swapped: bool = swap_clustered_head_from_artifact(backbone, source)
        if swapped:
            logger.info(
                "Swapped lm_head with clustered head from %s "
                "(out_features=%d, hidden=%d, config.vocab_size=%d retained)",
                source,
                int(backbone.lm_head.out_features),
                int(backbone.config.hidden_size),
                int(backbone.config.vocab_size),
            )
        else:
            logger.info(
                "No clustered head artifact under %s; keeping original lm_head "
                "(vocab_size=%d).",
                source,
                int(backbone.config.vocab_size),
            )
        # No .to(device) here — Lightning's DDP strategy assigns per-rank GPUs.

    peft_settings = resolve_peft_settings(
        model_cfg.get("peft"),
        model_type=getattr(backbone.config, "model_type", None),
        huggingface_model_class=str(model_cfg.get("huggingface_model_class", "")),
    )
    if peft_settings.enabled:
        backbone, _ = apply_peft_adapter(backbone, settings=peft_settings)
        # PEFT freezes the entire backbone, so the input-embedding output has
        # ``requires_grad=False`` by default. Without explicitly opting in here,
        # the backward graph stops at the embedding output and gradient
        # checkpointing silently retains every layer's intermediate (peak
        # memory ~33 GiB instead of ~14 GiB on Mistral-7B). Standard PEFT
        # recipe documented in HF transformers#26450.
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()

    # Modern HF Mistral.forward no longer auto-disables ``use_cache`` when
    # gradient checkpointing is on — if config.use_cache=True, K/V tensors
    # for every layer are materialized AND retained in the autograd graph.
    # On Mistral-7B that's ~1 GB per sub-batch × 16 sub-batches = ~16 GB of
    # avoidable memory.
    if hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable()
    encoder = LENSBiEncoder(model=backbone, normalize=bool(model_cfg.get("normalize", True)))
    return encoder


def _maybe_compile_encoder(
    encoder: LENSBiEncoder, cfg: DictConfig
) -> LENSBiEncoder:
    """Optionally wrap ``_encode_sub_batch`` with ``torch.compile``.

    Profiling showed each Phase-1 step launches ~1,770 ATen ops/sec, with
    GPU active only 17%. The bottleneck is eager-mode CPU dispatch — fewer
    bigger kernels (via Inductor fusion) is the textbook fix.

    Activation precedence: ``LENS_COMPILE`` env var (if set) overrides
    ``cfg.training.torch_compile``. Mode follows the same precedence.
    """
    env_flag = os.environ.get("LENS_COMPILE")
    cfg_flag = bool(cfg.training.get("torch_compile", False))
    enabled = (env_flag == "1") if env_flag is not None else cfg_flag
    if not enabled:
        return encoder
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is not available; skipping compile.")
        return encoder
    mode = os.environ.get(
        "LENS_COMPILE_MODE",
        str(cfg.training.get("torch_compile_mode", "default")),
    )
    # dynamic=True is critical here. Phase 1 uses ``max_padding=False`` so
    # each sub-batch has a different (sub_batch_size, seq_len) shape after
    # tokenization. Without dynamic, Inductor recompiles on every shape
    # change and the result is *slower* than eager. With dynamic=True the
    # generated kernel handles a range of shapes from one compile.
    # ``LENS_COMPILE_DYNAMIC=0`` opts back to static if you ever want to
    # A/B compare.
    dynamic_env = os.environ.get("LENS_COMPILE_DYNAMIC", "1")
    dynamic = dynamic_env != "0"
    logger.info(
        "Compiling LENSBiEncoder._encode_sub_batch (mode=%s, dynamic=%s). "
        "First step will pay a one-time inductor warmup cost.",
        mode, dynamic,
    )
    encoder._encode_sub_batch = torch.compile(  # type: ignore[method-assign]
        encoder._encode_sub_batch, mode=mode, dynamic=dynamic
    )
    return encoder


def _build_profiler(cfg: DictConfig) -> Profiler | None:
    """Opt-in PyTorch profiler for diagnosing per-step bottlenecks.

    Enable with ``LENS_PROFILE=1``. Captures a 20-step window after a short
    warmup so the trace is small enough to inspect in chrome://tracing.
    """
    if os.environ.get("LENS_PROFILE", "0") != "1":
        return None
    profile_dir = Path(cfg.log_dir) / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    wait = int(os.environ.get("LENS_PROFILE_WAIT", "1"))
    warmup = int(os.environ.get("LENS_PROFILE_WARMUP", "4"))
    active = int(os.environ.get("LENS_PROFILE_ACTIVE", "20"))
    schedule = torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=1
    )
    logger.info(
        "PyTorchProfiler enabled: wait=%d warmup=%d active=%d -> %s",
        wait, warmup, active, profile_dir,
    )
    return PyTorchProfiler(
        dirpath=str(profile_dir),
        filename="lens_trace",
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="train_lens_official")
def main(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)

    encoder: LENSBiEncoder = _build_encoder(cfg)
    encoder = _maybe_compile_encoder(encoder, cfg)

    total_steps: int | None = cfg.training.get("max_steps")
    if total_steps is None or int(total_steps) <= 0:
        total_steps = None
    else:
        total_steps = int(total_steps)

    training_module = LENSTrainingModule(
        encoder=encoder,
        temperature=float(cfg.training.temperature),
        cross_device_negatives=bool(cfg.training.lens_multi_task.cross_device_negatives),
        distill_enabled=bool(cfg.training.distill.enabled),
        distill_weight=float(cfg.training.distill.weight),
        lr=float(cfg.training.lr),
        warmup_steps=int(cfg.training.warmup_steps),
        total_steps=total_steps,
        weight_decay=0.0,
    )
    data_module = LENSTrainDataModule(cfg=cfg)

    checkpoint_dir: str = os.path.join(cfg.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step{step}-loss{train/loss:.4f}",
        every_n_train_steps=int(cfg.training.get("checkpoint_every_n_steps", 2000)),
        save_top_k=-1,
        save_last=True,
    )
    csv_logger = CSVLogger(save_dir=cfg.log_dir, name="lightning_logs")
    loggers: list[Logger] = [csv_logger]

    # MLflow logger (mirrors the SPLADE training script's pattern).
    mlflow_cfg = cfg.training.get("mlflow") if cfg.training.get("mlflow") else None
    mlflow_enabled: bool = (
        bool(mlflow_cfg.get("enabled", True))
        if mlflow_cfg is not None
        else bool(os.environ.get("MLFLOW_TRACKING_URI"))
    )
    if mlflow_enabled:
        tracking_uri: str | None = None
        experiment_name: str = "Train-LENS"
        run_name: str | None = None
        if mlflow_cfg is not None:
            tracking_uri = mlflow_cfg.get("tracking_uri") or None
            experiment_name = str(mlflow_cfg.get("experiment_name") or experiment_name)
            run_name = mlflow_cfg.get("run_name") or None
        tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if run_name is None:
            tag = cfg.get("tag")
            run_name = str(tag) if tag else str(cfg.model.name)
        mlflow_logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            save_dir=str(cfg.log_dir),
            log_model=False,
        )
        # Log the full resolved config as hyperparameters once.
        try:
            mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        except Exception as exc:
            logger.warning("MLflow hparam logging failed: %s", exc)
        loggers.append(mlflow_logger)
        logger.info("MLflow logger configured: experiment=%s, run_name=%s, uri=%s",
                    experiment_name, run_name, tracking_uri)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy=str(cfg.training.strategy),
        precision=str(cfg.training.precision),
        max_epochs=int(cfg.training.get("num_epochs", 1)),
        max_steps=-1 if total_steps is None else total_steps,
        accumulate_grad_batches=int(cfg.training.grad_accumulation),
        gradient_clip_val=float(cfg.training.get("max_grad_norm", 1.0) or 0.0),
        logger=loggers,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        profiler=_build_profiler(cfg),
        default_root_dir=cfg.log_dir,
        # Our SameDatasetBatchSampler already handles per-rank slicing via its
        # `rank` argument; tell Lightning not to wrap it in a DistributedSampler.
        use_distributed_sampler=False,
    )
    trainer.fit(training_module, datamodule=data_module)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
