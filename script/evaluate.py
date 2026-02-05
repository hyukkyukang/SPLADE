import argparse
import json
import logging
import os
import sys
from typing import Any, Sequence

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

from config.path import ABS_CONFIG_DIR
from src.data.pl_module import RerankingDataModule, RetrievalDataModule
from src.model.pl_module import RerankingLightningModule, RetrievalEvalLightningModule
from src.utils import log_if_rank_zero
from src.utils.logging import get_logger
from src.utils.model_utils import (
    apply_checkpoint_model_config,
    build_splade_model,
    load_splade_checkpoint,
)
from src.utils.script_setup import (
    configure_script_environment,
    initialize_run,
    normalize_optional_path,
    resolve_model_source,
    resolve_trainer_settings,
)
from src.utils.sparse_encoder import (
    build_doc_only_sparse_encoder_adapter,
    build_sparse_encoder_from_checkpoint,
    build_sparse_encoder_from_huggingface,
    resolve_nanobeir_compatibility,
)

logger: logging.Logger = get_logger(__name__, __file__)

configure_script_environment(
    load_env=True,
    set_tokenizers_parallelism=True,
    set_matmul_precision=True,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


def _resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve the torch device from the testing config."""
    use_cpu: bool = bool(cfg.testing.use_cpu)
    if use_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_standard_eval(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=True)

    cfg = apply_checkpoint_model_config(
        cfg,
        checkpoint_path=cfg.testing.checkpoint_path,
        logger=logger,
    )

    eval_type: str = str(cfg.evaluation.type).lower()
    eval_module: L.LightningModule
    data_module: L.LightningDataModule
    if eval_type == "retrieval":
        eval_module = RetrievalEvalLightningModule(cfg=cfg)
        data_module = RetrievalDataModule(cfg=cfg)
    elif eval_type == "reranking":
        eval_module = RerankingLightningModule(cfg=cfg)
        data_module = RerankingDataModule(cfg=cfg)
    else:
        raise ValueError(f"Unsupported evaluation.type: {eval_type}")
    eval_module.eval()

    testing_cfg: DictConfig = cfg.testing
    trainer_kwargs, precision = resolve_trainer_settings(testing_cfg)

    trainer: L.Trainer = L.Trainer(
        precision=precision,
        default_root_dir=cfg.log_dir,
        logger=False,
        **trainer_kwargs,
    )

    trainer.test(model=eval_module, datamodule=data_module)


def _run_nanobeir_eval(cfg: DictConfig) -> None:
    initialize_run(cfg, logger=logger, suppress_lightning_tips=False)

    checkpoint_path_value: str | None = normalize_optional_path(
        cfg.testing.checkpoint_path
    )
    use_huggingface_model: bool = bool(cfg.nanobeir.use_huggingface_model)
    if not checkpoint_path_value and not use_huggingface_model:
        raise ValueError(
            "testing.checkpoint_path must be set for NanoBEIR eval "
            "unless nanobeir.use_huggingface_model=true."
        )
    if checkpoint_path_value and not use_huggingface_model:
        checkpoint_path: str = str(checkpoint_path_value)
        cfg = apply_checkpoint_model_config(
            cfg,
            checkpoint_path=checkpoint_path,
            logger=logger,
        )

    # Normalize dataset names to strings for the evaluator.
    dataset_names: list[str] = []
    dataset_name: str
    for dataset_name in cfg.nanobeir.datasets:
        dataset_names.append(str(dataset_name))
    if not dataset_names:
        raise ValueError("nanobeir.datasets must contain at least one dataset name.")

    batch_size: int = int(cfg.nanobeir.batch_size)
    save_json: bool = bool(cfg.nanobeir.save_json)

    # Resolve device before model instantiation to keep tensors aligned.
    device: torch.device = _resolve_device(cfg)
    log_if_rank_zero(logger, f"Using device: {device}")

    sparse_encoder: SparseEncoder
    if use_huggingface_model:
        # Build from Hugging Face weights when no checkpoint override is desired.
        hf_model_name: str = str(cfg.model.huggingface_name)
        if not hf_model_name:
            raise ValueError(
                "model.huggingface_name must be set when using Hugging Face "
                "weights for NanoBEIR evaluation."
            )
        if checkpoint_path_value:
            log_if_rank_zero(
                logger,
                "Ignoring testing.checkpoint_path because "
                "nanobeir.use_huggingface_model=true.",
            )
        log_if_rank_zero(logger, f"Loading Hugging Face model weights: {hf_model_name}")
        sparse_encoder = build_sparse_encoder_from_huggingface(cfg=cfg, device=device)
    else:
        if not checkpoint_path_value:
            raise ValueError(
                "testing.checkpoint_path must be set for NanoBEIR eval when "
                "nanobeir.use_huggingface_model=false."
            )
        checkpoint_path = str(checkpoint_path_value)

        doc_only_enabled: bool = bool(cfg.model.doc_only)
        if doc_only_enabled:
            model: Any = build_splade_model(cfg, use_cpu=cfg.testing.use_cpu)
            missing: list[str]
            unexpected: list[str]
            missing, unexpected = load_splade_checkpoint(
                model, checkpoint_path, logger=logger
            )
            log_if_rank_zero(
                logger,
                f"Loaded checkpoint. Missing: {len(missing)}, unexpected: {len(unexpected)}",
            )
            sparse_encoder = build_doc_only_sparse_encoder_adapter(
                cfg=cfg,
                model=model,
                device=device,
                batch_size=batch_size,
            )
        else:
            compatible: bool
            reason: str | None
            compatible, reason = resolve_nanobeir_compatibility(cfg)
            if not compatible:
                raise ValueError(f"NanoBEIR evaluation incompatible: {reason}")

            # Convert the Lightning checkpoint into a SparseEncoder for NanoBEIR.
            sparse_encoder = build_sparse_encoder_from_checkpoint(
                cfg=cfg, checkpoint_path=checkpoint_path, device=device
            )

    # Run NanoBEIR proxy evaluation and collect metrics.
    evaluator: SparseNanoBEIREvaluator = SparseNanoBEIREvaluator(
        dataset_names=dataset_names,
        batch_size=batch_size,
    )

    results: dict[str, Any] = evaluator(sparse_encoder)
    metric_name: str
    metric_value: Any
    for metric_name, metric_value in results.items():
        log_if_rank_zero(logger, f"NanoBEIR {metric_name}: {metric_value}")

    if save_json:
        output_path: str = os.path.join(cfg.log_dir, "nanobeir_metrics.json")
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(results, json_file, indent=2)
        log_if_rank_zero(logger, f"Saved NanoBEIR metrics to {output_path}")


def _has_config_name_override(argv: Sequence[str]) -> bool:
    for arg in argv:
        if arg in ("--config-name", "-cn"):
            return True
        if arg.startswith("--config-name="):
            return True
    return False


def _extract_benchmark_arg(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--benchmark",
        choices=("msmarco", "nanobeir"),
        type=str.lower,
    )
    parser.add_argument(
        "--mode",
        choices=("msmarco", "nanobeir"),
        type=str.lower,
    )
    args, remaining = parser.parse_known_args(list(argv))
    benchmark: str | None = args.benchmark or args.mode
    return benchmark, remaining


def _apply_benchmark_config_name(
    argv: Sequence[str], benchmark: str | None
) -> list[str]:
    if benchmark is None:
        return list(argv)
    if _has_config_name_override(argv):
        return list(argv)
    config_name = "evaluate_nanobeir" if benchmark == "nanobeir" else "evaluate"
    return ["--config-name", config_name, *argv]


def _is_nanobeir_run(cfg: DictConfig) -> bool:
    return hasattr(cfg, "nanobeir")


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate")
def main(cfg: DictConfig) -> None:
    is_nanobeir: bool = _is_nanobeir_run(cfg)
    cfg = resolve_model_source(cfg, logger=logger, set_nanobeir_flag=is_nanobeir)
    if is_nanobeir:
        _run_nanobeir_eval(cfg)
    else:
        _run_standard_eval(cfg)
    log_if_rank_zero(logger, "Evaluation complete")


if __name__ == "__main__":
    benchmark, remaining_args = _extract_benchmark_arg(sys.argv[1:])
    sys.argv = [sys.argv[0]] + _apply_benchmark_config_name(remaining_args, benchmark)
    main()  # pylint: disable=no-value-for-parameter
