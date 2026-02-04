import json
import logging
import os
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

from config.path import ABS_CONFIG_DIR
from src.utils import log_if_rank_zero, set_seed
from src.utils.logging import get_logger, setup_tqdm_friendly_logging
from src.utils.script_setup import configure_script_environment
from src.utils.model_utils import (
    apply_checkpoint_model_config,
    build_splade_model,
    load_splade_checkpoint,
)
from src.utils.sparse_encoder import (
    build_doc_only_sparse_encoder_adapter,
    build_sparse_encoder_from_checkpoint,
    build_sparse_encoder_from_huggingface,
    resolve_nanobeir_compatibility,
)

logger: logging.Logger = get_logger("scripts.evaluate_nanobeir", __file__)

configure_script_environment(
    load_env=False,
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

    device_id: int | None = cfg.testing.device_id
    if torch.cuda.is_available():
        if device_id is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{int(device_id)}")
    return torch.device("cpu")


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate_nanobeir"
)
def main(cfg: DictConfig) -> None:
    setup_tqdm_friendly_logging()
    os.makedirs(cfg.log_dir, exist_ok=True)
    set_seed(cfg.seed)
    log_if_rank_zero(logger, f"Random seed set to: {cfg.seed}")

    checkpoint_path_value: str | None = cfg.testing.checkpoint_path
    if not checkpoint_path_value:
        raise ValueError("testing.checkpoint_path must be set for NanoBEIR eval.")
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
    use_huggingface_model: bool = bool(cfg.nanobeir.use_huggingface_model)

    # Resolve device before model instantiation to keep tensors aligned.
    device: torch.device = _resolve_device(cfg)
    log_if_rank_zero(logger, f"Using device: {device}")

    checkpoint_path_value: str | None = cfg.testing.checkpoint_path
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
        checkpoint_path: str = str(checkpoint_path_value)

    doc_only_enabled: bool = (
        bool(cfg.model.doc_only) if hasattr(cfg.model, "doc_only") else False
    )
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
        sparse_encoder: Any = build_doc_only_sparse_encoder_adapter(
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


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
