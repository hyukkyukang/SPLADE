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
from src.utils.sparse_encoder import build_sparse_encoder_from_checkpoint

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
    use_cpu: bool = bool(getattr(cfg.testing, "use_cpu", False))
    if use_cpu:
        return torch.device("cpu")

    device_id: int | None = getattr(cfg.testing, "device_id", None)
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

    checkpoint_path_value: str | None = getattr(cfg.testing, "checkpoint_path", None)
    if not checkpoint_path_value:
        raise ValueError("testing.checkpoint_path must be set for NanoBEIR eval.")
    checkpoint_path: str = str(checkpoint_path_value)

    # Normalize dataset names to strings for the evaluator.
    dataset_names: list[str] = []
    dataset_name: str
    for dataset_name in cfg.nanobeir.datasets:
        dataset_names.append(str(dataset_name))
    if not dataset_names:
        raise ValueError("nanobeir.datasets must contain at least one dataset name.")

    batch_size: int = int(cfg.nanobeir.batch_size)
    save_json: bool = bool(getattr(cfg.nanobeir, "save_json", False))

    # Resolve device before model instantiation to keep tensors aligned.
    device: torch.device = _resolve_device(cfg)
    log_if_rank_zero(logger, f"Using device: {device}")

    # Convert the Lightning checkpoint into a SparseEncoder for NanoBEIR.
    sparse_encoder: SparseEncoder = build_sparse_encoder_from_checkpoint(
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
