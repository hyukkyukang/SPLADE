import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# For Python 3.14 compatibility
from src.utils.logging import patch_hydra_argparser_for_python314

patch_hydra_argparser_for_python314()

import os

import hydra
import torch
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.metric.beir_evaluator import BEIREvaluator
from src.model.splade import SpladeModel
from src.tokenization.tokenizer import build_tokenizer
from src.utils.logging import (
    get_logger,
    setup_tqdm_friendly_logging,
)

logger = get_logger(__name__, __file__)


def _load_model(cfg: DictConfig) -> SpladeModel:
    dtype = torch.float16 if cfg.model.dtype == "float16" else None
    model = SpladeModel(
        model_name=cfg.model.huggingface_name,
        query_pooling=cfg.model.query_pooling,
        doc_pooling=cfg.model.doc_pooling,
        sparse_activation=cfg.model.sparse_activation,
        attn_implementation=cfg.model.attn_implementation,
        dtype=dtype,
        normalize=cfg.model.normalize,
    )

    if cfg.testing.checkpoint_path:
        checkpoint = torch.load(cfg.testing.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        filtered = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                filtered[key.replace("model.", "", 1)] = value
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        logger.info(
            f"Loaded checkpoint. Missing: {len(missing)}, unexpected: {len(unexpected)}"
        )
    return model


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate")
def main(cfg: DictConfig) -> None:
    setup_tqdm_friendly_logging()
    device = torch.device("cpu" if cfg.testing.use_cpu else "cuda")

    model = _load_model(cfg).to(device)
    tokenizer = build_tokenizer(cfg.model.huggingface_name)

    evaluator = BEIREvaluator(
        model=model,
        tokenizer=tokenizer,
        max_query_length=cfg.dataset.max_query_length,
        max_doc_length=cfg.dataset.max_doc_length,
        batch_size=cfg.testing.batch_size,
        device=device,
    )

    metrics = evaluator.evaluate_hf(
        hf_name=cfg.dataset.hf_name,
        split=cfg.dataset.hf_split,
        metrics=cfg.testing.metrics,
        top_k=100,
        cache_dir=cfg.dataset.hf_cache_dir,
    )

    for name, value in metrics.items():
        logger.info(f"{cfg.dataset.name} {name}: {value:.4f}")


if __name__ == "__main__":
    main()
