import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from config.path import ABS_CONFIG_DIR
from src.data.datasets.retrieval_hf import HFRetrievalDataset
from src.model.retriever.sparse.neural.splade import SPLADE
from src.model.retriever.sparse.neural.splade_model import SpladeModel
from src.utils.model_utils import build_splade_model, load_splade_checkpoint
from src.utils.script_setup import configure_script_environment
from src.utils.transformers import build_tokenizer
from src.utils.logging import get_logger

logger: logging.Logger = get_logger(__name__, __file__)

configure_script_environment(
    load_env=False,
    set_tokenizers_parallelism=True,
    set_matmul_precision=True,
    suppress_lightning_tips=False,
    suppress_httpx=False,
    suppress_dataloader_workers=False,
)


def _load_model(cfg: DictConfig) -> SpladeModel:
    model: SpladeModel = build_splade_model(cfg, use_cpu=cfg.testing.use_cpu)
    checkpoint_path: str | None = getattr(cfg.testing, "checkpoint_path", None)
    if checkpoint_path:
        missing: list[str]
        unexpected: list[str]
        missing, unexpected = load_splade_checkpoint(model, checkpoint_path)
        logger.info(
            "Loaded checkpoint. Missing: %d, unexpected: %d",
            len(missing),
            len(unexpected),
        )
    return model


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate")
def main(cfg: DictConfig) -> None:
    device: torch.device = torch.device("cpu" if cfg.testing.use_cpu else "cuda")
    model: SpladeModel = _load_model(cfg).to(device)
    # Build tokenizer with padding support for indexing.
    tokenizer: PreTrainedTokenizerBase = build_tokenizer(cfg.model.huggingface_name)

    hf_name: str | None = cfg.dataset.hf_name or (
        f"BeIR/{cfg.dataset.beir_dataset}" if cfg.dataset.beir_dataset else None
    )
    if hf_name is None:
        raise ValueError("dataset.hf_name or dataset.beir_dataset must be set.")

    dataset: HFRetrievalDataset = HFRetrievalDataset.from_hf(
        hf_name=hf_name,
        split=cfg.dataset.hf_split,
        cache_dir=cfg.dataset.hf_cache_dir,
        tokenizer=tokenizer,
        max_query_length=cfg.dataset.max_query_length,
    )

    retriever: SPLADE = SPLADE(
        model=model,
        tokenizer=tokenizer,
        max_query_length=cfg.dataset.max_query_length,
        max_doc_length=cfg.dataset.max_doc_length,
        batch_size=cfg.testing.batch_size,
        device=device,
    )

    doc_ids: list[str]
    doc_reps: torch.Tensor
    doc_ids, doc_reps = retriever._encode_corpus(dataset.corpus)
    index_path: Path = Path(cfg.model.index_path or "index")
    index_path.mkdir(parents=True, exist_ok=True)

    torch.save(doc_reps, index_path / "doc_embeddings.pt")
    with (index_path / "doc_ids.json").open("w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    logger.info(f"Saved index to {index_path}")


if __name__ == "__main__":
    main()
