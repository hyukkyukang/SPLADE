import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.dataset.datasets.retrieval import RetrievalDataset
from src.model.retriever.sparse_retriever import SparseRetriever
from src.model.splade import SpladeModel
from src.tokenization.tokenizer import build_tokenizer
from src.utils.logging import get_logger, patch_hydra_argparser_for_python314


logger = get_logger(__name__, __file__)
patch_hydra_argparser_for_python314()


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
        model.load_state_dict(filtered, strict=False)
    return model


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="evaluate")
def main(cfg: DictConfig) -> None:
    device = torch.device("cpu" if cfg.testing.use_cpu else "cuda")
    model = _load_model(cfg).to(device)
    tokenizer = build_tokenizer(cfg.model.huggingface_name)

    if cfg.dataset.use_hf and cfg.dataset.hf_name:
        dataset = RetrievalDataset.from_hf(
            hf_name=cfg.dataset.hf_name,
            split=cfg.dataset.hf_split,
            cache_dir=cfg.dataset.hf_cache_dir,
        )
    else:
        dataset = RetrievalDataset(
            corpus_path=cfg.dataset.corpus_path,
            queries_path=cfg.dataset.queries_path,
            qrels_path=cfg.dataset.qrels_path,
        )

    retriever = SparseRetriever(
        model=model,
        tokenizer=tokenizer,
        max_query_length=cfg.dataset.max_query_length,
        max_doc_length=cfg.dataset.max_doc_length,
        batch_size=cfg.testing.batch_size,
        device=device,
    )

    doc_ids, doc_reps = retriever._encode_corpus(dataset.corpus)
    index_path = Path(cfg.model.index_path or "index")
    index_path.mkdir(parents=True, exist_ok=True)

    torch.save(doc_reps, index_path / "doc_embeddings.pt")
    with (index_path / "doc_ids.json").open("w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    logger.info(f"Saved index to {index_path}")


if __name__ == "__main__":
    main()
