import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR
from src.indexing.sparse_index import (
    ShardInfo,
    build_inverted_index_from_shards,
    load_shard_manifest,
    resolve_numpy_dtype,
)
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.script_setup import configure_script_environment

logger: logging.Logger = get_logger(__name__, __file__)

configure_script_environment(
    load_env=False,
    set_tokenizers_parallelism=True,
    set_matmul_precision=False,
    suppress_lightning_tips=True,
    suppress_httpx=True,
    suppress_dataloader_workers=True,
)


@hydra.main(version_base=None, config_path=ABS_CONFIG_DIR, config_name="encode")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
    encode_path_value: str | None = getattr(cfg.model, "encode_path", None)
    index_path_value: str | None = getattr(cfg.model, "index_path", None)
    encode_path: Path = Path(encode_path_value or "encode")
    index_path: Path = Path(index_path_value or "index")
    index_path.mkdir(parents=True, exist_ok=True)

    shard_infos: list[ShardInfo]
    metadata: dict[str, Any]
    shard_infos, metadata = load_shard_manifest(encode_path)
    vocab_size: int | None = None
    if metadata.get("vocab_size") is not None:
        vocab_size = int(metadata["vocab_size"])
    if vocab_size is None:
        raise ValueError("Missing vocab_size in encode metadata.")

    value_dtype_name: str = str(metadata.get("value_dtype") or cfg.encoding.value_dtype)
    value_dtype: np.dtype = resolve_numpy_dtype(value_dtype_name)

    term_ptr: np.ndarray
    post_doc_ids: np.ndarray
    post_weights: np.ndarray
    doc_ids: list[str]
    term_ptr, post_doc_ids, post_weights, doc_ids = build_inverted_index_from_shards(
        shard_infos, vocab_size=vocab_size, value_dtype=value_dtype
    )

    np.save(index_path / "term_ptr.npy", term_ptr)
    np.save(index_path / "post_doc_ids.npy", post_doc_ids)
    np.save(index_path / "post_weights.npy", post_weights)

    with (index_path / "doc_ids.json").open("w", encoding="utf-8") as doc_file:
        json.dump(doc_ids, doc_file)

    metadata_out: dict[str, Any] = {
        "vocab_size": vocab_size,
        "doc_count": len(doc_ids),
        "nnz": int(term_ptr[-1]),
        "value_dtype": value_dtype_name,
        "encode_path": str(encode_path),
        "top_k": metadata.get("top_k"),
        "min_weight": metadata.get("min_weight"),
        "exclude_token_ids": metadata.get("exclude_token_ids"),
    }
    with (index_path / "metadata.json").open("w", encoding="utf-8") as meta_file:
        json.dump(metadata_out, meta_file, indent=2)

    log_if_rank_zero(
        logger,
        f"Saved inverted index to {index_path} (docs={len(doc_ids)}, nnz={int(term_ptr[-1])}).",
    )


if __name__ == "__main__":
    main()
