from __future__ import annotations

import argparse

from hydra import compose, initialize_config_dir
import torch

from config.path import ABS_CONFIG_DIR
from src.data.registry import build_dataset
from src.data.term_supervision import OrderedMaskSlotTermSupervisor
from src.utils.logging import patch_hydra_argparser_for_python314
from src.utils.transformers import build_tokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ordered mask-slot TF-IDF supervision cache."
    )
    parser.add_argument(
        "--config-name",
        default="train_ordered_mask_slot_splade",
        help="Hydra config name to compose.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, for example: dataset@train_dataset=msmarco_hard_negatives",
    )
    return parser.parse_args()


def main() -> None:
    patch_hydra_argparser_for_python314()
    args = _parse_args()
    with initialize_config_dir(version_base=None, config_dir=ABS_CONFIG_DIR):
        cfg = compose(config_name=str(args.config_name), overrides=list(args.overrides))

    dataset_cfg = cfg.train_dataset if "train_dataset" in cfg else cfg.dataset
    dataset = build_dataset(dataset_cfg)

    tokenizer_name = str(
        cfg.model.get("tokenizer_name") or cfg.model.get("huggingface_name")
    )
    tokenizer = build_tokenizer(
        tokenizer_name,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        revision=cfg.model.get("model_revision"),
    )

    ordered_cfg = cfg.model.ordered_mask_slots
    exclude_token_ids = cfg.model.get("exclude_token_ids")
    supervisor = OrderedMaskSlotTermSupervisor(
        dataset=dataset,
        tokenizer=tokenizer,
        cache_dir=ordered_cfg.idf_cache_dir,
        excluded_token_ids=(
            None
            if exclude_token_ids is None
            else torch.tensor([int(token_id) for token_id in exclude_token_ids])
        ),
        idf_batch_size=int(ordered_cfg.idf_batch_size),
        idf_log_interval=int(ordered_cfg.idf_log_interval),
        cache_wait_timeout_seconds=float(ordered_cfg.idf_cache_wait_timeout_seconds),
        idf_num_workers=int(ordered_cfg.get("idf_num_workers", 0)),
        idf_shards_per_worker=int(ordered_cfg.get("idf_shards_per_worker", 4)),
    )
    supervisor.prepare()
    print(supervisor._cache_path())


if __name__ == "__main__":
    main()
