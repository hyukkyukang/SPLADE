from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.dataset.msmarco import MSMARCO
from src.data.dataset.msmarco_local_triplets import MSMARCOLocalTriplets
from src.utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")


def _build_hf_msmarco_dataset(
    cfg: DictConfig,
    global_cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    load_teacher_scores: bool | None = None,
    require_teacher_scores: bool | None = None,
) -> MSMARCO:
    return MSMARCO(
        cfg=cfg,
        global_cfg=global_cfg,
        tokenizer=tokenizer,
        load_teacher_scores=load_teacher_scores,
        require_teacher_scores=require_teacher_scores,
    )


def _build_local_triplets_dataset(
    cfg: DictConfig,
    global_cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    load_teacher_scores: bool | None = None,
    require_teacher_scores: bool | None = None,
) -> MSMARCOLocalTriplets:
    _ = load_teacher_scores, require_teacher_scores
    return MSMARCOLocalTriplets(cfg=cfg, global_cfg=global_cfg, tokenizer=tokenizer)


for name in ("msmarco_hf_train", "msmarco_hf_val", "msmarco_minilm_scores"):
    DATASET_REGISTRY.register(name)(_build_hf_msmarco_dataset)

DATASET_REGISTRY.register("msmarco_local_triplets")(_build_local_triplets_dataset)
