from typing import Any, Callable

from src.data.dataset.msmarco import MSMARCO
from src.utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")


def _build_hf_msmarco_dataset(
    cfg,
    global_cfg,
    tokenizer,
    load_teacher_scores: bool | None = None,
    require_teacher_scores: bool | None = None,
):
    return MSMARCO(
        cfg=cfg,
        global_cfg=global_cfg,
        tokenizer=tokenizer,
        load_teacher_scores=load_teacher_scores,
        require_teacher_scores=require_teacher_scores,
    )


for name in ("msmarco_hf_train", "msmarco_hf_val", "msmarco_minilm_scores"):
    DATASET_REGISTRY.register(name)(_build_hf_msmarco_dataset)
