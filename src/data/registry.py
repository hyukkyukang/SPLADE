from typing import Any, Callable

from src.data.datasets.train_hf import (
    HFMSMarcoTrainDataset,
    HFMSMarcoTrainIterableDataset,
)
from src.utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")


def _build_hf_msmarco_dataset(
    cfg,
    global_cfg,
    tokenizer,
    load_teacher_scores: bool | None = None,
    require_teacher_scores: bool | None = None,
):
    dataset_cls: Callable[..., Any] = (
        HFMSMarcoTrainIterableDataset
        if getattr(cfg, "hf_streaming", False)
        else HFMSMarcoTrainDataset
    )
    return dataset_cls(
        cfg=cfg,
        global_cfg=global_cfg,
        tokenizer=tokenizer,
        load_teacher_scores=load_teacher_scores,
        require_teacher_scores=require_teacher_scores,
    )


for name in ("msmarco_hf_train", "msmarco_hf_val", "msmarco_minilm_scores"):
    DATASET_REGISTRY.register(name)(_build_hf_msmarco_dataset)
