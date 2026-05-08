"""Inline online validation: run MTEB on a single small task during training.

Lightning calls ``LightningModule.on_validation_epoch_end`` between training
windows; this module provides the wiring so we can run an MTEB single-task
evaluation against the **live in-memory model** (PEFT-wrapped, in eval mode)
without going through disk-checkpoint -> merge -> reload.

Why not merge-then-eval: PEFT models already apply the LoRA delta during
forward when not merged, so the eval output is bit-equivalent to the
merge_and_unload path. Skipping the merge saves the ~30s and avoids
doubling VRAM for the working copy.

Why a thin subclass of LENSMTEBEncoder: keeps the tested ``_encode``
(tokenization, pooling, normalization) instead of reimplementing it.
We just bypass the constructor's model-loading path and inject a
pre-built model + tokenizer.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from src.utils.lens_mteb_encoder import (
    LENSMTEBEncoder,
    _QUERY_TOKEN,
)
from src.utils.lens_mteb_instructions import task_to_instruction

logger = logging.getLogger(__name__)


class InlineLENSEncoder(LENSMTEBEncoder):
    """LENSMTEBEncoder that wraps an already-instantiated model.

    Bypasses the parent's HF-from-disk loading. Everything else (encoding,
    pooling, query/doc instruction handling) is inherited unchanged.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device | str,
        batch_size: int = 64,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        default_instruction: str = (
            "Given a question, retrieve relevant documents that best answer the question."
        ),
    ) -> None:
        # NOTE: deliberately NOT calling super().__init__ -- it does the
        # disk load we are bypassing. We assemble the parent's expected
        # attributes by hand.
        from mteb.models.model_meta import ModelMeta

        self.model_name_or_path: str = "<inline>"
        self.batch_size: int = int(batch_size)
        self.max_length: int = int(max_length)
        self.normalize_embeddings: bool = bool(normalize_embeddings)
        self.default_instruction: str = str(default_instruction)

        self.device = torch.device(device)
        self.model = model
        self.tokenizer = tokenizer
        self._query_token_id = int(
            self.tokenizer.convert_tokens_to_ids(_QUERY_TOKEN)
        )
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(name="<inline>", revision=None)
        )


def run_single_task_eval(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device | str,
    task_name: str = "NFCorpus",
    batch_size: int = 64,
    max_length: int = 512,
) -> float | None:
    """Run MTEB on one task, return the headline metric (nDCG@10 for retrieval).

    Returns ``None`` if the task fails to run or the result shape is unexpected
    -- in that case the caller should treat this as "no signal" and not stop
    training based on it.
    """
    import mteb

    try:
        instruction = task_to_instruction(task_name, is_query=True)
    except Exception:
        instruction = (
            "Given a question, retrieve relevant documents that best answer "
            "the question."
        )

    encoder = InlineLENSEncoder(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        default_instruction=instruction,
    )

    # Resolve the task object (MTEB 2.x: get_tasks then evaluate).
    try:
        task_objs = list(mteb.get_tasks(tasks=[task_name], languages=["eng"]))
    except Exception as exc:
        logger.warning("get_tasks(%s) failed: %s", task_name, exc)
        return None
    if not task_objs:
        logger.warning("No MTEB task object for %r", task_name)
        return None
    task_obj = task_objs[0]
    # Pin eval split.
    try:
        if hasattr(task_obj.metadata, "eval_splits"):
            object.__setattr__(task_obj.metadata, "eval_splits", ["test"])
    except Exception:
        pass

    try:
        result = mteb.evaluate(
            encoder,
            task_obj,
            encode_kwargs={"batch_size": batch_size, "show_progress_bar": False},
            overwrite_strategy="always",
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.warning(
            "Inline %s validation crashed during evaluate(): %s",
            task_name, exc,
        )
        return None

    # Pull main_score (nDCG@10 for retrieval tasks).
    try:
        for tr in result.task_results:
            if str(tr.task_name) == task_name:
                return float(tr.main_score)
        return float(result.task_results[0].main_score)
    except (IndexError, AttributeError, TypeError) as exc:
        logger.warning(
            "Inline %s validation got an unparsable result: %s",
            task_name, exc,
        )
        return None
