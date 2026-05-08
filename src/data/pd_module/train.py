import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.collator import UniversalCollator
from src.data.dataclass import TrainingDataItem
from src.data.pd_module import PDModule
from src.data.term_supervision import OrderedMaskSlotTermSupervisor
from src.data.pd_module.utils import (
    build_rerank_inputs,
    uses_ordered_mask_slot_pooling,
)


class TrainingPDModule(PDModule):
    """Training PyTorch dataset module."""

    # --- Special methods ---
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        *,
        model_cfg: DictConfig | None = None,
        seed: int,
        load_teacher_scores: bool | None = None,
        require_teacher_scores: bool | None = None,
        cache_namespace: str | None = None,
    ) -> None:
        _ = cache_namespace
        super().__init__(
            cfg=cfg,
            tokenizer=tokenizer,
            model_cfg=model_cfg,
            seed=seed,
            load_teacher_scores=load_teacher_scores,
            require_teacher_scores=require_teacher_scores,
        )
        self._collator: UniversalCollator | None = None
        self._term_supervision: OrderedMaskSlotTermSupervisor | None = None
        self._slot_target_ignore_index: int = -100

    def __getitem__(self, idx: int) -> TrainingDataItem:
        with torch.autograd.profiler.record_function("splade.data.__getitem__"):
            data_idx: int = int(idx)
            with torch.autograd.profiler.record_function(
                "splade.data.build_meta_item"
            ):
                meta_item = self._build_meta_item(data_idx)
            with torch.autograd.profiler.record_function(
                "splade.data.build_rerank_inputs"
            ):
                inputs = build_rerank_inputs(
                    dataset=self.dataset,
                    tokenizer=self.tokenizer,
                    meta_item=meta_item,
                    model_cfg=self.model_cfg,
                    max_query_length=self.max_query_length,
                    max_doc_length=self.max_doc_length,
                    max_padding=self.max_padding,
                    term_supervision=self._term_supervision,
                    term_supervision_ignore_index=self._slot_target_ignore_index,
                )
            total_docs: int = int(inputs.num_pos + inputs.num_neg)
            label_tensor: torch.Tensor = torch.zeros(total_docs, dtype=torch.float)
            if inputs.num_pos:
                label_tensor[: inputs.num_pos] = 1.0
            pos_scores_tensor: torch.Tensor | None = (
                None
                if meta_item.pos_scores is None
                else torch.as_tensor(meta_item.pos_scores, dtype=torch.float)
            )
            neg_scores_tensor: torch.Tensor | None = (
                None
                if meta_item.neg_scores is None
                else torch.as_tensor(meta_item.neg_scores, dtype=torch.float)
            )
            return TrainingDataItem(
                data_idx=data_idx,
                qid=inputs.qid,
                pos_ids=inputs.pos_ids,
                neg_ids=inputs.neg_ids,
                query_text=inputs.query_text,
                doc_texts=inputs.doc_texts,
                query_input_ids=inputs.query_input_ids,
                query_attention_mask=inputs.query_attention_mask,
                query_pooling_mask=inputs.query_pooling_mask,
                doc_input_ids=inputs.doc_input_ids,
                doc_attention_mask=inputs.doc_attention_mask,
                doc_pooling_mask=inputs.doc_pooling_mask,
                doc_mask=inputs.doc_mask,
                pos_mask=inputs.pos_mask,
                teacher_scores=inputs.teacher_scores,
                labels=label_tensor,
                pos_scores=pos_scores_tensor,
                neg_scores=neg_scores_tensor,
                query_slot_target_ids=inputs.query_slot_target_ids,
                doc_slot_target_ids=inputs.doc_slot_target_ids,
            )

    # --- Property methods ---
    @property
    def collator(self) -> UniversalCollator:
        if self._collator is None:
            max_docs: int = int(self.num_positives + self.num_negatives)
            self._collator = UniversalCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                require_teacher_scores=self.require_teacher_scores,
                slot_target_ignore_index=self._slot_target_ignore_index,
                max_padding=self.max_padding,
                max_query_length=self.max_query_length,
                max_doc_length=self.max_doc_length,
                max_docs=max_docs,
            )
        return self._collator

    def setup(self) -> None:
        super().setup()
        if not uses_ordered_mask_slot_pooling(self.model_cfg):
            self._term_supervision = None
            return
        if self._term_supervision is None:
            ordered_cfg: DictConfig | None = None
            if self.model_cfg is not None and "ordered_mask_slots" in self.model_cfg:
                ordered_cfg = self.model_cfg.ordered_mask_slots
            cache_dir: str | None = (
                None if ordered_cfg is None else ordered_cfg.get("idf_cache_dir")
            )
            idf_batch_size: int = (
                1024 if ordered_cfg is None else int(ordered_cfg.get("idf_batch_size", 1024))
            )
            idf_log_interval: int = (
                100_000
                if ordered_cfg is None
                else int(ordered_cfg.get("idf_log_interval", 100_000))
            )
            idf_cache_wait_timeout_seconds: float = (
                7200.0
                if ordered_cfg is None
                else float(
                    ordered_cfg.get("idf_cache_wait_timeout_seconds", 7200.0)
                )
            )
            idf_num_workers: int = (
                0 if ordered_cfg is None else int(ordered_cfg.get("idf_num_workers", 0))
            )
            idf_shards_per_worker: int = (
                4
                if ordered_cfg is None
                else int(ordered_cfg.get("idf_shards_per_worker", 4))
            )
            exclude_token_ids = None
            if self.model_cfg is not None and "exclude_token_ids" in self.model_cfg:
                exclude_token_ids = torch.tensor(
                    [int(token_id) for token_id in self.model_cfg.exclude_token_ids],
                    dtype=torch.long,
                )
            self._term_supervision = OrderedMaskSlotTermSupervisor(
                dataset=self.dataset,
                tokenizer=self.tokenizer,
                cache_dir=cache_dir,
                excluded_token_ids=exclude_token_ids,
                idf_batch_size=idf_batch_size,
                idf_log_interval=idf_log_interval,
                cache_wait_timeout_seconds=idf_cache_wait_timeout_seconds,
                idf_num_workers=idf_num_workers,
                idf_shards_per_worker=idf_shards_per_worker,
            )
        self._term_supervision.prepare()

    def _requires_query_text_dataset(self) -> bool:
        return not bool(
            getattr(self.dataset, "provides_query_texts_inline", False)
        )

    def _requires_corpus_text_dataset(self) -> bool:
        return not bool(getattr(self.dataset, "provides_doc_texts_inline", False))

    def _requires_query_id_to_idx(self) -> bool:
        return self._requires_query_text_dataset()

    def _requires_corpus_id_to_idx(self) -> bool:
        return self._requires_corpus_text_dataset()
