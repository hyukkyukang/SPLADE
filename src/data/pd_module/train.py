import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.collator import UniversalCollator
from src.data.dataclass import MetaItem, TrainingDataItem
from src.data.pd_module import PDModule
from src.data.pd_module.utils import build_rerank_inputs


class TrainingPDModule(PDModule):
    """Training PyTorch datasets module."""

    # --- Special methods ---
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        *,
        seed: int,
        load_teacher_scores: bool | None = None,
        require_teacher_scores: bool | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            tokenizer=tokenizer,
            seed=seed,
            load_teacher_scores=load_teacher_scores,
            require_teacher_scores=require_teacher_scores,
        )
        self._collator: UniversalCollator | None = None

    def __getitem__(self, idx: int) -> TrainingDataItem:
        meta_item: MetaItem = self._build_meta_item(int(idx))
        inputs = build_rerank_inputs(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            meta_item=meta_item,
            max_query_length=self.max_query_length,
            max_doc_length=self.max_doc_length,
            max_padding=self.max_padding,
        )

        labels: list[float] = [1.0] * inputs.num_pos + [0.0] * inputs.num_neg
        label_tensor: torch.Tensor = torch.tensor(labels, dtype=torch.float)
        pos_scores_tensor: torch.Tensor | None = (
            None
            if meta_item.pos_scores is None
            else torch.tensor(meta_item.pos_scores, dtype=torch.float)
        )
        neg_scores_tensor: torch.Tensor | None = (
            None
            if meta_item.neg_scores is None
            else torch.tensor(meta_item.neg_scores, dtype=torch.float)
        )

        return TrainingDataItem(
            data_idx=int(idx),
            qid=inputs.qid,
            pos_ids=inputs.pos_ids,
            neg_ids=inputs.neg_ids,
            query_text=inputs.query_text,
            doc_texts=inputs.doc_texts,
            query_input_ids=inputs.query_input_ids,
            query_attention_mask=inputs.query_attention_mask,
            doc_input_ids=inputs.doc_input_ids,
            doc_attention_mask=inputs.doc_attention_mask,
            doc_mask=inputs.doc_mask,
            pos_mask=inputs.pos_mask,
            teacher_scores=inputs.teacher_scores,
            labels=label_tensor,
            pos_scores=pos_scores_tensor,
            neg_scores=neg_scores_tensor,
        )

    # --- Property methods ---
    @property
    def collator(self) -> UniversalCollator:
        if self._collator is None:
            max_docs: int = int(self.num_positives + self.num_negatives)
            self._collator = UniversalCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                require_teacher_scores=self.require_teacher_scores,
                max_padding=self.max_padding,
                max_query_length=self.max_query_length,
                max_doc_length=self.max_doc_length,
                max_docs=max_docs,
            )
        return self._collator
