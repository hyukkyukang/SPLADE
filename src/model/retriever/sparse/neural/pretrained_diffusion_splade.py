from __future__ import annotations

from collections.abc import Sequence

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.model.retriever.sparse.neural.mdlm_splade import MDLMSpladeModel
from src.utils.normalize import normalize_optional_str
from src.utils.transformers import build_tokenizer


class PretrainedDiffusionSpladeModel(MDLMSpladeModel):
    """SPLADE-style sparse retriever initialized from a diffusion backbone.

    Retrieval behavior is identical to ``MDLMSpladeModel``: sparse vectors are
    still built from raw MLM logits on clean inputs. This subclass only adds
    initialization-time validation for diffusion-backbone experiments.
    """

    def __init__(
        self,
        family: str,
        model_name: str,
        huggingface_model_class: str,
        query_pooling: str,
        doc_pooling: str,
        sparse_activation: str,
        *,
        attn_implementation: str | None = None,
        dtype: torch.dtype | None = None,
        normalize: bool = False,
        doc_only: bool = False,
        tie_word_embeddings: bool = False,
        peft_cfg: DictConfig | None = None,
        freeze_backbone: bool = False,
        trust_remote_code: bool = False,
        model_revision: str | None = None,
        local_files_only: bool | None = None,
        exclude_token_ids: Sequence[int] | None = None,
        mask_token_id: int | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        use_fast_tokenizer: bool = True,
        require_fast_tokenizer: bool = False,
        enforce_same_tokenizer_as_baseline: bool = False,
        baseline_tokenizer_name: str | None = None,
    ) -> None:
        self.tokenizer_source: str = (
            normalize_optional_str(tokenizer_name) or str(model_name)
        )
        self.tokenizer_revision: str | None = normalize_optional_str(
            tokenizer_revision
        )
        self.baseline_tokenizer_name: str | None = normalize_optional_str(
            baseline_tokenizer_name
        )
        self.enforce_same_tokenizer_as_baseline: bool = bool(
            enforce_same_tokenizer_as_baseline
        )
        self.use_fast_tokenizer: bool = bool(use_fast_tokenizer)
        self.require_fast_tokenizer: bool = bool(require_fast_tokenizer)
        self.trust_remote_code: bool = bool(trust_remote_code)
        self.local_files_only: bool | None = (
            None if local_files_only is None else bool(local_files_only)
        )
        super().__init__(
            family=family,
            model_name=model_name,
            huggingface_model_class=huggingface_model_class,
            query_pooling=query_pooling,
            doc_pooling=doc_pooling,
            sparse_activation=sparse_activation,
            attn_implementation=attn_implementation,
            dtype=dtype,
            normalize=normalize,
            doc_only=doc_only,
            tie_word_embeddings=tie_word_embeddings,
            peft_cfg=peft_cfg,
            freeze_backbone=freeze_backbone,
            trust_remote_code=trust_remote_code,
            model_revision=model_revision,
            local_files_only=local_files_only,
            exclude_token_ids=exclude_token_ids,
            mask_token_id=mask_token_id,
        )
        tokenizer: PreTrainedTokenizerBase = self._build_configured_tokenizer(
            self.tokenizer_source,
            revision=self.tokenizer_revision,
        )
        self._validate_tokenizer_matches_model(tokenizer)
        if self.enforce_same_tokenizer_as_baseline:
            baseline_source: str = (
                self.baseline_tokenizer_name or "distilbert-base-uncased"
            )
            baseline_tokenizer: PreTrainedTokenizerBase = (
                self._build_configured_tokenizer(baseline_source, revision=None)
            )
            self._validate_tokenizer_matches_baseline(
                tokenizer=tokenizer,
                baseline_tokenizer=baseline_tokenizer,
                baseline_source=baseline_source,
            )

    def _build_configured_tokenizer(
        self,
        source: str,
        *,
        revision: str | None,
    ) -> PreTrainedTokenizerBase:
        return build_tokenizer(
            source,
            use_fast_tokenizer=self.use_fast_tokenizer,
            trust_remote_code=self.trust_remote_code,
            require_fast_tokenizer=self.require_fast_tokenizer,
            local_files_only=self.local_files_only,
            revision=revision,
        )

    def _validate_tokenizer_matches_model(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        tokenizer_vocab_size: int = int(len(tokenizer))
        model_vocab_size: int = int(self.encoder.vocab_size)
        if tokenizer_vocab_size != model_vocab_size:
            raise ValueError(
                "PretrainedDiffusionSpladeModel requires tokenizer and model "
                "vocabulary sizes to match. "
                f"tokenizer={tokenizer_vocab_size}, model={model_vocab_size}, "
                f"tokenizer_source={self.tokenizer_source!r}."
            )

    def _validate_tokenizer_matches_baseline(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        baseline_tokenizer: PreTrainedTokenizerBase,
        baseline_source: str,
    ) -> None:
        tokenizer_vocab: dict[str, int] = dict(tokenizer.get_vocab())
        baseline_vocab: dict[str, int] = dict(baseline_tokenizer.get_vocab())
        if tokenizer_vocab != baseline_vocab:
            raise ValueError(
                "Primary diffusion-backbone ablations require the same tokenizer "
                "vocabulary as the baseline. "
                f"tokenizer_source={self.tokenizer_source!r}, "
                f"baseline_tokenizer_source={baseline_source!r}."
            )
