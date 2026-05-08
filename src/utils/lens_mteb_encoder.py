"""MTEB 2.x-compatible encoder wrapper for LENS.

Mirrors ``Yibin-Lei/LENS/eval/model.py::LENSModel`` semantics exactly but
implements MTEB 2.x's :class:`mteb.models.abs_encoder.AbsEncoder` interface so
that ``mteb.evaluate(model, task, ...)`` works out of the box.

For every task family, the query side uses the paper's detailed-instruction
template::

    <instruct>{task_instruction}\\n<query>{text}

and the corpus side uses plain text with last-two-token zeroed pooling. For
non-retrieval tasks (classification / clustering / STS / summ / pair) MTEB
calls the encoder with ``prompt_type=None``; we still apply the query-side
template with the per-task instruction, matching the official repo.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Union, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

from src.utils.lens_mteb_instructions import task_to_instruction
from src.utils.lens_official_loader import load_official_lens

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs


_QUERY_TOKEN: str = "<query>"
_SUFFIX: str = "</s>"


def _pooling_log1p_relu_max(vecs: Tensor, pooling_mask: Tensor) -> Tensor:
    """Official LENS pooling: max over log(1+relu(logits)) masked by `pooling_mask`."""
    activated: Tensor = torch.log1p(torch.relu(vecs))
    masked: Tensor = activated * pooling_mask.unsqueeze(-1).to(dtype=activated.dtype)
    return torch.max(masked, dim=1).values


def _text_length(text: Any) -> int:
    if isinstance(text, dict):
        return len(next(iter(text.values())))
    if not hasattr(text, "__len__"):
        return 1
    if len(text) == 0 or isinstance(text[0], int):
        return len(text)
    return sum(len(t) for t in text)


def _build_detailed_instruct(instruction: str, query: str) -> str:
    return f"<instruct>{instruction}\n{_QUERY_TOKEN}{query}"


def _build_query_pooling_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    query_token_id: int,
) -> Tensor:
    """Mirror official LENS query pooling: [<query>+1 ... -2]."""
    mask: Tensor = torch.zeros_like(attention_mask)
    for idx in range(input_ids.size(0)):
        seq = input_ids[idx]
        special_pos = (seq == query_token_id).nonzero()
        if len(special_pos) > 0:
            last_pos: int = int(special_pos[-1].item())
            mask[idx, last_pos:-2] = 1
        else:
            mask[idx] = attention_mask[idx]
    return mask


def _extract_texts(inputs: Any) -> List[str]:
    """Extract the list of strings from an MTEB BatchedInput stream or a flat list."""
    # MTEB passes a DataLoader[BatchedInput]; each batch has a "text" key.
    if hasattr(inputs, "__iter__") and not isinstance(inputs, (str, dict)):
        texts: List[str] = []
        for batch in inputs:
            if isinstance(batch, dict):
                text_field = batch.get("text", batch.get("sentence", None))
                if text_field is None:
                    # Fallback: concatenate all string-like values.
                    for v in batch.values():
                        if isinstance(v, list) and v and isinstance(v[0], str):
                            text_field = v
                            break
                if text_field is None:
                    continue
                if isinstance(text_field, str):
                    texts.append(text_field)
                else:
                    texts.extend(str(t) for t in text_field)
            elif isinstance(batch, str):
                texts.append(batch)
            else:
                # Plain iterable of strings
                texts.extend(str(t) for t in batch)
        return texts
    if isinstance(inputs, str):
        return [inputs]
    raise TypeError(f"Cannot extract texts from {type(inputs).__name__}")


class LENSMTEBEncoder(AbsEncoder):
    """MTEB encoder wrapping a LENS-style bidirectional Mistral + clustered head.

    Implements :class:`mteb.models.abs_encoder.AbsEncoder` so MTEB 2.x's
    ``evaluate(model, task)`` entrypoint treats it as a first-class encoder.

    Parameters
    ----------
    model_name_or_path : str
        HF repo id of the official LENS checkpoint, e.g. ``yibinlei/LENS-d8000``.
    batch_size : int
        Default encoding batch size.
    max_length : int
        Token truncation length for both queries and passages (paper uses 512).
    dtype : torch.dtype
        Parameter dtype (fp16 reproduces the paper; bf16 is a viable swap on A100).
    attn_implementation : str
        ``"flash_attention_2"`` or ``"sdpa"`` fallback.
    normalize_embeddings : bool
        L2-normalize the sparse embeddings. Official LENS uses True.
    device : torch.device | str | None
        Target CUDA device; ``None`` → auto-detect.
    """

    mteb_model_meta: ModelMeta

    def __init__(
        self,
        model_name_or_path: str,
        *,
        batch_size: int = 32,
        max_length: int = 512,
        dtype: torch.dtype = torch.float16,
        attn_implementation: str = "sdpa",
        normalize_embeddings: bool = True,
        device: Union[torch.device, str, None] = None,
        default_instruction: str = (
            "Given a query, retrieval relevant passages that answer the query."
        ),
        revision: str | None = None,
    ) -> None:
        self.model_name_or_path: str = model_name_or_path
        self.batch_size: int = int(batch_size)
        self.max_length: int = int(max_length)
        self.normalize_embeddings: bool = bool(normalize_embeddings)
        self.default_instruction: str = str(default_instruction)

        if device is None:
            resolved_device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            resolved_device = torch.device(device)
        self.device: torch.device = resolved_device

        backbone, tokenizer = load_official_lens(
            model_name_or_path,
            dtype=dtype,
            device=resolved_device,
            attn_implementation=attn_implementation,
            strict_official_tokenizer=True,
        )
        self.model = backbone  # AbsEncoder attribute
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self._query_token_id: int = int(
            self.tokenizer.convert_tokens_to_ids(_QUERY_TOKEN)
        )
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(name=model_name_or_path, revision=revision)
        )

    # --- AbsEncoder interface ------------------------------------------------

    def similarity(self, embeddings1: "Array", embeddings2: "Array") -> "Array":
        a: Tensor = torch.as_tensor(embeddings1, dtype=torch.float32)
        b: Tensor = torch.as_tensor(embeddings2, dtype=torch.float32)
        return (a @ b.T).numpy()

    def encode(
        self,
        inputs: "DataLoader[BatchedInput]",
        *,
        task_metadata: "TaskMetadata",
        hf_split: str,
        hf_subset: str,
        prompt_type: Union["PromptType", None] = None,
        **kwargs: Any,
    ) -> "Array":
        _ = hf_split, hf_subset  # not used, present for protocol compat
        sentences: List[str] = _extract_texts(inputs)
        batch_size: int = int(kwargs.get("batch_size") or self.batch_size)
        task_name: str = str(task_metadata.name)
        task_type: str = str(getattr(task_metadata, "type", "") or "")

        # Route corpus (document) vs query mode.
        is_document: bool = False
        try:
            is_document = (prompt_type == PromptType.document)
        except Exception:
            # Older MTEB versions
            is_document = str(prompt_type) in ("document", "passage")

        if is_document and task_type.lower() == "retrieval":
            # Doc-side: plain text + last-2-zeroed attention mask.
            return self._encode(sentences, mode="doc", batch_size=batch_size)

        # Query side for every other case. Include non-retrieval (classification,
        # clustering, STS, summarization, pair-classification, reranking) — for
        # these the official LENS pipeline always applies the detailed-instruct
        # template on both "sides" of the task.
        instruction: str = task_to_instruction(task_name, is_query=True) or \
            self.default_instruction
        detailed: List[str] = [
            _build_detailed_instruct(instruction, s) for s in sentences
        ]
        return self._encode(detailed, mode="query", batch_size=batch_size)

    # --- Private -------------------------------------------------------------

    @torch.no_grad()
    def _encode(
        self,
        sentences: Sequence[str],
        *,
        mode: str,
        batch_size: int,
    ) -> np.ndarray:
        self.model.eval()
        length_sorted_idx: np.ndarray = np.argsort(
            [-_text_length(s) for s in sentences]
        )
        sentences_sorted: List[str] = [sentences[i] for i in length_sorted_idx]

        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(sentences_sorted), batch_size):
            batch: List[str] = sentences_sorted[start : start + batch_size]
            batch_with_suffix: List[str] = [s + _SUFFIX for s in batch]
            enc = self.tokenizer(
                batch_with_suffix,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
                add_special_tokens=True,
            ).to(self.device)

            pooling_mask: Tensor
            if mode == "query":
                pooling_mask = _build_query_pooling_mask(
                    enc["input_ids"],
                    enc["attention_mask"],
                    self._query_token_id,
                )
            else:
                attn: Tensor = enc["attention_mask"].clone()
                attn[:, -2:] = 0
                pooling_mask = attn

            outputs = self.model(**enc, return_dict=True)
            logits: Tensor = outputs.logits
            embeddings: Tensor = _pooling_log1p_relu_max(logits, pooling_mask)
            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.float().cpu().numpy())

        stacked: np.ndarray = np.concatenate(all_embeddings, axis=0)
        inverse: np.ndarray = np.argsort(length_sorted_idx)
        return stacked[inverse]

    # --- Legacy helpers (keep old API for smoke tests / manual CLI use) ------

    def set_instruction(self, instruction: str) -> None:
        self.default_instruction = str(instruction)

    def encode_queries(
        self,
        queries: Sequence[str],
        *,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        _ = kwargs
        if isinstance(queries, str):
            queries = [queries]
        detailed: List[str] = [
            _build_detailed_instruct(self.default_instruction, q) for q in queries
        ]
        return self._encode(
            detailed,
            mode="query",
            batch_size=batch_size or self.batch_size,
        )

    def encode_corpus(
        self,
        corpus: Any,
        *,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        _ = kwargs
        sentences: List[str] = _corpus_to_texts(corpus)
        return self._encode(
            sentences,
            mode="doc",
            batch_size=batch_size or self.batch_size,
        )


def _corpus_to_texts(corpus: Any) -> List[str]:
    """Convert MTEB's varied corpus payload into a list of strings."""
    if isinstance(corpus, list) and corpus and isinstance(corpus[0], str):
        return list(corpus)
    if isinstance(corpus, list) and corpus and isinstance(corpus[0], dict):
        texts: List[str] = []
        for row in corpus:
            title: str = str(row.get("title", "") or "")
            body: str = str(row.get("text", "") or row.get("body", "") or "")
            if title:
                texts.append(f"{title} {body}".strip())
            else:
                texts.append(body)
        return texts
    if isinstance(corpus, dict):
        titles: Iterable[str] = corpus.get("title", [])
        bodies: Iterable[str] = corpus.get("text", corpus.get("body", []))
        return [
            (f"{t} {b}".strip() if t else str(b))
            for t, b in zip(titles, bodies)
        ]
    if isinstance(corpus, str):
        return [corpus]
    raise TypeError(f"Unsupported corpus payload: {type(corpus).__name__}")


__all__ = ["LENSMTEBEncoder"]
