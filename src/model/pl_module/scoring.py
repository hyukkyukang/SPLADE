import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import lightning as L
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, T5Config, T5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.model.pl_module.utils import resolve_cudagraph_mark_step, validate_torch_compile_mode
from src.utils.logging import get_logger, log_if_rank_zero
from src.utils.script_setup import normalize_optional_str

logger = get_logger("CrossEncoderScoringModule")


@dataclass(frozen=True)
class _RowPayload:
    row: dict[str, Any]
    qid: str
    doc_ids: list[str]
    labels: list[float] | None
    doc_sources: list[str] | None


@dataclass
class _RowState:
    payload: _RowPayload
    scores: list[float]
    remaining: int


def _append_rank_suffix(path: Path, rank: int) -> Path:
    suffix: str = path.suffix
    stem: str = path.stem
    if suffix:
        return path.with_name(f"{stem}.rank{rank}{suffix}")
    return path.with_name(f"{path.name}.rank{rank}")


def _extract_scores_tensor(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits to a detached 1D CPU tensor."""
    if logits.ndim == 2 and logits.shape[1] > 1:
        scores_tensor: torch.Tensor = logits[:, 0]
    else:
        scores_tensor = logits.squeeze(-1)
    return scores_tensor.detach().to(device="cpu", dtype=torch.float32)


def _strip_score_fields(row: dict[str, Any], score_key: str) -> dict[str, Any]:
    drop_keys: set[str] = {"score", "scores", score_key}
    return {key: value for key, value in row.items() if key not in drop_keys}


class T5EncoderRerank(torch.nn.Module):
    def __init__(
        self,
        base_model_name: str,
        checkpoint_path: str,
        *,
        config_name: str | None = None,
    ) -> None:
        super().__init__()
        if config_name is not None:
            config = T5Config.from_pretrained(config_name)
            self.model = T5EncoderModel(config)
        else:
            self.model = T5EncoderModel.from_pretrained(base_model_name)
        self.config = self.model.config
        hidden_size: int = int(self.config.d_model)
        self.first_transform = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.linear = torch.nn.Linear(hidden_size, 1)
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state, strict=True)

    def forward(self, **kwargs: Any) -> SequenceClassifierOutput:
        result: torch.Tensor = self.model(**kwargs).last_hidden_state[:, 0, :]
        first_transformed: torch.Tensor = self.first_transform(result)
        layer_normed: torch.Tensor = self.layer_norm(first_transformed)
        logits: torch.Tensor = self.linear(layer_normed)
        return SequenceClassifierOutput(logits=logits)


class CrossEncoderScoringModule(L.LightningModule):
    """LightningModule for cross-encoder scoring."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.cfg: DictConfig = cfg
        scoring_cfg: DictConfig = cfg.scoring

        self.model_name: str = str(scoring_cfg.model_name)
        self.batch_size: int = int(scoring_cfg.batch_size)
        if self.batch_size <= 0:
            raise ValueError("scoring.batch_size must be a positive integer.")
        self.max_length: int = int(scoring_cfg.max_length)
        self.score_key: str = str(scoring_cfg.score_key)
        self.output_dir: str = str(scoring_cfg.output_dir)
        self.output_basename: str = str(scoring_cfg.output_basename)
        self.output_format: str = str(scoring_cfg.output_format).lower()
        self.overwrite: bool = bool(scoring_cfg.overwrite)
        self.flush_every: int = max(int(scoring_cfg.flush_every), 1)

        self.model_backend: str = str(scoring_cfg.model_backend).lower()
        checkpoint_path: str | None = normalize_optional_str(
            scoring_cfg.model_checkpoint_path
        )
        if self.model_backend == "auto_seq_cls":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
        elif self.model_backend == "t5_encoder_rerank":
            if checkpoint_path is None:
                raise ValueError(
                    "scoring.model_checkpoint_path must be set for "
                    "model_backend=t5_encoder_rerank."
                )
            config_name: str | None = normalize_optional_str(
                scoring_cfg.model_config_name
            )
            self.model = T5EncoderRerank(
                self.model_name,
                checkpoint_path,
                config_name=config_name,
            )
        else:
            raise ValueError(
                "scoring.model_backend must be one of: auto_seq_cls, "
                f"t5_encoder_rerank. Got: {self.model_backend}"
            )
        self._torch_compile_mark_step: Callable[[], None] | None = None
        self._setup_torch_compile()

        self._output_handle: Any | None = None
        self._rows_written: int = 0

    def on_predict_start(self) -> None:
        if self.output_format != "jsonl":
            raise ValueError("Only jsonl output is supported for scoring.")
        output_path: Path = Path(self.output_dir) / f"{self.output_basename}.jsonl"
        if int(self.trainer.world_size) > 1:
            output_path = _append_rank_suffix(
                output_path, int(self.trainer.global_rank)
            )
        if output_path.exists() and not self.overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_handle = output_path.open("w", encoding="utf-8")
        self._rows_written = 0
        self.model.eval()
        log_if_rank_zero(logger, f"Writing scored rows to {output_path}")

    def on_predict_end(self) -> None:
        if self._output_handle is None:
            return
        self._output_handle.flush()
        self._output_handle.close()
        self._output_handle = None

    def predict_step(self, batch: dict[str, Any] | None, batch_idx: int) -> None:
        _ = batch_idx
        if not batch:
            return
        if "pair_tokens" not in batch:
            raise ValueError("Dataloader must provide pair_tokens for scoring.")
        self._write_rows(self._score_tokenized_batch(batch))

    def _setup_torch_compile(self) -> None:
        scoring_cfg: DictConfig = self.cfg.scoring
        compile_enabled: bool = bool(scoring_cfg.torch_compile)
        compile_available: bool = hasattr(torch, "compile")
        self._torch_compile_mark_step = None
        if compile_enabled and not compile_available:
            log_if_rank_zero(
                logger,
                "torch.compile is not available in this PyTorch build; continuing "
                "without compilation.",
                level="warning",
            )
            return
        if not compile_enabled or not compile_available:
            return
        compile_mode, compile_mode_kwargs = validate_torch_compile_mode(
            scoring_cfg.torch_compile_mode
        )
        if compile_mode in {"reduce-overhead", "max-autotune"}:
            self._torch_compile_mark_step = resolve_cudagraph_mark_step()
        self.model = torch.compile(self.model, **compile_mode_kwargs)

    def _score_tokenized_batch(
        self, batch: dict[str, Any]
    ) -> Iterable[dict[str, Any]]:
        rows: list[dict[str, Any]] = batch["rows"]
        qids: list[str] = batch["qids"]
        doc_ids_list: list[list[str]] = batch["doc_ids"]
        labels_list: list[list[float] | None] = batch["labels"]
        doc_sources_list: list[list[str] | None] = batch["doc_sources"]
        pair_row_ids: list[int] = batch["pair_row_ids"]
        pair_doc_idxs: list[int] = batch["pair_doc_idxs"]
        pair_tokens: dict[str, torch.Tensor] = batch["pair_tokens"]
        if not pair_row_ids:
            return []

        expected_pairs: int = sum(len(doc_ids) for doc_ids in doc_ids_list)
        if len(pair_row_ids) != expected_pairs or len(pair_doc_idxs) != expected_pairs:
            raise ValueError(
                "Tokenized pairs are misaligned with doc_ids. "
                f"Expected {expected_pairs} pairs, got "
                f"{len(pair_row_ids)} row ids and {len(pair_doc_idxs)} doc idxs."
            )
        token_batch_size: int | None = None
        for key, value in pair_tokens.items():
            if value.ndim == 0:
                raise ValueError(f"pair_tokens[{key}] must be batched.")
            token_batch_size = int(value.shape[0]) if token_batch_size is None else token_batch_size
            if int(value.shape[0]) != token_batch_size:
                raise ValueError(
                    "pair_tokens batch size mismatch for key "
                    f"{key}: expected {token_batch_size}, got {value.shape[0]}."
                )
        if token_batch_size is not None and token_batch_size != expected_pairs:
            raise ValueError(
                "pair_tokens batch size does not match doc_ids. "
                f"Expected {expected_pairs}, got {token_batch_size}."
            )

        row_queue: deque[int] = deque()
        row_states: dict[int, _RowState] = {}

        for row_idx, (row, qid, doc_ids, labels, doc_sources) in enumerate(
            zip(rows, qids, doc_ids_list, labels_list, doc_sources_list)
        ):
            row_states[row_idx] = _RowState(
                payload=_RowPayload(
                    row=row,
                    qid=qid,
                    doc_ids=doc_ids,
                    labels=labels,
                    doc_sources=doc_sources,
                ),
                scores=[0.0] * len(doc_ids),
                remaining=len(doc_ids),
            )
            row_queue.append(row_idx)

        def _flush_ready() -> Iterable[dict[str, Any]]:
            while row_queue and row_states[row_queue[0]].remaining == 0:
                row_id = row_queue.popleft()
                state = row_states.pop(row_id)
                payload = state.payload
                output_row: dict[str, Any] = {
                    "query_id": payload.qid,
                    "doc_ids": payload.doc_ids,
                    "labels": payload.labels,
                    "doc_sources": payload.doc_sources,
                    self.score_key: state.scores,
                }
                yield output_row

        total_pairs: int = len(pair_row_ids)
        start_idx = 0
        pair_token_items: list[tuple[str, torch.Tensor]] = list(pair_tokens.items())
        token_batch: dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            while start_idx < total_pairs:
                end_idx = min(start_idx + self.batch_size, total_pairs)
                token_batch.clear()
                key: str
                value: torch.Tensor
                for key, value in pair_token_items:
                    token_batch[key] = value[start_idx:end_idx].to(
                        self.device, non_blocking=True
                    )
                if self._torch_compile_mark_step is not None:
                    self._torch_compile_mark_step()
                outputs = self.model(**token_batch)
                batch_scores: torch.Tensor = _extract_scores_tensor(outputs.logits)
                offset: int
                for offset in range(int(batch_scores.shape[0])):
                    pair_idx = start_idx + offset
                    row_id = pair_row_ids[pair_idx]
                    doc_idx = pair_doc_idxs[pair_idx]
                    state = row_states[row_id]
                    state.scores[doc_idx] = float(batch_scores[offset].item())
                    state.remaining -= 1
                yield from _flush_ready()
                start_idx = end_idx
            yield from _flush_ready()

    def _write_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        if self._output_handle is None:
            raise RuntimeError("Output handle is not initialized.")
        for row in rows:
            self._output_handle.write(json.dumps(row) + "\n")
            self._rows_written += 1
            if self._rows_written % self.flush_every == 0:
                self._output_handle.flush()
