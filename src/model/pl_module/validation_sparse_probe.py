from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from src.data.dataclass import TrainingDataItem
from src.utils.logging import log_if_rank_zero


class ValidationSparseProbeLogger:
    """Log fixed human-readable sparse validation probes as local/MLflow artifacts."""

    def __init__(self, *, module: Any, cfg: DictConfig, logger: Any) -> None:
        self._module: Any = module
        self._cfg: DictConfig = cfg
        self._logger: Any = logger
        probe_cfg: DictConfig | None = cfg.training.get("validation_sparse_probe")
        self._probe_cfg: DictConfig | None = probe_cfg
        self.enabled: bool = bool(
            probe_cfg is not None and bool(probe_cfg.get("enabled", False))
        )
        self._num_pairs: int = (
            0 if probe_cfg is None else max(int(probe_cfg.get("num_pairs", 10)), 1)
        )
        self._top_k_sparse: int = (
            0 if probe_cfg is None else max(int(probe_cfg.get("top_k_sparse", 20)), 1)
        )
        self._top_k_slot: int = (
            0 if probe_cfg is None else max(int(probe_cfg.get("top_k_slot", 10)), 1)
        )
        self._include_slot_logits: bool = bool(
            False if probe_cfg is None else probe_cfg.get("include_slot_logits", True)
        )
        self._log_every_n_val: int = (
            1
            if probe_cfg is None
            else max(int(probe_cfg.get("log_every_n_val", 1)), 1)
        )
        self._selection_seed: int = (
            int(cfg.seed)
            if probe_cfg is None
            else int(probe_cfg.get("selection_seed", int(cfg.seed)))
        )
        self._artifact_dir_name: str = (
            "validation_sparse_probe"
            if probe_cfg is None
            else str(probe_cfg.get("artifact_dir", "validation_sparse_probe")).strip()
            or "validation_sparse_probe"
        )
        self._selection_filename: str = (
            "selection.json"
            if probe_cfg is None
            else str(
                probe_cfg.get("persist_selection_filename", "selection.json")
            ).strip()
            or "selection.json"
        )
        self._write_json: bool = bool(
            True if probe_cfg is None else probe_cfg.get("write_json", True)
        )
        self._write_markdown: bool = bool(
            True if probe_cfg is None else probe_cfg.get("write_markdown", True)
        )
        self._log_to_mlflow: bool = bool(
            True if probe_cfg is None else probe_cfg.get("log_to_mlflow", True)
        )
        self._mlflow_timeout_seconds: int = (
            10
            if probe_cfg is None
            else max(int(probe_cfg.get("mlflow_timeout_seconds", 10)), 1)
        )
        self._mlflow_max_retries: int = (
            0
            if probe_cfg is None
            else max(int(probe_cfg.get("mlflow_max_retries", 0)), 0)
        )
        self._mlflow_backoff_factor: int = (
            0
            if probe_cfg is None
            else max(int(probe_cfg.get("mlflow_backoff_factor", 0)), 0)
        )
        self._disable_mlflow_after_failure: bool = bool(
            True
            if probe_cfg is None
            else probe_cfg.get("disable_mlflow_after_failure", True)
        )
        raw_indices: Any = None if probe_cfg is None else probe_cfg.get("probe_indices")
        self._configured_probe_indices: list[int] | None = (
            None
            if raw_indices is None
            else [int(index) for index in raw_indices][: self._num_pairs]
        )
        self._cached_probe_indices: list[int] | None = None
        self._validation_counter: int = 0
        self._mlflow_upload_disabled: bool = False

    def run_validation_epoch_end(self) -> None:
        if not self._should_run_now():
            return
        dataset: Any | None = self._resolve_validation_dataset()
        if dataset is None:
            return
        tokenizer: PreTrainedTokenizerBase | None = self._resolve_tokenizer()
        if tokenizer is None:
            log_if_rank_zero(
                self._logger,
                "Skipping validation sparse probe logging because the tokenizer "
                "could not be resolved.",
                level="warning",
            )
            return
        probe_indices: list[int] = self._resolve_probe_indices(dataset)
        if not probe_indices:
            log_if_rank_zero(
                self._logger,
                "Skipping validation sparse probe logging because no valid "
                "validation probe pairs were found.",
                level="warning",
            )
            return

        samples: list[dict[str, Any]] = self._build_probe_samples(
            dataset=dataset,
            tokenizer=tokenizer,
            probe_indices=probe_indices,
        )
        if not samples:
            log_if_rank_zero(
                self._logger,
                "Skipping validation sparse probe logging because sampled probe "
                "pairs could not be materialized.",
                level="warning",
            )
            return

        payload: dict[str, Any] = {
            "global_step": int(self._module.global_step),
            "epoch": int(getattr(self._module, "current_epoch", 0)),
            "probe_indices": probe_indices,
            "samples": samples,
        }
        output_dir: Path = Path(str(self._cfg.log_dir)) / self._artifact_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths: list[Path] = []
        step_stem: str = f"step_{int(self._module.global_step):08d}"
        if self._write_json:
            json_path: Path = output_dir / f"{step_stem}.json"
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            artifact_paths.append(json_path)
        if self._write_markdown:
            markdown_path: Path = output_dir / f"{step_stem}.md"
            markdown_path.write_text(
                self._render_markdown(payload),
                encoding="utf-8",
            )
            artifact_paths.append(markdown_path)
        self._log_artifacts_to_mlflow(artifact_paths)
        log_if_rank_zero(
            self._logger,
            "Saved validation sparse probe artifacts to "
            f"{output_dir.as_posix()} for step {int(self._module.global_step)}.",
        )

    def _should_run_now(self) -> bool:
        if not self.enabled:
            return False
        trainer: Any | None = getattr(self._module, "trainer", None)
        if trainer is None:
            return False
        if not bool(getattr(trainer, "is_global_zero", False)):
            return False
        if bool(getattr(trainer, "sanity_checking", False)):
            return False
        self._validation_counter += 1
        return self._validation_counter % self._log_every_n_val == 0

    def _resolve_validation_dataset(self) -> Any | None:
        trainer: Any | None = getattr(self._module, "trainer", None)
        datamodule: Any | None = None if trainer is None else getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None
        return getattr(datamodule, "val_dataset", None)

    def _resolve_tokenizer(self) -> PreTrainedTokenizerBase | None:
        trainer: Any | None = getattr(self._module, "trainer", None)
        datamodule: Any | None = None if trainer is None else getattr(trainer, "datamodule", None)
        tokenizer: Any | None = None if datamodule is None else getattr(datamodule, "tokenizer", None)
        if tokenizer is None:
            return None
        return tokenizer

    def _selection_path(self) -> Path:
        return Path(str(self._cfg.log_dir)) / self._artifact_dir_name / self._selection_filename

    def _resolve_probe_indices(self, dataset: Any) -> list[int]:
        if self._cached_probe_indices is not None:
            return list(self._cached_probe_indices)
        if self._configured_probe_indices is not None:
            resolved_indices: list[int] = self._validate_probe_indices(
                dataset=dataset,
                candidate_indices=self._configured_probe_indices,
            )
            self._cached_probe_indices = resolved_indices
            return list(resolved_indices)
        selection_path: Path = self._selection_path()
        if selection_path.is_file():
            try:
                payload: dict[str, Any] = json.loads(
                    selection_path.read_text(encoding="utf-8")
                )
                candidate_indices: list[int] = [
                    int(index) for index in payload.get("indices", [])
                ]
                resolved_indices = self._validate_probe_indices(
                    dataset=dataset,
                    candidate_indices=candidate_indices,
                )
                self._cached_probe_indices = resolved_indices
                return list(resolved_indices)
            except Exception as exc:
                log_if_rank_zero(
                    self._logger,
                    "Failed to load persisted validation sparse probe selection; "
                    f"recomputing it. Error: {exc}",
                    level="warning",
                )
        resolved_indices = self._sample_probe_indices(dataset)
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_payload: dict[str, Any] = {
            "indices": resolved_indices,
            "num_pairs": len(resolved_indices),
            "selection_seed": int(self._selection_seed),
            "dataset_length": int(len(dataset)),
        }
        selection_path.write_text(
            json.dumps(selection_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._cached_probe_indices = resolved_indices
        return list(resolved_indices)

    def _validate_probe_indices(
        self,
        *,
        dataset: Any,
        candidate_indices: list[int],
    ) -> list[int]:
        dataset_length: int = int(len(dataset))
        resolved_indices: list[int] = []
        seen: set[int] = set()
        candidate_index: int
        for candidate_index in candidate_indices:
            if candidate_index in seen:
                continue
            if candidate_index < 0 or candidate_index >= dataset_length:
                continue
            item: Any = dataset[candidate_index]
            if not self._item_has_positive_pair(item):
                continue
            resolved_indices.append(candidate_index)
            seen.add(candidate_index)
            if len(resolved_indices) >= self._num_pairs:
                break
        return resolved_indices

    def _sample_probe_indices(self, dataset: Any) -> list[int]:
        dataset_length: int = int(len(dataset))
        shuffled_indices: list[int] = list(range(dataset_length))
        random.Random(self._selection_seed).shuffle(shuffled_indices)
        selected: list[int] = []
        index: int
        for index in shuffled_indices:
            if len(selected) >= self._num_pairs:
                break
            item: Any = dataset[index]
            if not self._item_has_positive_pair(item):
                continue
            selected.append(index)
        return selected

    def _item_has_positive_pair(self, item: Any) -> bool:
        pos_mask: Any | None = getattr(item, "pos_mask", None)
        if not isinstance(pos_mask, torch.Tensor) or pos_mask.numel() == 0:
            return False
        return bool(pos_mask.any())

    def _build_probe_samples(
        self,
        *,
        dataset: Any,
        tokenizer: PreTrainedTokenizerBase,
        probe_indices: list[int],
    ) -> list[dict[str, Any]]:
        items: list[TrainingDataItem] = [
            item
            for item in (dataset[index] for index in probe_indices)
            if isinstance(item, TrainingDataItem)
        ]
        if not items:
            return []
        batch: dict[str, Any] = dataset.collator(items)
        query_input_ids: torch.Tensor = batch["query_input_ids"].to(self._module.device)
        query_attention_mask: torch.Tensor = batch["query_attention_mask"].to(
            self._module.device
        )
        query_pooling_mask: torch.Tensor | None = batch.get("query_pooling_mask")
        if query_pooling_mask is not None:
            query_pooling_mask = query_pooling_mask.to(self._module.device)

        positive_positions: list[int] = []
        probe_metadata: list[dict[str, Any]] = []
        item: TrainingDataItem
        for dataset_index, item in zip(probe_indices, items):
            positive_candidates: torch.Tensor = torch.nonzero(
                item.pos_mask, as_tuple=False
            ).flatten()
            if int(positive_candidates.numel()) <= 0:
                continue
            positive_position: int = int(positive_candidates[0].item())
            doc_ids: list[str] = list(item.pos_ids) + list(item.neg_ids)
            doc_id: str = (
                doc_ids[positive_position]
                if positive_position < len(doc_ids)
                else ""
            )
            probe_metadata.append(
                {
                    "dataset_index": int(dataset_index),
                    "qid": str(item.qid),
                    "doc_id": doc_id,
                    "query_text": str(item.query_text),
                    "doc_text": str(item.doc_texts[positive_position]),
                }
            )
            positive_positions.append(positive_position)
        if not positive_positions:
            return []

        positive_index_tensor: torch.Tensor = torch.tensor(
            positive_positions,
            dtype=torch.long,
            device=self._module.device,
        )
        batch_index_tensor: torch.Tensor = torch.arange(
            len(positive_positions),
            dtype=torch.long,
            device=self._module.device,
        )
        doc_input_ids: torch.Tensor = batch["doc_input_ids"][
            batch_index_tensor, positive_index_tensor
        ].to(self._module.device)
        doc_attention_mask: torch.Tensor = batch["doc_attention_mask"][
            batch_index_tensor, positive_index_tensor
        ].to(self._module.device)
        doc_pooling_mask: torch.Tensor | None = batch.get("doc_pooling_mask")
        resolved_doc_pooling_mask: torch.Tensor | None = None
        if doc_pooling_mask is not None:
            resolved_doc_pooling_mask = doc_pooling_mask[
                batch_index_tensor.cpu(), positive_index_tensor.cpu()
            ].to(self._module.device)

        supports_slot_logging: bool = bool(
            self._include_slot_logits
            and getattr(self._module.model, "supports_ordered_mask_slot_loss", False)
        )
        with torch.no_grad():
            if supports_slot_logging:
                query_reps, query_slot_logits = (
                    self._module.model.encode_queries_with_slot_logits(
                        query_input_ids,
                        query_attention_mask,
                        pooling_mask=query_pooling_mask,
                    )
                )
                doc_reps, doc_slot_logits = (
                    self._module.model.encode_docs_with_slot_logits(
                        doc_input_ids,
                        doc_attention_mask,
                        pooling_mask=resolved_doc_pooling_mask,
                    )
                )
            else:
                query_reps = self._module.model.encode_queries(
                    query_input_ids,
                    query_attention_mask,
                    pooling_mask=query_pooling_mask,
                )
                doc_reps = self._module.model.encode_docs(
                    doc_input_ids,
                    doc_attention_mask,
                    pooling_mask=resolved_doc_pooling_mask,
                )
                query_slot_logits = None
                doc_slot_logits = None

        samples: list[dict[str, Any]] = []
        sample_idx: int
        metadata: dict[str, Any]
        for sample_idx, metadata in enumerate(probe_metadata):
            sample_payload: dict[str, Any] = {
                "probe_rank": int(sample_idx),
                "dataset_index": int(metadata["dataset_index"]),
                "qid": str(metadata["qid"]),
                "doc_id": str(metadata["doc_id"]),
                "query_text": str(metadata["query_text"]),
                "doc_text": str(metadata["doc_text"]),
                "query_top_sparse": self._extract_top_tokens(
                    vector=query_reps[sample_idx],
                    tokenizer=tokenizer,
                    top_k=self._top_k_sparse,
                    positive_only=True,
                ),
                "doc_top_sparse": self._extract_top_tokens(
                    vector=doc_reps[sample_idx],
                    tokenizer=tokenizer,
                    top_k=self._top_k_sparse,
                    positive_only=True,
                ),
            }
            if query_slot_logits is not None and doc_slot_logits is not None:
                sample_payload["query_slot_topk"] = self._extract_slot_top_tokens(
                    slot_logits=query_slot_logits[sample_idx],
                    tokenizer=tokenizer,
                    top_k=self._top_k_slot,
                )
                sample_payload["doc_slot_topk"] = self._extract_slot_top_tokens(
                    slot_logits=doc_slot_logits[sample_idx],
                    tokenizer=tokenizer,
                    top_k=self._top_k_slot,
                )
            samples.append(sample_payload)
        return samples

    def _extract_top_tokens(
        self,
        *,
        vector: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        top_k: int,
        positive_only: bool,
    ) -> list[dict[str, Any]]:
        flattened_vector: torch.Tensor = vector.detach().float().cpu().reshape(-1)
        if positive_only:
            active_mask: torch.Tensor = flattened_vector > 0
            active_indices: torch.Tensor = torch.nonzero(
                active_mask, as_tuple=False
            ).flatten()
            if int(active_indices.numel()) <= 0:
                return []
            active_values: torch.Tensor = flattened_vector[active_indices]
            top_count: int = min(int(top_k), int(active_values.numel()))
            top_values, top_positions = torch.topk(active_values, k=top_count)
            top_indices: torch.Tensor = active_indices[top_positions]
        else:
            top_count = min(int(top_k), int(flattened_vector.numel()))
            top_values, top_indices = torch.topk(flattened_vector, k=top_count)
        return self._token_entries_from_topk(
            tokenizer=tokenizer,
            token_ids=top_indices,
            values=top_values,
            include_activation=False,
        )

    def _extract_slot_top_tokens(
        self,
        *,
        slot_logits: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        top_k: int,
    ) -> list[dict[str, Any]]:
        slot_payloads: list[dict[str, Any]] = []
        num_slots: int = int(slot_logits.shape[0])
        slot_idx: int
        for slot_idx in range(num_slots):
            flattened_logits: torch.Tensor = (
                slot_logits[slot_idx].detach().float().cpu().reshape(-1)
            )
            top_count: int = min(int(top_k), int(flattened_logits.numel()))
            top_values, top_indices = torch.topk(flattened_logits, k=top_count)
            slot_payloads.append(
                {
                    "slot": int(slot_idx),
                    "top_tokens": self._token_entries_from_topk(
                        tokenizer=tokenizer,
                        token_ids=top_indices,
                        values=top_values,
                        include_activation=True,
                    ),
                }
            )
        return slot_payloads

    def _token_entries_from_topk(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        token_ids: torch.Tensor,
        values: torch.Tensor,
        include_activation: bool,
    ) -> list[dict[str, Any]]:
        if int(token_ids.numel()) <= 0:
            return []
        token_id_list: list[int] = [int(token_id) for token_id in token_ids.tolist()]
        raw_tokens: list[str] = tokenizer.convert_ids_to_tokens(token_id_list)
        entries: list[dict[str, Any]] = []
        token_id: int
        token_str: str
        value: float
        for token_id, token_str, value in zip(
            token_id_list,
            raw_tokens,
            [float(raw_value) for raw_value in values.tolist()],
        ):
            entry: dict[str, Any] = {
                "token_id": int(token_id),
                "token": str(token_str),
                "decoded": str(
                    tokenizer.decode(
                        [int(token_id)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                ),
                "score": float(value),
            }
            if include_activation:
                entry["activated_score"] = float(torch.log1p(torch.relu(torch.tensor(value))).item())
            entries.append(entry)
        return entries

    def _render_markdown(self, payload: dict[str, Any]) -> str:
        lines: list[str] = [
            "# Validation Sparse Probe",
            "",
            f"- Global step: {int(payload['global_step'])}",
            f"- Epoch: {int(payload['epoch'])}",
            f"- Probe indices: {payload['probe_indices']}",
            "",
        ]
        sample: dict[str, Any]
        for sample in payload["samples"]:
            lines.extend(
                [
                    f"## Sample {int(sample['probe_rank']) + 1}",
                    "",
                    f"- Dataset index: {int(sample['dataset_index'])}",
                    f"- Query ID: `{sample['qid']}`",
                    f"- Document ID: `{sample['doc_id']}`",
                    "",
                    "### Query Text",
                    "",
                    "```text",
                    str(sample["query_text"]),
                    "```",
                    "",
                    "### Document Text",
                    "",
                    "```text",
                    str(sample["doc_text"]),
                    "```",
                    "",
                    "### Query Top Sparse Terms",
                    "",
                    "| Rank | Token | Decoded | Score |",
                    "| --- | --- | --- | ---: |",
                ]
            )
            lines.extend(
                self._render_token_rows(sample.get("query_top_sparse", []))
                or ["| - | - | - | - |"]
            )
            lines.extend(
                [
                    "",
                    "### Document Top Sparse Terms",
                    "",
                    "| Rank | Token | Decoded | Score |",
                    "| --- | --- | --- | ---: |",
                ]
            )
            lines.extend(
                self._render_token_rows(sample.get("doc_top_sparse", []))
                or ["| - | - | - | - |"]
            )
            if "query_slot_topk" in sample:
                lines.extend(
                    [
                        "",
                        "### Query Slot Predictions",
                        "",
                    ]
                )
                lines.extend(self._render_slot_rows(sample["query_slot_topk"]))
            if "doc_slot_topk" in sample:
                lines.extend(
                    [
                        "",
                        "### Document Slot Predictions",
                        "",
                    ]
                )
                lines.extend(self._render_slot_rows(sample["doc_slot_topk"]))
            lines.append("")
        return "\n".join(lines)

    def _render_token_rows(self, entries: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        rank: int
        entry: dict[str, Any]
        for rank, entry in enumerate(entries, start=1):
            decoded: str = str(entry.get("decoded", "")).replace("\n", " ")
            token: str = str(entry.get("token", "")).replace("|", "\\|")
            decoded = decoded.replace("|", "\\|")
            lines.append(
                f"| {rank} | `{token}` | `{decoded}` | {float(entry.get('score', 0.0)):.4f} |"
            )
        return lines

    def _render_slot_rows(self, slots: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        slot_payload: dict[str, Any]
        for slot_payload in slots:
            slot_idx: int = int(slot_payload.get("slot", 0))
            top_entries: list[dict[str, Any]] = list(slot_payload.get("top_tokens", []))
            summary: str = ", ".join(
                (
                    f"`{entry.get('token', '')}` "
                    f"({float(entry.get('score', 0.0)):.3f})"
                )
                for entry in top_entries
            )
            if not summary:
                summary = "-"
            lines.append(f"- Slot {slot_idx}: {summary}")
        if not lines:
            lines.append("- No slot predictions logged.")
        return lines

    def _log_artifacts_to_mlflow(self, artifact_paths: list[Path]) -> None:
        if not artifact_paths:
            return
        if not self._log_to_mlflow or self._mlflow_upload_disabled:
            return
        trainer: Any | None = getattr(self._module, "trainer", None)
        if trainer is None:
            return
        loggers: list[Any] = list(getattr(trainer, "loggers", []) or [])
        logger_instance: Any
        for logger_instance in loggers:
            run_id: str | None = getattr(logger_instance, "run_id", None)
            experiment: Any | None = getattr(logger_instance, "experiment", None)
            if run_id is None or experiment is None or not hasattr(
                experiment, "log_artifact"
            ):
                continue
            artifact_path: Path
            for artifact_path in artifact_paths:
                try:
                    with self._mlflow_fail_fast_request_env():
                        experiment.log_artifact(
                            run_id=str(run_id),
                            local_path=artifact_path.as_posix(),
                            artifact_path=self._artifact_dir_name,
                        )
                except Exception as exc:
                    log_if_rank_zero(
                        self._logger,
                        "Failed to log validation sparse probe artifact to MLflow: "
                        f"{artifact_path.as_posix()} ({exc})",
                        level="warning",
                    )
                    if self._disable_mlflow_after_failure:
                        self._mlflow_upload_disabled = True
                        log_if_rank_zero(
                            self._logger,
                            "Disabling further validation sparse probe MLflow "
                            "artifact uploads for this run after the first failure. "
                            "Local probe artifacts will still be written.",
                            level="warning",
                        )
                        return
            return

    @contextmanager
    def _mlflow_fail_fast_request_env(self) -> Any:
        overrides: dict[str, str] = {
            "MLFLOW_HTTP_REQUEST_TIMEOUT": str(self._mlflow_timeout_seconds),
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": str(self._mlflow_max_retries),
            "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": str(self._mlflow_backoff_factor),
        }
        original_values: dict[str, str | None] = {
            key: os.environ.get(key) for key in overrides
        }
        try:
            env_key: str
            env_value: str
            for env_key, env_value in overrides.items():
                os.environ[env_key] = env_value
            yield
        finally:
            env_key = ""
            original_value: str | None
            for env_key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = original_value
