# Refactor Architecture

This document captures the current module boundaries after the SPLADE refactor.

## Training Stack (`src/model/pl_module`)

- `train.py`
  - Orchestrates training/validation lifecycle only.
  - Delegates compile, loss, metrics, and NanoBEIR behavior to dedicated services.
- `compile_policy.py`
  - Owns `torch.compile` policy and DDP-safe fallback behavior.
  - Handles full-model vs wrapper-only compile selection.
- `loss_service.py`
  - Builds loss computer and regularization wiring.
- `metrics_service.py`
  - Applies logging policy for training metrics.
  - Supports interval gating for expensive step-only metrics.
- `validation_service.py`
  - Computes per-query reranking metrics (`MRR/nDCG/Recall`) from validation batches.
- `nanobeir_runner.py`
  - Isolates NanoBEIR execution and runtime cache management.

## Data Pipeline (`src/data/pd_module`)

- `train.py`
  - High-level data module composition.
- `pretokenize_lifecycle.py`
  - Cache manifest validation, lock/done lifecycle, prepare/load decisions.
- `pretokenize_writer.py`
  - Cache shard writing and row-index generation.
- `pretokenize_runtime_reader.py`
  - Worker-local streaming cache setup and meta-row-pointer reads.
- `pretokenize_row_materializer.py`
  - `__getitem__` materialization to `TrainingDataItem`.
- `pretokenize_backend.py`
  - Backend protocol/FS implementation for pretokenize artifacts.
- `token_store.py`
  - Streaming token store over row indexes + sidecar/parquet shards.

## Evaluation And Validation

- `script/validation.py`
  - Uses training-style validation semantics (per-query candidate reranking).
  - Shares metric IO helpers via `src/utils/metrics_io.py`.
- `script/evaluation.py`
  - Enforces retrieval-only path using index-based evaluation.
  - Uses `src/utils/evaluation_mode.py` guardrail.

## Shared Utility Layers

- `src/utils/checkpoint_compat.py`
  - Central state-dict aliasing for checkpoint key compatibility.
- `src/utils/metrics_io.py`
  - Shared metric serialization and validation-result formatting.

