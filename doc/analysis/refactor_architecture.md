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
  - High-level training data module composition.
  - Direct runtime tokenization-based `__getitem__` path.
- `utils.py`
  - Shared tokenization helpers and rerank input materialization.
- `base.py`
  - Dataset loading policy and required text/id artifact warmup hooks.

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
