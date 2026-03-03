# SPLADE Refactor TODO

## Goals
- Improve modularity and readability with minimal behavior changes first.
- Reduce training/evaluation complexity by isolating responsibilities.
- Keep performance equal or better after each step, with measurable benchmarks.

## Working Rules
- Refactor in small, reversible slices.
- Add tests for each extracted module before broad rewires.
- Do not change training semantics unless explicitly listed.
- Track benchmark deltas after each optimization step.

## Phase 0: Baseline + Safety
- [ ] Capture baseline metrics:
  - [ ] Train throughput (`it/s`, tokens/s) for current primary configs.
  - [ ] Validation runtime (MSMARCO rerank + NanoBEIR).
  - [ ] Retrieval eval runtime (index-based).
- [x] Freeze baseline command matrix in docs.
- [x] Add a single smoke script for `train -> encode -> index -> evaluation`.

## Phase 1: Low-Risk Modularization (No behavior changes)
- [x] **Step 1**: Extract shared tokenizer/dataloader helper utilities for Lightning data modules.
- [x] **Step 2**: Extract shared script bootstrap utility (`init/run/trainer logger wiring`) used by `train/encode/evaluation/validation`.
- [x] **Step 3**: Consolidate duplicate normalization helpers (`optional str/path/bool`) into one utility module.
- [x] **Step 4**: Remove wrapper-only scripts and replace with direct entrypoints where possible.
- [x] **Step 5**: Standardize module-local logger construction and rank-zero logging policy.

## Phase 2: Core Training Module Decomposition
- [x] Split `src/model/pl_module/train.py` into:
  - [x] Compile policy manager.
  - [x] Loss/regularization service.
  - [x] Metrics/logging service.
  - [x] Validation candidate/metric accumulator.
  - [x] NanoBEIR evaluation runner.
- [x] Introduce clear interfaces between train-step compute and logging.
- [x] Keep checkpoint compatibility via adapter layer.

## Phase 3: Data Pipeline Decomposition
- [x] Split `src/data/pd_module/train.py` into:
  - [x] Runtime row materialization path.
  - [x] Shared tokenization/rerank-input utilities.
- [x] Remove pretokenization cache codepaths and keep a single runtime-tokenization flow.
- [x] Keep dataset/text/id artifact warmup explicit through `PDModule` loading policy hooks.

## Phase 4: Evaluation/Validation Unification
- [x] Ensure standalone `script/validation.py` uses exactly training-time validation semantics.
- [x] Keep retrieval evaluation (`script/evaluation.py`) isolated and explicit.
- [x] Share metric serialization and formatting helpers across both paths.
- [x] Add consistency checks for `val_loss`, `val_MRR_k`, `val_nDCG_k`, `val_Recall_k`.

## Phase 5: Performance Optimization
- [x] Reduce Python overhead in `__getitem__` hot paths.
- [x] Tighten scoring backend buffer reuse and minimize allocations.
- [x] Gate expensive per-step logging metrics by interval.
- [x] Add benchmark harness for:
  - [x] training throughput baseline
  - [x] validation runtime baseline
  - [x] retrieval runtime baseline
- [x] Record and compare ETA/throughput for active runtime paths.

## Phase 6: Config + CI + Test Hygiene
- [x] Remove ambiguous config artifacts (example duplicate names).
- [x] Add Hydra config composition tests for key model/training/dataset matrices.
- [x] Expand tests for DDP + `torch_compile` combinations (frozen/unfrozen).
- [x] Add lint + type-check pipeline (`ruff` + `mypy`).
- [x] Fix CI test discovery/coverage consistency for `test/` layout.

## Phase 7: Documentation and Migration
- [x] Document new module boundaries and architecture.
- [x] Add migration notes for renamed/relocated modules.
- [x] Provide command cookbook for train/eval/validation with common overrides.
- [x] Add "debugging runbook" for cache, DDP, compile, and MLflow issues.

## Immediate Execution Queue
- [x] Create this TODO plan.
- [x] Execute Phase 1 Step 1 now.
- [x] Run static checks after Step 1 and fix regressions.
- [x] Proceed to Phase 1 Step 2.
- [x] Proceed to Phase 1 Step 3.
- [x] Proceed to Phase 1 Step 4.
- [x] Proceed to Phase 1 Step 5.
- [x] Proceed to Phase 2 Step 1 (extract compile policy manager from training module).
- [x] Proceed to Phase 2 Step 2 (extract loss/regularization service from training module).
- [x] Proceed to Phase 2 Step 3 (extract metrics/logging service from training module).
- [x] Proceed to Phase 2 Step 4 (extract validation candidate/metric accumulator).
- [x] Proceed to Phase 2 Step 5 (extract NanoBEIR evaluation runner).
- [x] Proceed to Phase 3 Step 1 (simplify training data module flow).
- [x] Proceed to Phase 3 Step 2 (consolidate runtime row materialization path).
- [x] Proceed to Phase 3 Step 3 (remove legacy cache module surfaces).
- [x] Proceed to Phase 3 Step 4 (finalize direct runtime tokenization path).
- [x] Proceed to Phase 4 Step 1 (validation semantics parity check and consolidation).
- [x] Proceed to Phase 4 Step 2 (retrieval evaluation isolation/explicitness hardening).
- [x] Proceed to Phase 6 Step 2 (Hydra config composition tests).
- [x] Proceed to Phase 6 Step 3 (DDP + torch_compile matrix tests).
