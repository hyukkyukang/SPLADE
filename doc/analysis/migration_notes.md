# Migration Notes

This note summarizes behavior-preserving changes that affect extension points.

## Training Module Extraction

- `src/model/pl_module/train.py` now delegates core concerns:
  - compile policy -> `compile_policy.py`
  - loss/regularization -> `loss_service.py`
  - metrics logging -> `metrics_service.py`
  - validation accumulation -> `validation_service.py`
  - NanoBEIR execution -> `nanobeir_runner.py`

If you previously modified `train.py` directly for these concerns, migrate edits
to the dedicated service module first.

## Pretokenize Pipeline Extraction

- Pretokenization logic moved out of `src/data/pd_module/train.py` into:
  - `pretokenize_lifecycle.py`
  - `pretokenize_writer.py`
  - `pretokenize_runtime_reader.py`
  - `pretokenize_row_materializer.py`
  - `pretokenize_backend.py`

Behavior is intentionally preserved, but the entry points are now smaller and
module-specific.

## Validation/Evaluation Semantics

- Standalone validation (`script/validation.py`) is aligned with training-time
  validation semantics.
- Validation candidates are per-query and fixed-size (100 negatives style) for
  consistent comparison.
- Retrieval evaluation remains isolated in `script/evaluation.py` with explicit
  retrieval-mode enforcement.

## Checkpoint Compatibility

- State-dict key alias handling is centralized in:
  - `src/utils/checkpoint_compat.py`

If new prefix/key migrations are needed, add aliases there rather than ad-hoc
in individual scripts.

## Config Naming Cleanup

- Removed ambiguous model artifact:
  - `config/model/splade_v2 copy.yaml`
- Encoding config `name` fields are unique:
  - `config/encoding/anna_base.yaml` -> `name: anna_base`
  - `config/encoding/bert_large.yaml` -> `name: bert_large`

