# SPLADE Code Analysis

## Scope
- `src/data/dataset/msmarco.py`
- `src/data/dataset/retrieval.py`
- `src/model/retriever/base.py`
- `src/model/retriever/sparse/neural/splade_model.py`
- `src/model/module/train.py`
- `src/model/losses.py`

## Potential Correctness Issues
- **Teacher score loading is effectively disabled for separate datasets.** `MSMARCO._load_teacher_scores()` is never called, and `MSMARCO.setup()` raises when `hf_teacher_name` is set in streaming mode. This means configs that specify `hf_teacher_name` (e.g., `msmarco_hf_train`) cannot actually use external teacher scores, despite having those config fields. If you intend to support separate teacher datasets, wire `_load_teacher_scores()` into `setup()` (non-streaming) or explicitly document that only inline scores are supported in streaming mode.
- **KL distillation omits temperature scaling.** `distillation_loss()` uses raw logits in both student and teacher softmaxes. Standard distillation typically applies a temperature to soften distributions. If you expect temperature, add it to the config and apply in `distillation_loss()` to avoid mismatched gradients.
- **Edge case: all-negative batch entries.** `multi_positive_contrastive_loss()` uses `logsumexp` over positives without guarding against the “no positive” case. If a batch sample has no positives (possible with malformed data or aggressive negative-only sampling), the loss becomes `inf`. Consider adding a guard that skips or masks such rows.

## Optimization Opportunities
- **FAISS-free in-batch similarity.** `_compute_in_batch_scores()` materializes a full `bsz x (bsz * doc_count)` matrix and a full boolean mask of the same size. This can be memory-heavy with large batches. Consider chunking over doc blocks or using a lower precision buffer to reduce peak memory.
- **Repeated text lookups in training.** `_row_to_item()` repeatedly converts IDs to text via `query_dataset` and `corpus_dataset` lookups per sample. For large-scale training, caching or prefetching hot queries/docs (or relying on inline text) could reduce dataset access overhead.
- **Doc encoding flatten/reshape cost.** `_training_step_shared()` flattens `doc_input_ids` and `doc_attention_mask` every step. This is correct but can be optimized by avoiding extra `view` calls when the shape already matches or by using a single contiguous view and reusing it in regularization.

## Refactoring Opportunities
- **Split `MSMARCO` into focused helpers.** The class handles dataset loading, id mapping, score loading, integer-id preprocessing, and row parsing in one file. Consider extracting helpers for (a) dataset loading, (b) ID mapping and caching, and (c) row parsing to simplify testing and reduce cognitive load.
- **Centralize column resolution logic.** Both `MSMARCO` and `RetrievalDataset` implement column resolution. A shared utility would reduce duplication and keep column resolution rules consistent across datasets.
- **Clarify streaming vs map-style behavior.** `MSMARCO.setup()` always streams unless integer-id caching is used. This is efficient, but it interacts poorly with external teacher datasets and some mapping operations. A small refactor to make streaming a configurable choice (e.g., `hf_streaming`) would help align behavior with config.
