# EmbeddingGemma-LSR Build Pipeline

This document describes the end-to-end pipeline for building the lexical target
vocabulary and model artifacts used by the EmbeddingGemma-based SPLADE/LSR
experiments in this repository.

It covers:

1. The vocabulary build pipeline.
2. The native SPLADE backbone build path.
3. The prototype LSR model-init path.
4. Fast rerun workflows using cached statistics.
5. The refactored entrypoints and reusable helper modules.

## Pipeline Summary

There are two downstream model-building paths that share the same vocabulary
build stage.

1. Vocabulary build:
   `script/preprocess/build_embeddinggemma_lsr_vocab.py`
2. Native SPLADE backbone build:
   `script/model_creation/embeddinggemma_splade/build_hf_backbone.py`
3. Prototype projection-head init:
   `script/model/init_embeddinggemma_lsr.py`
4. Training:
   `script/train.py` for the native SPLADE path, or
   `script/train_embeddinggemma_lsr.py` for the prototype path.

The new orchestration entrypoint for the native model-creation path is:

`script/model_creation/embeddinggemma_splade/build_pipeline.py`

The new shared helper modules are:

1. `src/prototype/embeddinggemma_lsr/cli.py`
2. `src/prototype/embeddinggemma_lsr/artifacts.py`
3. `src/prototype/embeddinggemma_lsr/vocab_filtering.py`
4. `src/prototype/embeddinggemma_lsr/vocab_audit.py`

These remove duplicated config-override, device/dtype, and artifact-loading
logic from the individual scripts, and move pure vocabulary filtering/audit
logic out of the giant builder script.

## Current Default Behavior

The default native vocabulary config is:

`config/model_creation/embeddinggemma_splade/vocab.yaml`

Important defaults in that config:

1. `use_all_corpus_documents: true`
2. `max_docs: null`
3. `map_reduce_sharding: true`
4. `map_reduce_num_shards: 48`
5. `map_reduce_num_workers: 48`
6. `save_term_stats_cache: true`
7. `term_stats_cache_path: outputs/model_creation/embeddinggemma_splade/vocab/term_statistics.pkl`
8. Strict post-selection cleanup is enabled.

Operationally, this means the default pipeline is:

1. Scan the full corpus.
2. Build and save corpus-wide term statistics once.
3. Select the final vocabulary from those saved statistics.
4. Build a Hugging Face EmbeddingGemma backbone aligned to that vocabulary.

## Vocabulary Build Stages

The vocabulary builder is large because it performs both corpus statistics
collection and multiple ranking/filtering passes. The process is:

1. Resolve the text corpus from the configured Hugging Face dataset.
2. Extract token terms, noun chunks, and named entities.
3. Aggregate DF/TF and source-specific DF counters.
4. Save `term_statistics.pkl` when caching is enabled.
5. Apply candidate filtering:
   stopwords, noise, function-led phrases, canonicalization, noun-form
   normalization, artifact filtering, generic unigram filtering, entity quality,
   phrase cohesion, POS gate, numeric quality, and strict post-selection cleanup.
6. Rank candidate terms by BM25-style utility with source-proportion-weighted
   boosts.
7. Save the final vocabulary artifacts.

## Vocabulary Artifacts

Default output directory:

`outputs/model_creation/embeddinggemma_splade/vocab`

Files:

1. `v_target.txt`
   Final selected vocabulary list, one term per line.
2. `df_map.json`
   Final selected term to DF map.
3. `vocab_stats.json`
   Full selection summary plus per-term ranked metadata.
4. `manifest.json`
   Run arguments, source stats, spaCy stats, and summary.
5. `term_statistics.pkl`
   Cached corpus-wide DF/TF/source statistics used for selection-only reruns.

The cache file is the expensive part to recreate. It should be treated as the
checkpoint for the vocabulary build process.

## Pipeline Run Logs

The orchestration script writes a per-run log directory by default:

`outputs/model_creation/embeddinggemma_splade/pipeline_runs`

Each run directory contains:

1. `manifest.json`
2. `events.jsonl`

These files capture:

1. Resolved arguments and important paths.
2. Stage status transitions.
3. Commands executed.
4. Pipeline start/end timestamps.

## Native SPLADE Model-Creation Path

This path creates a standard Hugging Face CausalLM-style backbone that can be
trained with the repository’s native SPLADE training stack.

### One-command path

```bash
python -u script/model_creation/embeddinggemma_splade/build_pipeline.py \
  --config config/model_creation/embeddinggemma_splade/pipeline.yaml
```

This runs:

1. `build_target_vocab.py`
2. `build_hf_backbone.py`

### Stage-by-stage path

Build the target vocabulary:

```bash
python -u script/model_creation/embeddinggemma_splade/build_target_vocab.py \
  --config config/model_creation/embeddinggemma_splade/vocab.yaml
```

Build the HF backbone:

```bash
python -u script/model_creation/embeddinggemma_splade/build_hf_backbone.py \
  --config config/model_creation/embeddinggemma_splade/hf_backbone.yaml
```

Default backbone output directory:

`outputs/model_creation/embeddinggemma_splade/hf_backbone`

Backbone artifacts include:

1. Standard Hugging Face model files.
2. `target_vocab.txt`
3. `df_map.json`
4. `term_to_token_id.json`
5. `unresolved_terms.json`
6. `init_summary.json`
7. `splade_compact_head.pt`

### Native training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u script/train.py \
  --config-name train_embeddinggemma_splade_v2_pp \
  training.num_devices=4
```

The corresponding train entry config is:

`config/train_embeddinggemma_splade_v2_pp.yaml`

The model config points at the HF backbone produced by the model-creation
pipeline.

## Prototype LSR Path

This path uses the same vocabulary artifacts, but initializes the custom LSR
projection model instead of a native SPLADE backbone.

Build vocabulary:

```bash
python -u script/preprocess/build_embeddinggemma_lsr_vocab.py \
  --config config/prototype/embeddinggemma_lsr_vocab.yaml
```

Initialize the prototype model:

```bash
python -u script/model/init_embeddinggemma_lsr.py \
  --config config/prototype/embeddinggemma_lsr_init.yaml
```

Train the prototype:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u script/train_embeddinggemma_lsr.py \
  --config config/prototype/embeddinggemma_lsr_train.yaml
```

Prototype init output directory by default:

`outputs/embeddinggemma_lsr/model_init`

Prototype training output directory by default:

`outputs/embeddinggemma_lsr/train_run`

## Fast Rerun Workflows

### Re-run only vocabulary selection from cached corpus statistics

Use this when you are changing only selection/filtering logic and do not want to
re-extract terms from the full corpus.

```bash
python -u script/model_creation/embeddinggemma_splade/build_target_vocab.py \
  --config config/model_creation/embeddinggemma_splade/vocab.yaml \
  --selection-only
```

This requires:

`outputs/model_creation/embeddinggemma_splade/vocab/term_statistics.pkl`

### Run pipeline with cached statistics already present

```bash
python -u script/model_creation/embeddinggemma_splade/build_pipeline.py \
  --config config/model_creation/embeddinggemma_splade/pipeline.yaml \
  --selection-only
```

### Skip vocab stage and rebuild only the HF backbone

```bash
python -u script/model_creation/embeddinggemma_splade/build_pipeline.py \
  --config config/model_creation/embeddinggemma_splade/pipeline.yaml \
  --no-run-vocab
```

### Print the resolved pipeline commands without running them

```bash
python -u script/model_creation/embeddinggemma_splade/build_pipeline.py \
  --config config/model_creation/embeddinggemma_splade/pipeline.yaml \
  --print-commands-only
```

## Vocabulary Quality Audit

After a vocab build or selection-only rerun, audit the resulting vocabulary with:

```bash
python -u script/experiment/audit_embeddinggemma_vocab.py \
  --vocab-artifact-dir outputs/model_creation/embeddinggemma_splade/vocab
```

This reads `vocab_stats.json` and reports:

1. unigram/phrase balance
2. POS mix
3. short-alpha residuals
4. abbreviation-heavy phrase residuals
5. trailing-function phrase residuals
6. numeric boilerplate phrase residuals
7. probe membership for known bad patterns

## CPU And GPU Expectations

The vocabulary build is primarily CPU-bound.

Recommendations:

1. Use many CPU workers for the map-reduce corpus pass.
2. Expect `term_statistics.pkl` generation to dominate total runtime.
3. Use `--selection-only` for filter/ranking iteration.

The backbone build and prototype init can use GPU if desired:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u script/model_creation/embeddinggemma_splade/build_hf_backbone.py \
  --config config/model_creation/embeddinggemma_splade/hf_backbone.yaml
```

If `device: auto`, the script will use CUDA when available.

## Process Notes And Optimization Guidance

The main process optimizations already in place are:

1. Full-corpus map-reduce sharding for vocabulary statistics collection.
2. Persistent `term_statistics.pkl` cache.
3. Selection-only reruns for quality iteration.
4. Strict post-selection cleanup to remove residual noisy terms without
   recomputing term statistics.

The recommended operating pattern is:

1. Run the full corpus statistics build once.
2. Keep `term_statistics.pkl`.
3. Iterate on filters, scoring, and cleanup with `--selection-only`.
4. Rebuild the HF backbone only after the vocabulary is stable.

## Troubleshooting

### `selection-only` fails because cache is missing

You do not have `term_statistics.pkl` yet, or it was removed. Re-run the full
vocabulary build once.

### The vocabulary build is slow even on a large server

This is expected during the map stage. The dominant cost is corpus parsing and
term extraction. Selection-only reruns are the intended fast path.

### The backbone build fails on unresolved terms

Check:

1. `unresolved_terms.json`
2. tokenizer-added token behavior
3. `fragment_threshold`
4. `allow_unresolved_terms`

### You changed filtering logic but not the corpus

Use the cached rerun path instead of recomputing corpus statistics.

## Refactored File Map

Core entrypoints:

1. `script/preprocess/build_embeddinggemma_lsr_vocab.py`
2. `script/model_creation/embeddinggemma_splade/build_pipeline.py`
3. `script/model_creation/embeddinggemma_splade/build_hf_backbone.py`
4. `script/model/init_embeddinggemma_lsr.py`
5. `script/train_embeddinggemma_lsr.py`

Shared helpers:

1. `src/prototype/embeddinggemma_lsr/cli.py`
2. `src/prototype/embeddinggemma_lsr/artifacts.py`

Key configs:

1. `config/model_creation/embeddinggemma_splade/vocab.yaml`
2. `config/model_creation/embeddinggemma_splade/hf_backbone.yaml`
3. `config/model_creation/embeddinggemma_splade/pipeline.yaml`
4. `config/prototype/embeddinggemma_lsr_vocab.yaml`
5. `config/prototype/embeddinggemma_lsr_init.yaml`
6. `config/prototype/embeddinggemma_lsr_train.yaml`
7. `config/train_embeddinggemma_splade_v2_pp.yaml`
