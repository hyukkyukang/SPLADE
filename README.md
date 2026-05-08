# SPLADE / LENS

Training + evaluation repo for sparse retrieval models, including:

- SPLADE v1/v2/v3
- SPLADE++-style variants with EmbeddingGemma backbones
- LENS with bidirectional Mistral, clustered compact heads, and LoRA/PEFT

The repo supports Hydra-driven training, sparse corpus encoding + inverted
indexing, retrieval/reranking evaluation, NanoBEIR/MTEB proxy benchmarks, and
model-creation utilities.

## Setup

```
pip install -r requirements.txt
```

## Configuration

All entrypoints use Hydra configs under `config/`.

- Override config groups: `model=splade_v2_doc`, `dataset=beir/trec-covid`,
  `training=splade_v2`.
- Override parameters: `training.use_cpu=true`,
  `training.resume_checkpoint_path=...`, `testing.checkpoint_path=...`,
  `encoding.checkpoint_path=...`.
- Logs go to `log/.../no_tag` by default; set `tag=...` to replace `no_tag`.

Datasets default to the Hugging Face Hub; dataset configs live in
`config/dataset/`.

Common entrypoints:

- `script/train.py`: training
- `script/encode.py`: sparse corpus encoding
- `script/index.py`: inverted-index build
- `script/evaluate.py`: retrieval, reranking, and benchmark dispatch
- `script/model_creation/lens/*.py`: LENS artifact build pipeline

## Refactor docs

- Architecture: `doc/analysis/refactor_architecture.md`
- Migration notes: `doc/analysis/migration_notes.md`
- Command cookbook: `doc/analysis/command_cookbook.md`
- Debug runbook: `doc/analysis/debug_runbook.md`
- Baseline command matrix: `doc/analysis/baseline_command_matrix.md`
- EmbeddingGemma build pipeline: `doc/embeddinggemma_lsr_build_pipeline.md`

## LENS model creation

Build the default 4k-cluster LENS artifacts:

```
python script/model_creation/lens/build_pipeline.py \
  --config config/model_creation/lens/pipeline_4k.yaml
```

Build the 8k-cluster variant:

```
python script/model_creation/lens/build_pipeline.py \
  --config config/model_creation/lens/pipeline_8k.yaml
```

The pipeline prepares a self-contained Hugging Face backbone with LENS special
tokens (`<instruct>`, `<query>`, `<response>`) and then writes a clustered
`splade_compact_head.pt`. The default outputs are:

- `outputs/model_creation/lens/hf_backbone`
- `outputs/model_creation/lens/mistral_cluster4k`
- `outputs/model_creation/lens/mistral_cluster8k`

The bundled LENS training configs expect the clustered artifact directories
above unless you override `model.huggingface_name`.

## Train

Train a SPLADE model and write checkpoints/logs:

```
python script/train.py training=splade_v1 model=splade_v1
```

Train the default LENS setup:

```
python script/train.py --config-name train_lens_mistral
```

`train_lens_mistral` uses:

- `model=lens_mistral_cluster4k`
- `training=lens_mistral`
- MS MARCO training rows
- MSMARCO dev-small-negatives validation
- NanoBEIR validation during training

Switch to the 8k clustered head with:

```
python script/train.py --config-name train_lens_mistral \
  model=lens_mistral_cluster8k
```

The default LENS presets use bidirectional Mistral, max-pooling over activated
LM logits, and LoRA/PEFT adapters.

Use MS MARCO (HF Hub) for both train/val:

```
python script/train.py \
  dataset@train_dataset=msmarco \
  dataset@val_dataset=msmarco
```

Train with triplet meta rows + scored hard negatives (margin-MSE distillation):

```
python script/train.py \
  training=splade_v2_distill \
  model=splade_v2_hf \
  dataset@train_dataset=msmarco_triplet_scores \
  dataset@val_dataset=msmarco
```

Train with MSMARCO dev hard negatives for reranking-style validation
(`antoinelouis/msmarco-dev-small-negatives`) while keeping NanoBEIR enabled:

```
python script/train.py \
  --config-name train_devneg
```

Train SPLADE v2++ with `google/embeddinggemma-300m` as backbone
(requires gated-model access and `HF_TOKEN` in `.env`):

```
python script/train.py \
  model=splade_v2_pp_embeddinggemma_300m \
  training=splade_v2_pp
```

Build an English-focused pruned EmbeddingGemma backbone
(drops non-English vocabulary tokens and image-special tokens):

```
python script/preprocess/prune_embeddinggemma_vocab.py \
  --input-model google/embeddinggemma-300m \
  --output-dir data/model/trained_embeddinggemma_300m_pruned \
  --overwrite
```

Train SPLADE v2++ with the pruned EmbeddingGemma backbone:

```
python script/train.py \
  model=splade_v2_pp_embeddinggemma_300m_pruned \
  training=splade_v2_pp
```

Train directly from scored distillation rows (no triplet join):

```
python script/train.py \
  training=splade_v2_distill \
  dataset@train_dataset=msmarco_distill_scores \
  dataset@val_dataset=msmarco
```

Train the fixed-vocabulary pretrained diffusion-backbone ablation with UDLM
(`kuleshov-group/udlm-lm1b`):

```
python script/train.py --config-name train_pretrained_diffusion_splade
```

Train the paired UDLM + MDLM-aux arm:

```
python script/train.py --config-name train_pretrained_diffusion_mdlm_splade
```

These UDLM presets pin query/document length to 128 to match the upstream
checkpoint context length. The Hugging Face checkpoint also uses
`trust_remote_code` and currently imports `einops` and `flash_attn`, so those
packages must be available in the runtime environment.

Train directly from a multi-teacher score dataset
(`teacher_scores` + per-model columns):

```
python script/train.py \
  training=splade_v2_distill \
  dataset@train_dataset=msmarco_multi_teacher_scores \
  dataset@val_dataset=msmarco
```

Override the scored dataset or number of negatives per query:

```
python script/train.py \
  dataset@train_dataset=msmarco_triplet_scores \
  train_dataset.score_hf_name=Hyukkyu/msmarco-spladev2-hard-negatives-scores \
  train_dataset.score_negatives_per_query=1
```

Disable MLflow logging for quick debug runs (optional):

```
python script/train.py tag=debug
```

Train with explicit MLflow tracking settings:

```
python script/train.py \
  training.mlflow.tracking_uri=https://mlflow.hyukkyu.com
```

MLflow GPU system metrics use NVML via `nvidia-ml-py`. If GPU utilization or
memory metrics are missing, refresh the environment with `pip install -r requirements.txt`.

Resume an interrupted training run from a Lightning checkpoint:

```
python script/train.py \
  training.resume_checkpoint_path=log/train/splade_v2/no_tag/checkpoints/last.ckpt
```

Checkpoint options for training:

- `training.init_checkpoint_path`: initialize model weights only (fresh optimizer,
  scheduler, and global step).
- `training.resume_checkpoint_path`: resume full Lightning state (model + optimizer
  + scheduler + step/epoch counters).
- Do not set both at the same time.

## Encode corpus

Encode documents into sparse shards (for retrieval indexing):

```
python script/encode.py \
  encoding.checkpoint_path=log/train/splade_v2/no_tag/checkpoints/last.ckpt \
  dataset=beir/trec-covid \
  encoding.encode_dir=log/encode
```

For clustered LENS artifacts, encoded dimensions are latent cluster ids rather
than tokenizer ids. Encode and index manifests record that output-space
alignment automatically.

## Build inverted index

Build a sparse inverted index from encoded shards:

```
python script/index.py \
  encoding.encode_dir=log/encode \
  encoding.index_dir=log/index
```

Index scoring supports `testing.scoring_method=full|wand|bmw`. Block-Max WAND
(`bmw`) requires bounds stored in the index. If you enable `bmw` or change
`encoding.wand_block_size`, rebuild the index with `script/index.py`.

## Evaluate (retrieval / reranking)

Index-based retrieval evaluation:

```
python script/evaluate.py \
  evaluation.type=retrieval \
  testing.checkpoint_path=log/train/splade_v2/no_tag/checkpoints/last.ckpt \
  dataset=beir/trec-covid \
  encoding.index_dir=log/index
```

Reranking evaluation (no index required):

```
python script/evaluate.py \
  evaluation.type=reranking \
  testing.checkpoint_path=log/train/splade_v2/no_tag/checkpoints/last.ckpt \
  dataset=beir/msmarco
```

## Evaluate (NanoBEIR / MTEB proxy)

Quick proxy evaluation without full-corpus encoding:

```
python script/evaluate.py --benchmark nanobeir \
  testing.checkpoint_path=log/train/splade_v2/no_tag/checkpoints/last.ckpt \
  nanobeir.datasets='[msmarco, nfcorpus, nq]' \
  nanobeir.save_json=true
```

Run the shared sparse benchmark path through the MTEB entrypoint:

```
python script/evaluate.py --benchmark mteb \
  --config-name evaluate_mteb \
  testing.checkpoint_path=log/train/splade_v2/no_tag/checkpoints/last.ckpt
```

Use HF weights instead of a checkpoint:

```
python script/evaluate.py --benchmark nanobeir \
  nanobeir.use_huggingface_model=true \
  model.huggingface_name=naver/splade_v2
```

Evaluate a LENS checkpoint on the native sparse benchmark adapter:

```
python script/evaluate.py --benchmark nanobeir \
  --config-name evaluate_nanobeir \
  model=lens_mistral_cluster4k \
  testing.checkpoint_path=log/train/lens_mistral_cluster4k/no_tag/checkpoints/last.ckpt
```

Benchmark routing is backend-aware:

- vanilla SPLADE can use the SentenceTransformers MLM sparse adapter
- LENS, PEFT-wrapped models, and non-MLM backbones use the native sparse adapter

## Patent document evaluation

This repo also supports patent document retrieval evaluation with:

- label dataset: `Hyukkyu/patent-us-small`
- corpus dataset: `Hyukkyu/patent-us-corpus-small`

The label dataset stores document-to-document relevance pairs:

- `question_id`: patent document id used as the query document id
- `label_id`: list of relevant patent document ids

Queries are reconstructed by looking up each `question_id` in the patent corpus
and rendering the same document text template used for corpus encoding:

```text
Title: {title}
Abstract: {abstract}
Claims: {claims}
Description: {description}
```

The checked-in `patent_us_corpus_small` and `patent_us_small_eval` presets
expect the small patent corpus parquet shards under
`.cache/hf/patent-us-corpus-small/data/*.parquet`. The artifact builder and
dataset loader both use `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` from `.env` when
accessing Hugging Face datasets.

Build local `queries.parquet` and `qrels.parquet` artifacts for the `test` split:

```
python script/preprocess/patent/build_patent_us_eval_artifacts.py \
  --benchmark-repo Hyukkyu/patent-us-small \
  --benchmark-split test \
  --corpus-repo Hyukkyu/patent-us-corpus-small \
  --corpus-split train \
  --output-dir data/eval/patent_us_small
```

Two long-document encoding modes are supported for learned sparse models such as
SPLADE:

- `encoding.long_doc_strategy=truncate`: encode only the first `max_doc_length`
  tokens from the templated patent text
- `encoding.long_doc_strategy=sliding_window`: encode the full templated patent
  text in non-overlapping windows and aggregate the windows

Use the same mode on both the corpus side and the query side.

Encode the patent corpus with `naver/splade-v3` in `truncate` mode:

```
python script/encode.py \
  model=splade_v3_naver \
  dataset=patent_us_corpus_small \
  tag=patent_us_small_splade_v3_truncate \
  encoding.batch_size=160 \
  encoding.max_windows_per_forward=160
```

Encode the same corpus in `sliding_window` mode:

```
python script/encode.py \
  model=splade_v3_naver \
  dataset=patent_us_corpus_small \
  tag=patent_us_small_splade_v3_sliding \
  encoding.batch_size=160 \
  encoding.long_doc_strategy=sliding_window \
  encoding.max_windows_per_forward=160
```

The current encode defaults already include the optimized settings used for the
patent corpus benchmarks:

- `encoding.prefetch_factor=4`
- `encoding.torch_compile=true`
- `encoding.torch_compile_mode=default`
- `encoding.torch_compile_ddp_safe_mode=false`
- `encoding.distributed_sampler_strategy=row_group_interleaved`

Build the inverted index from the encoded shards:

```
python script/index.py \
  model=splade_v3_naver \
  dataset=patent_us_corpus_small \
  tag=patent_us_small_splade_v3_truncate
```

Evaluate on the `test` split of `Hyukkyu/patent-us-small`:

```
python script/evaluation.py \
  model=splade_v3_naver \
  dataset=patent_us_small_eval \
  testing=patent_us_small_eval \
  tag=patent_us_small_splade_v3_truncate \
  dataset.query_long_doc_strategy=truncate
```

For `sliding_window`, switch both sides together:

```
python script/evaluation.py \
  model=splade_v3_naver \
  dataset=patent_us_small_eval \
  testing=patent_us_small_eval \
  tag=patent_us_small_splade_v3_sliding \
  dataset.query_long_doc_strategy=sliding_window
```

The `testing=patent_us_small_eval` preset reports only retrieval metrics for:

- `MRR@1`
- `MRR@5`
- `MRR@10`
- `MRR@16`
- `MRR@32`
- `MRR@63`
- `MRR@150`
- `MRR@300`
- `Recall@1`
- `Recall@5`
- `Recall@10`
- `Recall@16`
- `Recall@32`
- `Recall@63`
- `Recall@150`
- `Recall@300`

It also enables `testing.exclude_self_match=true`, which is important because
each query is itself a patent document from the corpus.

If you want to evaluate a trained checkpoint instead of raw Hugging Face model
weights, pass the same checkpoint to both steps:

- `encoding.checkpoint_path=/path/to/checkpoint.ckpt`
- `testing.checkpoint_path=/path/to/checkpoint.ckpt`

## Preprocess

Mine hard negatives with a trained checkpoint:

```
python script/preprocess/mine_hard_negative.py \
  mining.checkpoint_path=log/train/splade_v2/no_tag/checkpoints/last.ckpt \
  mining.output_dir=data/hard_negatives \
  mining.output_format=triplet
```

Score candidate pairs with a cross-encoder:

```
python script/preprocess/mine_distillation_score.py \
  dataset@score_dataset=msmarco \
  scoring.model_name=cross-encoder/ms-marco-MiniLM-L-12-v2
```

Score MS MARCO hard negatives (merge all neg keys):

```
python script/preprocess/mine_distillation_score.py \
  --config-name score_cross_encoder_msmarco_hard_negatives \
  scoring.max_rows=10
```

## ANNA Conversion And Tokenizer Backend

The source-of-truth ANNA implementation now lives under:

```
src/tokenization/anna_tokenizer.py
src/preprocess/anna_conversion_utils.py
```

The files under `script/preprocess/anna/` remain compatibility wrappers for the
older script layout.

Convert ANNA checkpoints to Hugging Face artifacts with:

```
python script/preprocess/anna/convert_to_hf.py \
  --input-dir data/model/anna \
  --output-dir data/model/anna_large_hf \
  --overwrite
```

Backwards-compatible wrapper (same behavior):

```
python script/preprocess/convert_anna_to_hf.py --help
```

Build the Rust-backed ANNA tokenizer extension (Linux):

```
cd script/preprocess/anna/native/anna_fast_rs
maturin build --release -i python
pip install --force-reinstall target/wheels/*.whl
```

Quick throughput benchmark (slow vs fast):

```
python script/experiment/benchmark_anna_tokenizer_speed.py \
  --model-dir data/model/anna_large_hf \
  --local-files-only
```

Runtime code loads tokenizer/model through Hugging Face standard APIs
(`AutoTokenizer.from_pretrained`, `AutoModelForMaskedLM.from_pretrained`).

## LENS CPU smoke test

Run the end-to-end CPU smoke path before the first real LENS training run:

```
python script/etc/lens_cpu_smoke.py \
  --output-dir outputs/smoke/lens_cpu_smoke \
  --cluster-count 8
```

This builds a tiny local Mistral artifact, runs the LENS backbone + clustered
head pipeline, checks query/doc encoding, and executes a one-step
train/validation loop on CPU.

## SPLADE-v3 data generation (before training)

This pipeline uses `sentence-transformers/msmarco-hard-negatives`, builds a
balanced 100-negatives/query dataset, scores 5 cross-encoders, normalizes and
rescales scores, trims to 8 negatives per query, and uploads the final dataset
to `Hyukkyu/msmarco-spladev3-scores`. Training then uses existing configs.

1) Extract balanced hard negatives (top-50 + random-50 per source):

```
python script/preprocess/extract_hard_negatives.py \
  --config-name extract_hard_negatives
```

2) Score the extracted dataset with 5 cross-encoders:

```
export RANKT53B_CHECKPOINT_PATH=/path/to/trecdl22-crossencoder-rankT53b-repro/pytorch_model.bin
python script/preprocess/score_cross_encoder_ensemble.py \
  --config-name score_cross_encoder_ensemble
```

3) Build a clean multi-teacher dataset (single `train` split with 5 score
columns + `teacher_scores`) and optionally upload to HF:

```
export HF_TOKEN=...
python script/preprocess/build_multi_teacher_scores_dataset.py \
  --config-name build_multi_teacher_scores_dataset \
  upload.enabled=true \
  upload.repo_id=YOUR_USER/msmarco-hardneg-100-multi-teacher-scores
```

4) Finalize (min-max + rescore + trim to 8) and upload to HF:

```
export HF_TOKEN=...
python script/preprocess/finalize_distill_dataset.py \
  --config-name finalize_distill_dataset
```

5) Train SPLADE-v3 (existing code):

```
python script/train.py --config-name train_splade_v3 \
  training.init_checkpoint_path=/path/to/splade_pp_selfdistil.ckpt
```

Use `training.init_checkpoint_path` here for warm-starting weights. Use
`training.resume_checkpoint_path=/path/to/last.ckpt` instead when you need to
continue an interrupted run.

## Experiments and utilities

Logit distribution experiment (writes JSON + PNG):

```
python script/experiment/logit_stats.py --output_dir script/experiment/output
```

GPU burn utility:

```
python script/etc/gpu_burn.py --devices all --dtype float16
```

## Docker

```
docker build -f docker/Dockerfile -t splade-repro .
docker run --gpus all -v "$PWD:/workspace" -it splade-repro bash
```

## Config toggles

- Paper-faithful regularization: `training.regularization.paper_faithful=true|false`
- SPLADE variants: `training=splade_v1|splade_v2|splade_v2_doc`
- Distillation: `training.distill.enabled=true`
- Encode compilation: `encoding.torch_compile=true` (compiles encoder + sparsify core)
