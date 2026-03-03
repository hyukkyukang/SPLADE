# SPLADE

Training + evaluation repo for SPLADE v1/v2 sparse retrieval with BEIR and
NanoBEIR support.

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

## Refactor docs

- Architecture: `doc/analysis/refactor_architecture.md`
- Migration notes: `doc/analysis/migration_notes.md`
- Command cookbook: `doc/analysis/command_cookbook.md`
- Debug runbook: `doc/analysis/debug_runbook.md`
- Baseline command matrix: `doc/analysis/baseline_command_matrix.md`

## Train

Train a SPLADE model and write checkpoints/logs:

```
python script/train.py training=splade_v1 model=splade_v1
```

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
  training.mlflow.experiment_name=SPLADE \
  training.mlflow.tracking_uri=http://127.0.0.1:5000
```

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

## Evaluate (NanoBEIR proxy)

Quick proxy evaluation without full-corpus encoding:

```
python script/evaluate.py --benchmark nanobeir \
  testing.checkpoint_path=log/train/splade_v2/no_tag/checkpoints/last.ckpt \
  nanobeir.datasets='[msmarco, nfcorpus, nq]' \
  nanobeir.save_json=true
```

Use HF weights instead of a checkpoint:

```
python script/evaluate.py --benchmark nanobeir \
  nanobeir.use_huggingface_model=true \
  model.huggingface_name=naver/splade_v2
```

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

ANNA tokenizer and conversion source code lives under:

```
script/preprocess/anna/
```

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
