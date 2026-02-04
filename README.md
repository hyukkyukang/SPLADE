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
  `training=splade_v2_max`.
- Override parameters: `training.use_cpu=true`,
  `testing.checkpoint_path=...`, `encoding.checkpoint_path=...`.
- Logs go to `log/` by default; set `tag=...` to create per-run log dirs.

Datasets default to the Hugging Face Hub; dataset configs live in
`config/dataset/`.

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

Disable W&B logging (optional):

```
python script/train.py training.wandb.mode=disabled
```

## Encode corpus

Encode documents into sparse shards (for retrieval indexing):

```
python script/encode.py \
  encoding.checkpoint_path=log/checkpoints/last.ckpt \
  dataset=beir/trec-covid \
  model.encode_path=log/encode
```

## Build inverted index

Build a sparse inverted index from encoded shards:

```
python script/index.py \
  model.encode_path=log/encode \
  model.index_path=log/index
```

## Evaluate (retrieval / reranking)

Index-based retrieval evaluation:

```
python script/evaluate.py \
  evaluation.type=retrieval \
  testing.checkpoint_path=log/checkpoints/last.ckpt \
  dataset=beir/trec-covid \
  model.index_path=log/index
```

Reranking evaluation (no index required):

```
python script/evaluate.py \
  evaluation.type=reranking \
  testing.checkpoint_path=log/checkpoints/last.ckpt \
  dataset=beir/msmarco
```

## Evaluate (NanoBEIR proxy)

Quick proxy evaluation without full-corpus encoding:

```
python script/evaluate_nanobeir.py \
  testing.checkpoint_path=log/checkpoints/last.ckpt \
  nanobeir.datasets='[msmarco, nfcorpus, nq]' \
  nanobeir.save_json=true
```

Use HF weights instead of a checkpoint:

```
python script/evaluate_nanobeir.py \
  nanobeir.use_huggingface_model=true \
  model.huggingface_name=naver/splade_v2_max
```

## Preprocess

Mine hard negatives with a trained checkpoint:

```
python script/preprocess/mine_hard_negatives.py \
  mining.checkpoint_path=log/checkpoints/last.ckpt \
  mining.output_dir=data/hard_negatives \
  mining.output_format=triplet
```

Score candidate pairs with a cross-encoder:

```
python script/preprocess/score_cross_encoder.py \
  dataset@score_dataset=msmarco \
  scoring.model_name=cross-encoder/ms-marco-MiniLM-L-12-v2
```

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
- SPLADE variants: `training=splade_v1|splade_v2_max|splade_v2_doc`
- Distillation: `training.distill.enabled=true`
