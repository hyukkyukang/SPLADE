# SPLADE

Training + evaluation repo for SPLADE v1/v2 variants with BEIR support.

## Setup

```
pip install -r requirements.txt
```

## Datasets (Hugging Face Hub)

This repo is configured to use Hugging Face Hub datasets for training, evaluation, and indexing.
All dataset selection happens via config (e.g., `dataset.hf_name`, `dataset.hf_subset`, `dataset.hf_split`).

## Train

```
python scripts/train.py training=splade_v1 model=splade_v1_sum
```

Use Hugging Face MS MARCO (sentence-transformers/msmarco):

```
python scripts/train.py \
  dataset@train_dataset=msmarco_hf_train \
  dataset@val_dataset=msmarco_hf_val
```

Use MiniLM distillation score dataset for training (triplets with scores):

```
python scripts/train.py \
  dataset@train_dataset=msmarco_minilm_scores \
  training.distill.enabled=true
```

Enable BEIR sampled evaluation per epoch:

```
python scripts/train.py \
  training.beir_eval.enabled=true \
  training.beir_eval.datasets='[trec-covid, nfcorpus]' \
  training.beir_eval.sample_size=128
```

## Evaluate (full BEIR / MS MARCO)

```
python scripts/evaluation.py \
  testing.checkpoint_path=logs/checkpoints/last.ckpt \
  dataset=beir/trec-covid
```

Datasets are loaded from the Hub by default, so no local file paths are required.

## Build a index

```
python scripts/index.py \
  testing.checkpoint_path=logs/checkpoints/last.ckpt \
  dataset=beir/trec-covid \
  model.index_path=logs/index
```

## Docker

```
docker build -f docker/Dockerfile -t splade-repro .
docker run --gpus all -v "$PWD:/workspace" -it splade-repro bash
```

## Config toggles

- Paper-faithful vs normal: `training.regularization.paper_faithful=true|false`
- SPLADE v1 vs v2: `training=splade_v1` or `training=splade_v2`
- Distillation: `training.distill.enabled=true`
