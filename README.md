# SPLADE

Training + evaluation repo for SPLADE v1/v2 variants with BEIR support.

## Setup

```
pip install -r requirements.txt
```

## Data layout

```
data/
  msmarco/
    train.jsonl
    val.jsonl
    corpus.jsonl
    queries.dev.jsonl
    qrels.dev.tsv
    teacher_scores.jsonl   # optional
  beir/
    trec-covid/
      corpus.jsonl
      queries.jsonl
      qrels/
        test.tsv
```

### Training JSONL format

Each line must include a query, positives, and negatives.

```
{
  "query_id": "q1",
  "query": "what is splade",
  "positives": [{"doc_id":"d1","text":"...","teacher_scores": 2.3}],
  "negatives": [{"doc_id":"d2","text":"...","teacher_scores": -1.1}]
}
```

Notes:

- `positives` and `negatives` can be lists of strings or dicts with `text`.
- If distillation is enabled, `teacher_scores` must be present (or provided via `teacher_scores.jsonl`).

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

Use Hugging Face BEIR dataset directly:

```
python scripts/evaluation.py \
  testing.checkpoint_path=logs/checkpoints/last.ckpt \
  dataset=beir/trec-covid \
  dataset.use_hf=true \
  dataset.hf_name=BeIR/trec-covid
```

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
