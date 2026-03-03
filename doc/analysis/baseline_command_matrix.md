# Baseline Command Matrix

This matrix defines the baseline commands used for reproducible runtime/throughput tracking.

## 1) Train Throughput Baseline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python script/train.py \
  --config-name train_embeddinggemma_splade_v2_pp \
  training.num_devices=4 \
  training.strategy=ddp
```

Reference snapshot:
- `benchmark_results/*train*`

## 2) Validation Runtime Baseline (MSMARCO rerank + NanoBEIR)

```bash
python script/validation.py \
  validation.checkpoint_path=/abs/path/to/checkpoint.ckpt \
  validation.include_nanobeir=true
```

## 3) Retrieval Evaluation Runtime Baseline (index-based)

```bash
python script/evaluation.py \
  testing.checkpoint_path=/abs/path/to/checkpoint.ckpt \
  encoding.index_dir=/abs/path/to/index \
  dataset=beir/nfcorpus
```

## 4) End-to-End Smoke (train -> encode -> index -> evaluation)

```bash
bash script/smoke_train_encode_index_eval.sh
```
