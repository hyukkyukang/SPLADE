# Baseline Command Matrix

This matrix defines the baseline commands used for reproducible runtime/throughput tracking.

## 1) Train Throughput Baseline (sidecar/hybrid/no-pretokenize)

```bash
python script/benchmark/compare_pretokenize_variants.py \
  --config-name train_embeddinggemma_splade_v2_pp \
  --model-name splade_v2_pp_embeddinggemma_300m_lsr \
  --cuda-visible-devices 0,1,2,3 \
  --output-json benchmark_results/pretokenize_compare_latest.json
```

Reference snapshot:
- `benchmark_results/pretokenize_compare_latest.json`
- `benchmark_results/pretokenize_compare_latest.md`

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

