# Command Cookbook

This is a practical command matrix for common train/eval/validation tasks.

## Train

Baseline SPLADE v2++:

```bash
python script/train.py \
  model=splade_v2_pp \
  training=splade_v2_pp
```

EmbeddingGemma SPLADE v2++:

```bash
python script/train.py \
  --config-name train_embeddinggemma_splade_v2_pp
```

Four-GPU DDP with explicit devices:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python script/train.py \
  --config-name train_embeddinggemma_splade_v2_pp \
  training.num_devices=4 \
  training.strategy=ddp
```

Enable compile + max-autotune:

```bash
python script/train.py \
  --config-name train_embeddinggemma_splade_v2_pp \
  training.torch_compile=true \
  training.torch_compile_mode=max-autotune
```

## Validation (training-style reranking + NanoBEIR)

```bash
python script/validation.py \
  testing.checkpoint_path=/abs/path/to/checkpoint.ckpt
```

Override validation subset sizing:

```bash
python script/validation.py \
  testing.checkpoint_path=/abs/path/to/checkpoint.ckpt \
  val_dataset.hf_max_samples=4096 \
  nanobeir.enabled=true
```

## Retrieval Evaluation (index-based)

```bash
python script/evaluation.py \
  testing.checkpoint_path=/abs/path/to/checkpoint.ckpt \
  encoding.index_dir=/abs/path/to/index \
  dataset=beir/nfcorpus
```

## Encode + Index

```bash
python script/encode.py \
  encoding.checkpoint_path=/abs/path/to/checkpoint.ckpt \
  dataset=beir/nfcorpus \
  encoding.encode_dir=/abs/path/to/encode_out
```

```bash
python script/index.py \
  encoding.encode_dir=/abs/path/to/encode_out \
  encoding.index_dir=/abs/path/to/index_out
```

## Pretokenize Benchmark

Three-way sidecar/hybrid/no-pretokenize comparison:

```bash
python script/benchmark/compare_pretokenize_variants.py \
  --config-name train_embeddinggemma_splade_v2_pp \
  --model-name splade_v2_pp_embeddinggemma_300m_lsr \
  --cuda-visible-devices 0,1,2,3
```

