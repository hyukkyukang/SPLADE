# EmbeddingGemma SPLADE Native Workflow

For the detailed build and rerun guide, see
`doc/embeddinggemma_lsr_build_pipeline.md`.

This workflow keeps model creation separate from training:

1. Build a target vocabulary from corpus statistics.
2. Build a Hugging Face CausalLM backbone with LM-head rows initialized for target terms.
3. Train with the original SPLADE training stack (`script/train.py`) using `splade_v2_pp` style.

## 1) Build Target Vocabulary

```bash
python -u script/model_creation/embeddinggemma_splade/build_target_vocab.py \
  --config config/model_creation/embeddinggemma_splade/vocab.yaml
```

Outputs:
- `outputs/model_creation/embeddinggemma_splade/vocab/v_target.txt`
- `outputs/model_creation/embeddinggemma_splade/vocab/df_map.json`

## 2) Build HF Backbone

```bash
python -u script/model_creation/embeddinggemma_splade/build_hf_backbone.py \
  --config config/model_creation/embeddinggemma_splade/hf_backbone.yaml
```

Outputs:
- Hugging Face model dir: `outputs/model_creation/embeddinggemma_splade/hf_backbone`
- Includes `config.json`, model weights, tokenizer files, and initialization metadata.

## 3) Train With Original SPLADE Codebase

The model config `config/model/splade_v2_pp_embeddinggemma_300m_lsr.yaml` points to the created HF backbone and keeps SPLADE settings compatible with native training.

Run with 4 GPUs (DDP):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u script/train.py \
  --config-name train_embeddinggemma_splade_v2_pp \
  training.num_devices=4
```

Notes:
- This uses native Lightning DDP from `training.strategy=ddp`.
- Training profile is `splade_v2_pp` style with only the model backbone changed.
