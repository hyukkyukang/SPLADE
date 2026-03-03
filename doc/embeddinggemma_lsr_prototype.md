# EmbeddingGemma-LSR Prototype

This prototype builds a learned sparse retriever with a custom target vocabulary (`V_target`) and semantic projection-head initialization for `google/embeddinggemma-300m`.

## Artifacts

1. Vocab build outputs:
- `v_target.txt`
- `df_map.json`
- `vocab_stats.json`
- `manifest.json`

2. Initialization outputs:
- HF backbone/tokenizer files in output directory
- `lsr_projection.pt`
- `lsr_config.json`
- `target_vocab.txt`
- `df_map.json`
- `tokenization_report.json`
- `init_metadata.json`

3. Training outputs:
- `best/` checkpoint directory
- `last/` checkpoint directory
- `train_logs.jsonl`
- `run_summary.json`

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install spaCy English model:
```bash
python -m spacy download en_core_web_trf
```

## Execute

1. Build `V_target` from train+val meta splits:
```bash
python script/preprocess/build_embeddinggemma_lsr_vocab.py \
  --config config/prototype/embeddinggemma_lsr_vocab.yaml
```

2. Initialize semantic projection head:
```bash
python script/model/init_embeddinggemma_lsr.py \
  --config config/prototype/embeddinggemma_lsr_init.yaml
```

3. Train full run:
```bash
python script/train_embeddinggemma_lsr.py \
  --config config/prototype/embeddinggemma_lsr_train.yaml
```

## Notes

- Fragmented terms are added via `tokenizer.add_tokens(...)` (regular added tokens, not special-control tokens).
- Training uses in-batch InfoNCE plus FLOPs regularization:
  - `L_total = L_InfoNCE + lambda_q * L_FLOP(q) + lambda_d * L_FLOP(d)`
- By default, validation metrics are computed at `@10` on sampled query/doc pools to keep evaluation tractable.
