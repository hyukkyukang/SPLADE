# LENS Training: DeepSpeed ZeRO-3 Implementation Plan

## Goal

Enable DeepSpeed ZeRO-3 (with optional CPU offload) in `script/train_lens.py`
so we can fit larger effective batches on 8×A100-40GB and close the gap to
paper-faithful training (target: MTEB family-avg within 2 pt of LENS-4000's
71.22). End-state: a `Trainer(strategy="deepspeed_stage_3")` run that
survives a 50-step smoke test at Phase 1 outer batch + sub=8, then ramps to
as close to paper batch as memory allows.

## Why ZeRO-3

Current Phase 1 VRAM (~32 GB / 33 GB available per A100-40GB after the 6 GB
orphan):

| | Phase 1 actual |
|---|---|
| Frozen Mistral-7B base (bf16) | 14 GB |
| LoRA adapters | 0.1 GB |
| Optimizer state (LoRA only, fp32) | 0.4 GB |
| Activations (sub_batch=8, with grad checkpointing) | ~17 GB |
| Total | ~32 GB |

ZeRO-3 shards model weights across 8 ranks, freeing ~12 GB per GPU. That
budget gets reallocated to bigger sub_batch_size + bigger outer batch,
which directly increases negatives-per-query in the contrastive loss.

Tradeoff: ~15-25% throughput cost from per-layer all-gather of params.

## Pre-flight checks

Run once before any code changes:

| # | Check | How | Pass criterion |
|---|---|---|---|
| 0.1 | DeepSpeed installed in venv | `python -c "import deepspeed; print(deepspeed.__version__)"` | ≥ 0.14 |
| 0.2 | Lightning DeepSpeed strategy importable | `python -c "from lightning.pytorch.strategies import DeepSpeedStrategy"` | No error |
| 0.3 | PEFT version compatibility with ZeRO-3 | `pip show peft` | ≥ 0.10 |
| 0.4 | The 6 GB GPU orphan still present? | `nvidia-smi` | If yes, ZeRO-3 helps less than projected |
| 0.5 | Phase 1 reproducible at current config | smoke run | Loss curve matches prior |

## Phase A — Plumbing (no behavior change)

**Goal:** make the strategy switchable from one knob, default unchanged.

### A.1 Config plumbing — `config/training/_base.yaml`

```yaml
strategy: ddp                 # existing
deepspeed:                    # new
  enabled: false
  stage: 3                    # 1, 2, or 3
  offload_optimizer: false    # page AdamW state to CPU
  offload_params: false       # page model weights to CPU between layers
  config_path: null           # custom JSON path; null = Lightning defaults
```

### A.2 Strategy builder — `script/train_lens.py`

Replace the single `strategy=str(cfg.training.strategy)` line in
`L.Trainer(...)` with a helper:

```python
def _build_strategy(cfg: DictConfig) -> str | object:
    ds = cfg.training.get("deepspeed") or {}
    if not bool(ds.get("enabled", False)):
        return str(cfg.training.strategy)  # current path
    from lightning.pytorch.strategies import DeepSpeedStrategy
    config_path = ds.get("config_path")
    if config_path:
        return DeepSpeedStrategy(config=config_path)
    return DeepSpeedStrategy(
        stage=int(ds.get("stage", 3)),
        offload_optimizer=bool(ds.get("offload_optimizer", False)),
        offload_parameters=bool(ds.get("offload_params", False)),
        partition_activations=False,
    )
```

### A.3 DeepSpeed-aware gradient clipping

`LENSTrainingModule.configure_gradient_clipping`: early-return when
DeepSpeed is in charge (it owns clipping via `gradient_clipping` config
field):

```python
def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                gradient_clip_algorithm=None):
    from lightning.pytorch.strategies import DeepSpeedStrategy
    if isinstance(self.trainer.strategy, DeepSpeedStrategy):
        return
    # ... existing manual clip path
```

### A.4 Launch script env override

`script/etc/launch_lens_phase2.sh`: add `ENABLE_DEEPSPEED` toggle that
appends `+training.deepspeed.enabled=true` (etc.) to the Hydra overrides.

### A.5 Acceptance for Phase A

Smoke run with `ENABLE_DEEPSPEED=0` (default) — must produce **identical**
loss curve to current main. Byte-for-byte same first-step loss vs commit
`6257157`.

**Effort:** 1-2 hours.

## Phase B — Minimal ZeRO-3 config + smoke

**Goal:** Lightning + DeepSpeed Stage 3 starts and runs 50 steps cleanly.

### B.1 `config/deepspeed_stage3.json`

```json
{
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": {"enabled": "auto"},
  "fp16": {"enabled": false},
  "optimizer": {
    "type": "AdamW",
    "params": {"lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto"}
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": "auto", "warmup_max_lr": "auto",
      "warmup_num_steps": "auto", "total_num_steps": "auto"
    }
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false,
  "steps_per_print": 1000
}
```

### B.2 Strip our fused-AdamW under DeepSpeed

`LENSTrainingModule.configure_optimizers`: when DeepSpeed is in charge,
return vanilla AdamW (no `fused=True`). DeepSpeed replaces it with
`FusedAdam` automatically.

### B.3 Smoke test command

```
LENS_PROFILE=1 ENABLE_DEEPSPEED=1 MAX_STEPS=50 \
  bash script/etc/launch_lens_phase2.sh
```

### B.4 Acceptance

1. Process runs 50 steps, no crash
2. Loss values sane (no NaN, broadly tracks Phase 1 magnitudes)
3. `nvidia-smi` shows reduced model-weight memory per GPU vs DDP
4. MLflow logging works
5. Profiler trace lands in `${log_dir}/profile/`

**Risks:**
- PEFT may trip on ZeRO-3's parameter discovery when LoRA is mixed with
  frozen base. Mitigation: `Z3LeafModule` annotation pattern, or fall back
  to ZeRO-2 if blocked.
- `dist_gather_with_local_grad` operates on outputs not params — should be
  unaffected; verify in this smoke.

**Effort:** 4-6 hours.

## Phase C — sub_batch_size memory probe

**Goal:** find the new ceiling for `sub_batch_size` under ZeRO-3 at Phase 1
outer batch.

```
for sub in 16 32 64; do
  ENABLE_DEEPSPEED=1 MAX_STEPS=30 SUB_BATCH_SIZE=${sub} \
    TAG=ds3_sub${sub}_probe \
    bash script/etc/launch_lens_phase2.sh
done
```

For each: record peak GPU memory (`nvidia-smi dmon`), avg step time,
OOM-or-not.

**Acceptance:** at least one config above sub=8 baseline runs cleanly for
30 steps. Expected: sub=64 fits comfortably under ZeRO-3.

**Effort:** ~half day.

## Phase D — Outer batch ramp toward paper

**Goal:** scale `BATCH`, `SYMM`, `TG` toward paper geometry, pinning
`SUB_BATCH_SIZE` to the value found in Phase C.

Smoke each step (30 steps each, ~10 min):

```
ds3_sym32      — bump symmetric_batch from 8 to 32
ds3_sym64      — to 64 (paper config: 256 / 4)
ds3_sym32_tg4  — also bump train_group from 2 to 4
ds3_sym64_tg4_b2 — also bump per-rank batch
... etc.
```

Effective negatives per query:
```
neg_per_query = SYMMETRIC_BATCH_SIZE × SYMMETRIC_TRAIN_GROUP_SIZE × n_ranks
```
- Phase 1: 8 × 2 × 8 = **128 negatives**
- Paper: 256 × 8 × 16 = **32,768 negatives**

**Acceptance:** find config maximizing `neg_per_query` while passing 30-step
smoke + memory < 36 GB.

**Effort:** ~1 day.

## Phase E — CPU offload (only if D insufficient)

**Goal:** if D's max `neg_per_query` is < 1000, enable optimizer (and
optionally param) CPU offload for additional headroom.

Toggle `cfg.training.deepspeed.offload_optimizer=true` (and optionally
`.offload_params=true`). Measure throughput cost vs Phase D's best config —
expect 30-50% slowdown.

**Acceptance:** find config that fits paper-near batch geometry with
acceptable throughput.

**Effort:** 2-4 hours.

## Phase F — Checkpoint merge tooling

**Goal:** convert DeepSpeed-sharded checkpoint to single-file HF format for
MTEB eval.

### F.1 `script/etc/merge_deepspeed_lens_ckpt.py`

Uses `deepspeed.utils.zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict`
to merge sharded state, then saves as `model.safetensors` + `config.json`
+ tokenizer files in HF format.

### F.2 Validation

Convert a 50-step Phase B checkpoint, run a 1-task MTEB eval (NFCorpus only)
on the merged file. Score should be plausible (we trained ~50 steps so won't
be great — just sanity check that the conversion preserved weights).

**Effort:** 2-3 hours.

## Phase G — Full Phase 2 launch

**Goal:** actual paper-faithful training run.

### G.1 Final config (filled from Phase D/E results)

```
ENABLE_DEEPSPEED=1 \
  TAG=phase2_ds3_paper-batch_$(date +%Y%m%d) \
  MAX_STEPS=33000 \                    # ~1 epoch at this batch geometry
  BATCH_SIZE=<from D> \
  SYMMETRIC_BATCH_SIZE=<from D> \
  TRAIN_GROUP_SIZE=<from D> \
  SUB_BATCH_SIZE=<from C> \
  LR=1e-4 \                            # paper value
  WARMUP_STEPS=100 \                   # paper value
  bash script/etc/launch_lens_phase2.sh
```

### G.2 Monitoring

Watcher polls every 30 min:
- Verify loss is non-NaN
- Post step / hour rate
- Save checkpoints every 1000 steps

Expected wall: 2-4 days for 1 epoch depending on D's batch and E's offload.

### G.3 Eval and compare

After completion:
1. Merge final checkpoint via Phase F's script
2. Run MTEB eval (drop heavy retrievals as in Phase 1, or commit to ~22h
   full eval)
3. `script/etc/compare_lens_mteb.py` against LENS-4000 paper Table 1

**Success criterion:** overall MTEB family-avg within 2 pt of paper's 71.22.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PEFT + ZeRO-3 incompatibility | medium | blocks B.4 | Fall back to ZeRO-2; or upgrade PEFT |
| `dist_gather` semantics change under ZeRO-3 | low | wrong loss | 2-rank test comparing tensor shapes/values |
| MLflow logger conflicts with DeepSpeed | low | logs missing | Easy fix |
| Custom `BatchSampler` not respected by DeepSpeed | medium | wrong cardinality | Verify in B.4 |
| Checkpoint merge produces wrong weights | low | bad eval | F.2 catches it |
| ZeRO-3 + grad-checkpointing incompatibility | medium | OOM or slow | Lightning handles it; if not, disable grad-checkpoint and rely on offload |
| Throughput cost makes 1 epoch impractical (7+ days) | medium | can't ship | Reduce max_steps or batch geometry |

## Rollback plan

Each phase is gated by acceptance criteria. If Phase B smoke fails after
debugging, set `cfg.training.deepspeed.enabled=false` and the existing
DDP+sub=8 path remains the safe default. Strategy is opt-in; no breaking
changes to existing entry points.

## Realistic timeline

| Phase | Effort |
|---|---|
| Pre-flight + A | 0.25 day |
| B (smoke) | 0.5 day |
| C (sub_batch sweep) | 0.5 day |
| D (outer batch ramp) | 1 day |
| E (offload, if needed) | 0.25 day |
| F (checkpoint merge) | 0.25 day |
| **Setup + smoke subtotal** | **~2.75 days** |
| G launch + train + eval | 10 min launch + 2-4 day train + ~22h eval |

## Open decisions

1. **PEFT version pin** — current (0.x) or upgrade to latest before starting?
2. **Lightning vs raw DeepSpeed** — Lightning's `DeepSpeedStrategy` is
   what's outlined here. HF Trainer would match paper's stack exactly but
   is a much bigger rewrite. Pick based on trust of Lightning's integration.
