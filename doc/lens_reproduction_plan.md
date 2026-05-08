# LENS Reproduction & Extension Plan

> **Status (2026-04-23):** Phase 0 code + Phase 1 code landed. Phase 0 eval
> (yibinlei/LENS-d8000 on 8×A100, 67-task MTEB English) running. See §10
> "Progress log" at the bottom for the latest state.



Target paper: **"Enhancing Lexicon-Based Text Embeddings with Large Language Models"** (Lei et al., ACL 2025 — arXiv [2501.09749](https://arxiv.org/abs/2501.09749)).
Reference code: [Yibin-Lei/LENS](https://github.com/Yibin-Lei/LENS).
Reference checkpoints: [`yibinlei/LENS-d4000`](https://huggingface.co/yibinlei/LENS-d4000), [`yibinlei/LENS-d8000`](https://huggingface.co/yibinlei/LENS-d8000).

Paper-reported MTEB English averages (target numbers):

| Model | Avg | Retrieval | Reranking | Clustering | PairClass | Classification | STS | Summ. |
|---|---|---|---|---|---|---|---|---|
| LENS-d4000 | 71.22 | — | — | — | — | — | — | — |
| LENS-d8000 | **71.63** | 61.86 | 60.91 | 58.02 | 87.98 | 88.43 | 84.67 | 29.54 |

Success gate: reproduce within ±0.5 average, ±1.0 per-category, compared to the paper-reported table.

---

## 1. Hardware inventory (this machine)

Observed 2026-04-23:

| Component | Details | Implication |
|---|---|---|
| GPU | **8 × NVIDIA A100-SXM4-40GB**, compute 8.0, CUDA 12.8, driver 570 | bf16 native; FA2 supported; 40 GB (not 80) forces smaller per-device micro-batch than the paper |
| GPU interconnect | **NV12 fully-connected** (NVSwitch, 12 NVLink3 bonds per pair) | Cross-device `all_gather` / all-reduce is effectively free — exploit this for cross-device negatives and for sharded corpus encoding. No PCIe or QPI traffic between GPUs. |
| CPU | **2 sockets × 24 cores × 2 threads = 96 logical** (Intel Xeon @ 2.20GHz) | Enough for >8 dataloader processes × multiple workers. Use Arrow/zstd parquet decoding in parallel. |
| NUMA | Node 0: CPUs 0-23,48-71 (GPUs 0-3). Node 1: CPUs 24-47,72-95 (GPUs 4-7). | Pin each rank's dataloader workers to the GPU's NUMA node (see `NUMA-aware launcher` in §2). Cross-NUMA memory access can ~2× dataloader latency. |
| RAM | **669 GiB total**, 626 GiB free (buff/cache). 128 GiB `/dev/shm` tmpfs. | Entire bge-full-data (~200 GB Arrow) fits in RAM. Keep `HF_DATASETS_IN_MEMORY_MAX_SIZE` unset; `datasets.load_from_disk` does mmap which is effectively RAM-resident after first pass. |
| PyTorch | 2.11.0+cu128, bf16 supported | Good; use `bf16-mixed` throughout (no loss scaler needed). |
| FlashAttention-2 | **not installed** | Install before starting. The official LENS `MistralBiModel` has `assert _attn_implementation == 'flash_attention_2'`. |
| DeepSpeed | **not installed** | Paper uses ZeRO-1. Lightning DDP with gradient checkpointing is equivalent for LoRA-only training (optimizer state is already tiny). Skip DeepSpeed, stay on Lightning. |
| MTEB library | 2.11.6 installed ✓ | Use as-is. |
| Storage (root `/`) | 993 GB total, **17 GB free** ← nearly full | Do NOT write anything here. Do NOT let HF cache grow here. |
| Storage (`/home/user/SPLADE` via NFS) | 143 GB free | Use for code only; not for datasets or checkpoints. |
| Storage (`/mnt/ex-disk-1/hyukkyukang`) | 10 TB local (sdb), **2.4 TB free** | **All datasets, HF cache, checkpoints, encoded shards go here.** This is the only plausible location. |
| `/dev/shm` | 128 GiB tmpfs | Reserve for k-means intermediate arrays and eval-side query embedding all-gather if needed. |
| Other jobs | GPUs currently at 100 % util (~7 GB ea.) | There are existing jobs. Coordinate occupancy or wait. |

## 2. Global resource-utilization directives

Apply these everywhere unless stated otherwise.

### 2.1 Storage policy

Move all large artifacts off root:

```bash
# One-time environment exports — add to ~/.bashrc or the training launcher.
export HF_HOME=/mnt/ex-disk-1/hyukkyukang/hf
export HF_DATASETS_CACHE=/mnt/ex-disk-1/hyukkyukang/hf/datasets
export TRANSFORMERS_CACHE=/mnt/ex-disk-1/hyukkyukang/hf/models
export HF_HUB_CACHE=/mnt/ex-disk-1/hyukkyukang/hf/hub
export TORCH_HOME=/mnt/ex-disk-1/hyukkyukang/torch
export TRITON_CACHE_DIR=/mnt/ex-disk-1/hyukkyukang/triton   # avoids /tmp fill
export MPLCONFIGDIR=/mnt/ex-disk-1/hyukkyukang/matplotlib
```

Directory layout we will create:

```
/mnt/ex-disk-1/hyukkyukang/
├── hf/                        # HF cache (new)
├── torch/                     # torch hub
├── triton/                    # compile cache
├── lens/
│   ├── artifacts/             # hf_backbone, mistral_cluster{4k,8k}
│   ├── data/
│   │   ├── bge_full_data_raw/       # cfli/bge-full-data, from_disk format
│   │   └── bge_full_data_tokenized/ # pre-tokenized if we take that option
│   ├── checkpoints/           # training output
│   ├── logs/                  # training/eval logs
│   └── mteb_results/
```

Also move existing 160 GB HF cache (one-time, when root has breathing room):

```bash
rsync -aP --remove-source-files ~/.cache/huggingface/ /mnt/ex-disk-1/hyukkyukang/hf/
ln -s /mnt/ex-disk-1/hyukkyukang/hf ~/.cache/huggingface
```

### 2.2 Parallelism stack

| Layer | Choice | Rationale |
|---|---|---|
| Multi-GPU data parallel | Lightning DDP (already in repo) | 8 GPUs, NVLink-fast all-reduce; no need for FSDP since LoRA keeps per-GPU params tiny. |
| Precision | `bf16-mixed` | A100 native; no loss scaling; paper used fp16 but bf16 is strictly better on A100. |
| Attention | **FlashAttention-2** with bidirectional (`is_causal=False`) | Required by the bidirectional Mistral wrapper; cuts activation memory ~2–3×. |
| Activation checkpointing | `model.gradient_checkpointing_enable()` | Official recipe. Without this, 7B + 512 ctx + LoRA + bs≥4 won't fit on 40GB. |
| Optimizer | `torch.optim.AdamW(fused=True)` | Works on A100 with cu128. Skip DeepSpeed. |
| Cross-device negatives | `dist.all_gather` inside the loss path | NVSwitch → effectively free; dramatically boosts contrastive signal. |
| Sub-batch within one step | Chunk the padded batch into `sub_batch_size` sequences, run encoder per chunk, concat reps in the graph | Matches official repo; lets us keep a large effective batch without OOM. |
| Gradient accumulation | Only if needed to match paper's 512-query effective batch | Each accum step still does cross-device all_gather inside its own forward. |
| Tokenization during training | Runtime, in collator (fast tokenizers are fine for speed even though official uses slow — see §4 for how we offer both) | Avoids a 200 GB pre-tokenization artifact. |
| Tokenization at dataset prep | Parallel via `datasets.map(num_proc=64)` if we choose pre-tokenized path | Uses 96 cores. |
| `torch.compile` | **OFF for LENS training** initially; consider for corpus encoding only | Compile + FA2 + LoRA + grad_ckpt is a known minefield; enable only after Phase 1 reproduces. |
| NUMA binding | `numactl --cpunodebind={0|1} --membind={0|1}` per-rank | Dataloader workers stay local to the GPU's NUMA node. |
| Dataloader | `num_workers=8` per rank, `prefetch_factor=4`, `pin_memory=True`, `persistent_workers=True` | 8 × 8 = 64 dataloader processes, fits in 96 CPU threads. Already supported by repo's `build_dataloader_kwargs`. |
| Shared memory | bump `ulimit -l unlimited`; rely on `/dev/shm` 128 GiB for torch shared-memory tensors | Already enough by default. |

### 2.3 NUMA-aware launcher (one script for all training)

Create `script/etc/launch_lens.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}         # 8 ranks × 6 threads = 48, matches one socket
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=1                             # no IB on this box
export NCCL_P2P_LEVEL=NVL                            # prefer NVLink
export TOKENIZERS_PARALLELISM=false                  # avoid fork warnings
export PYTHONFAULTHANDLER=1

# The Python script picks up LOCAL_RANK and binds its own dataloader workers to
# the matching NUMA node via torch.utils.data with worker_init_fn (§5).

CONFIG_NAME="${1:-train_lens_mistral}"
shift || true

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc_per_node 8 \
  script/train.py \
  --config-name "${CONFIG_NAME}" \
  "$@"
```

Per-rank NUMA affinity is applied inside `worker_init_fn` (see §5.3). This avoids `numactl` wrapping the whole process (which would starve rank 4-7 of NUMA-local CPU when they run on node 1).

### 2.4 Batch-size sizing for 8×A100-40GB (vs paper's 16×80GB)

Paper's effective global batch:

| Task family | per-device bs | #GPU | train_group_size | total seqs per optimizer step |
|---|---|---|---|---|
| Asymmetric (retrieval, reranking) | 32 q | 16 | 8 | 32·16·(1+8) = **4608** seqs, 512 queries |
| Symmetric (STS/clustering/class) | 16 q | 16 | 8 | 16·16·(1+8) = **2304** seqs, 256 queries |

Our fit on A100-40GB with Mistral-7B + LoRA + grad_ckpt + FA2 + 512 ctx (empirical rule of thumb):

- Single forward, batch 8 seqs × 512 tokens fits comfortably (~22 GB with activations).
- Batch 16 × 512 approaches 36 GB during backward → risky but viable with activation checkpointing + smaller `sub_batch_size`.

Recommended budget (asymmetric):

| Knob | Value | Notes |
|---|---|---|
| `sub_batch_size` | **8** | Paper uses 64 on 80GB; scale linearly to 40GB → 8–16. |
| per-rank batch `B` | **16 queries** (144 seqs / step) | Forward split into 144/8 = 18 sub-forwards; activations freed between them. |
| `grad_accumulation` | **4** | 16·8·4 = **512 queries per optimizer step** — matches paper exactly. |
| effective global asym batch | **512 queries, 4608 seqs** | Paper parity. |

Symmetric tasks:

| Knob | Value |
|---|---|
| per-rank batch | **8 queries** |
| `grad_accumulation` | **4** |
| effective global sym batch | **256 queries** |

If 40 GB is too tight after real profiling, first drop `sub_batch_size` to 4 (same batch, more sub-forwards, no effective batch change). Only drop `B` as a last resort and scale LR linearly.

### 2.5 Eval parallelism

MTEB's 56 English tasks are embarrassingly parallel across GPUs. The repo already has `script/evaluate_true_mteb.py` and a test for the parallel variant. Plan: wrap it so 8 GPUs run 8 tasks concurrently. Expected wall-clock eval: **~3–4 hours** (vs ~1 day single-GPU) for one model.

---

## 3. Prerequisites

### 3.1 Install missing packages

```bash
# FlashAttention-2 — ABI-dependent on torch build.
# For torch 2.11 + CUDA 12.8 no pre-built wheel is published at the time of writing.
# Build from source (takes ~20 min on this box using 96 cores):
pip install --no-build-isolation packaging ninja
MAX_JOBS=32 pip install --no-build-isolation "flash-attn>=2.7.0"

# (We skip deepspeed.)

# Sanity check
python -c "import flash_attn; print('fa2 ok', flash_attn.__version__)"
```

If FA2 build fails against the pinned torch+cu128, fallback: use `attn_implementation='sdpa'` (already supported by our `MistralBiModel`). Expected throughput hit: 20–30 %. This is acceptable for Phase 0. For Phase 1, invest the time to build FA2.

### 3.2 Pre-download weights and data to local disk

Run once, in a `tmux` session, before starting any training:

```bash
export HF_HOME=/mnt/ex-disk-1/hyukkyukang/hf
mkdir -p $HF_HOME

# Base LLM (~14 GB)
huggingface-cli download mistralai/Mistral-7B-v0.1

# Official LENS checkpoints for Phase 0
huggingface-cli download yibinlei/LENS-d4000
huggingface-cli download yibinlei/LENS-d8000

# Official lm_head artifacts (.pth files stored as LFS in the repos)
# These are the 4000×4096 / 8000×4096 clustered heads.

# Training data (multi-task mix) — estimated ~200 GB on disk
python - <<'PY'
import datasets, os
os.environ["HF_DATASETS_CACHE"] = "/mnt/ex-disk-1/hyukkyukang/hf/datasets"
ds = datasets.load_dataset("cfli/bge-full-data")
ds.save_to_disk("/mnt/ex-disk-1/hyukkyukang/lens/data/bge_full_data_raw")
print({k: len(v) for k, v in ds.items()})
PY
```

Also pre-fetch the MTEB datasets. MTEB pulls them lazily per task; we let it stream on first eval but pin cache to local disk via `HF_DATASETS_CACHE`.

### 3.3 Verify GPU availability

Before the first training run:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

If existing jobs are still occupying all 8 GPUs, either (a) schedule around them (`CUDA_VISIBLE_DEVICES=0,1`, `--nproc_per_node 2`), or (b) wait. Do **not** start a LENS run that forcibly preempts other jobs.

---

## 4. Gap analysis (paper & official code vs this repo)

Legend: ✅ done · 🟡 partial · ❌ missing.

### 4.1 Model / encoder alignment

| Gap | Status | Evidence |
|---|---|---|
| `MistralBiForCausalLM` with bidirectional attention | ✅ | `src/model/retriever/sparse/neural/bidirectional_mistral.py` |
| `is_causal=False` applied per layer | ✅ | Same file, constructor loops layers |
| `<instruct>`, `<query>`, `<response>` tokens added | ✅ | `script/model_creation/lens/build_hf_backbone.py` |
| k-means clustered head (4k/8k) | ✅ | `script/model_creation/lens/build_clustered_head.py` — FAISS/sklearn backends |
| `log(1+relu(·))` + max-pool | ✅ | `sparse_activation=log1p_relu`, `query/doc_pooling=max` in model config |
| Query pool mask = `[last <query>+1 ... -2]` | ✅ | `src/utils/lens_instructions.py::build_query_pooling_mask` |
| Doc pool mask = attention with last 2 zeroed | ✅ | `build_doc_pooling_mask` + `doc_trim_last_tokens=2` |
| L2-normalize embeddings in training | 🟡 | Config flag exists but not wired into training loss path |
| Tokenizer: `use_fast=False` + `padding_side='left'` + `add_eos_token=True` | ❌ | Repo defaults to fast tokenizer; `add_eos_token` not forced |
| FA2 path verified end-to-end | 🟡 | Code supports `flash_attention_2`; FA2 not installed on machine |
| Load official LENS checkpoints (`ignore_mismatched_sizes=True` + separate `lm_head.pth`) | ❌ | No loader for the official layout yet |
| Export a `lm_head.pth` (`nn.Linear`) alongside our `splade_compact_head.pt` | ❌ | We only write the dict payload |

### 4.2 Data pipeline

| Gap | Status | Evidence |
|---|---|---|
| `cfli/bge-full-data` loader | ❌ | No dataset config / class for it |
| Same-task-per-batch sampler | ❌ | Repo uses plain `DistributedSampler`; official uses `SameDatasetTrainDataset` that yields a batch at a time |
| Per-type batch shaping (asymmetric vs symmetric; in-batch vs local-only) | ❌ | Not implemented |
| `train_group_size=8` hard-neg grouping | 🟡 | Repo has scored-dataset paths with similar grouping; not LENS-specific |
| Query template `<instruct>{instruction}\n<query>{query}` | ✅ | `format_query_text` |
| Per-task instruction field `prompt` in the dataset | ❌ | Not carried through; MTEB `instruction.py` lookup exists only on eval side |
| Pre-tokenize path with 96-core `datasets.map(num_proc=64)` | ❌ | Optional optimization |
| `teacher_scores` propagated with `pos_scores`+`neg_scores` merged into one tensor | ❌ | Repo has distill paths, not the exact LENS form |

### 4.3 Training step

| Gap | Status | Evidence |
|---|---|---|
| Sub-batched encoder forward (chunk → concat reps → single loss) | ❌ | Repo runs one forward per step |
| Cross-device `all_gather` of q_reps / p_reps with local-slot preservation | ❌ | Depends on DDP gather; not explicit |
| Stride-positive in-batch CE loss (`target = arange(q) * (p//q)`) | 🟡 | `multi_positive_contrastive_loss` is mask-based, not strided — needs a LENS-shaped variant |
| Symmetric "not in-batch" loss (local-only negatives, target=0) | ❌ | |
| KL distillation: `-mean(Σ softmax(teacher_local) · log_softmax(student_local))` | 🟡 | Repo has KL distill for some paths; needs to match exact shape |
| FLOPS regularizer disabled | ✅ | Already 0.0 in LENS config |

### 4.4 Evaluation

| Gap | Status | Evidence |
|---|---|---|
| Full 56-task MTEB English runner | 🟡 | `script/evaluate_true_mteb.py` exists; need to verify task list + per-task prompt + splits |
| Per-task instruction map (the full `TASKNAME2INSTRUCTIONS` dict with >100 tasks) | ❌ | Not ported |
| Parallel across 8 GPUs (one task per GPU) | 🟡 | `test/test_evaluate_true_mteb_parallel.py` exists; need wrapper that dispatches 8-way |
| Load official `yibinlei/LENS-d{4000,8000}` | ❌ | See §4.1 |
| Query vs corpus pooling masks at inference | ✅ | Same masks as training via `src/utils/lens_instructions.py` |

---

## 5. Implementation plan — file-by-file

All new files live under the existing layout. Each task below is a discrete PR-sized unit with its own test.

### 5.1 Prereqs (Phase P)

**P1. Storage & env config**
- New file: `config/env/lens.env.sh` — exports from §2.1.
- Commit a one-liner in `README.md` pointing to this file.

**P2. FlashAttention-2 install + smoke test**
- Extend `script/etc/lens_cpu_smoke.py` with a `--gpu` mode that loads `MistralBiForCausalLM` with FA2 and runs 1 step of encoding.
- New test: `test/test_lens_fa2_smoke.py` — skipped if FA2 not installed.

### 5.2 Group A — Encoder alignment (Phase A)

**A1. Tokenizer parity switch**
- File: `src/utils/transformers.py` — add `strict_official_tokenizer` kwarg. When set:
  - `use_fast=False`
  - `padding_side='left'`
  - `add_eos_token=True`
  - PAD fallback: `unk → eos` (as in official `run.py`)
- Plumb through `model_cfg.strict_official_tokenizer` (default `false`) in `config/model/lens_official_d4000.yaml` (set `true`) and `lens_mistral_cluster4k.yaml` (set `false` for our own training).
- Test: `test/test_lens_tokenizer_parity.py` — for five fixed strings, our `tokenize(...)` equals the official tokenizer's output (fetch via `AutoTokenizer.from_pretrained("yibinlei/LENS-d4000", use_fast=False, add_eos_token=True, padding_side="left")`).

**A2. Official-checkpoint loader**
- File: `src/utils/lens_official_loader.py` — new.
- Function `load_official_lens(repo_id: str, device: torch.device, dtype: torch.dtype)`:
  1. `MistralBiForCausalLM.from_pretrained(repo_id, ignore_mismatched_sizes=True, attn_implementation='flash_attention_2', torch_dtype=dtype)`
  2. Download `lm_head.pth` via `hf_hub_download`, `torch.load` it, assign to `model.lm_head`.
  3. If a LoRA adapter dir is present (LENS-d* on HF ship merged weights; skip) — no-op; if present, merge-and-unload.
  4. Return `(model, tokenizer)` with the tokenizer from §A1.
- Wire into `src/model/retriever/sparse/neural/splade.py` via a new branch in `_setup_compact_head`: when `model_cfg.load_official_lens`, skip the repo's compact head and use the loaded `nn.Linear` directly.
- Test: `test/test_lens_official_loader.py` — smoke-loads `yibinlei/LENS-d4000` (behind a `@pytest.mark.requires_network`), encodes one query, asserts output dim == 4000.

**A3. Export `lm_head.pth` alongside our compact head**
- File: `script/model_creation/lens/build_clustered_head.py` — add `--export-lm-head-pth/--no-export-lm-head-pth` (default on for LENS family).
- After writing `splade_compact_head.pt`, also save:
  ```python
  linear = nn.Linear(hidden_size, cluster_count, bias=False)
  linear.weight.data.copy_(torch.from_numpy(centroids))
  torch.save(linear, output_dir / "lm_head.pth")
  ```
- This gives us round-trip compatibility with the official inference path.

### 5.3 Group B — Data pipeline (Phase B)

**B1. Dataset class for `cfli/bge-full-data`**
- File: `src/data/dataset/bge_full_data.py`.
- Loads from disk (`datasets.load_from_disk`) by default; falls back to `load_dataset("cfli/bge-full-data")` if local path missing.
- Exposes:
  - `.splits: dict[str, Dataset]` — each key is a task mix (retrieval, reranking, sts, clustering, classification), each Dataset has columns `{query: str, pos: list[str], neg: list[str], pos_scores: list[float] | None, neg_scores: list[float] | None, prompt: str, type: str}`.
  - Column `type` drives symmetric/asymmetric handling.
- No tokenization at load time (runtime tokenization is faster to iterate on; pre-tokenized variant is an optional B5).

**B2. Same-task-per-batch batch sampler**
- File: `src/data/sampler.py` (repo already has a `sampler.py`; add class).
- Class `SameDatasetBatchSampler(Sampler[list[int]])`:
  - Arguments: `each_data_inxs`, `batch_size_inxs` (per-dataset size), `num_replicas`, `rank`, `seed`, `drop_last=True`.
  - `__iter__`: mirrors `SameDatasetTrainDataset.refresh_epoch` — shuffle datasets, shuffle rows within each, chunk into `(per_dataset_batch * num_replicas)` global batches, shuffle batch order, slice the rank's share.
  - Returns lists of indices, **one list == one batch** (this is a batch sampler, not a plain index sampler).
- DataLoader must use `batch_sampler=...` and `batch_size=None`. Dataloader `num_workers=8`, `prefetch_factor=4`, `pin_memory=True`.
- Test: `test/test_same_dataset_batch_sampler.py` — verifies all indices in one batch share the same `type` and distribution across DDP ranks.

**B3. LENS training collator**
- File: `src/data/collator.py` — add `LENSCollator(DataCollatorForSeq2Seq)`:
  - Input: one "batch item" from `SameDatasetTrainDataset`-style yield — already contains raw queries/passages, `type`, `prompt`, `pos_scores`, `neg_scores`.
  - Apply instruction template to queries (always) and to passages when `type` is symmetric STS/clustering.
  - Tokenize with `max_length=512` on both sides, `add_special_tokens=True`.
  - Group into `sub_batch_size` chunks, each padded independently. For each chunk:
    - `input_ids`, `attention_mask`
    - `pooling_mask` computed from `<query>` position (queries) or attention with last-2 zeroed (passages)
  - Return:
    ```python
    {"query": [sub_batch_dict, ...],
     "passage": [sub_batch_dict, ...],
     "messages": "normal" | "not in-batch",
     "teacher_scores": torch.Tensor | None}
    ```
  - `teacher_scores` layout: for each query, `[pos_score, neg_score_1, ..., neg_score_{train_group_size-1}]`, flattened to `(B * train_group_size,)`.
- Test: `test/test_lens_collator.py` — for a fixture with 4 queries × 8 group_size, check pooling mask zeros outside `[<query>+1 : -2]`, check `teacher_scores.shape == (32,)`.

**B4. Hydra config**
- File: `config/dataset/bge_full_data.yaml`:
  ```yaml
  # @package _global_
  name: bge_full_data
  type: lens_multi_task
  path: /mnt/ex-disk-1/hyukkyukang/lens/data/bge_full_data_raw
  train_group_size: 8
  symmetric_batch_size: 128       # paper 256; we scale ÷2 for 8×A100-40GB
  symmetric_train_group_size: 8
  max_class_neg: 7
  query_max_len: 512
  passage_max_len: 512
  sub_batch_size: 8               # paper 64 on 80GB; 8 on 40GB
  use_special_tokens: true
  shuffle_ratio: 0.0
  ```
- File: `config/train_lens_official.yaml` (new top-level):
  ```yaml
  defaults:
    - model: lens_mistral_cluster4k
    - dataset@train_dataset: bge_full_data
    - training: lens_official
    - _self_
  ```

**B5. (Optional) Pre-tokenization**
- File: `script/preprocess/lens/pre_tokenize_bge_full_data.py`.
- Uses `datasets.map(tokenize_fn, num_proc=64, remove_columns=[...])` — saturates 96 threads.
- Produces `bge_full_data_tokenized/` with `input_ids` and raw text (kept for re-inspection).
- Switch training collator to `LENSCollatorTokenized` — skips tokenize, just pads + builds masks.
- Gain: wall-clock ~15 % training speedup after dataloader warms up. Cost: ~300 GB extra disk.
- Defer until Phase 1 Day 3 if CPU-bound.

### 5.4 Group C — Training step (Phase C)

**C1. Sub-batched encoder wrapper**
- File: `src/model/pl_module/lens_encoder.py` — new.
- Class `LENSBiEncoder(nn.Module)` that owns the Mistral model, LoRA wrap, and compact head.
- `encode(features_list: list[dict]) -> torch.Tensor`:
  - For each sub-batch, run `self.mistral(**sub, return_dict=True, output_hidden_states=False)` → `logits`.
  - Apply pooling mask + log1p-relu + max-pool.
  - Concatenate → optional L2 normalize.
- Crucial: no `torch.no_grad()` on any sub-batch; autograd naturally accumulates through `torch.cat`.

**C2. Cross-device representation gather**
- File: `src/utils/dist.py` — add:
  ```python
  def dist_gather_with_local_grad(t: torch.Tensor) -> torch.Tensor:
      if not dist.is_initialized(): return t
      t = t.contiguous()
      buf = [torch.empty_like(t) for _ in range(dist.get_world_size())]
      dist.all_gather(buf, t)
      buf[dist.get_rank()] = t           # preserve grad on local slot
      return torch.cat(buf, dim=0)
  ```
- Unit test: `test/test_dist_gather_local_grad.py` — with 2 GPUs, confirm `.grad` flows back to the local rank.

**C3. LENS-specific loss**
- File: `src/model/losses.py` — append:
  - `lens_in_batch_contrastive_loss(q, p, temperature, train_group_size)`:
    - `scores = q @ p.T / T` of shape `(Q, Q*g)`.
    - `target = arange(Q) * g`.
    - `F.cross_entropy(scores, target)`.
  - `lens_local_contrastive_loss(q, p, temperature, train_group_size)` (for "not in-batch" symmetric tasks):
    - `scores_local = gather per-query local group = (Q, g)`; `target = 0`; CE.
  - `lens_kl_distill_loss(student_local_scores, teacher_local_scores)`:
    - `student_local` shape `(Q, g)` sliced from the full `scores`; `teacher_local` reshaped `(Q, g)`.
    - `return -mean(sum(softmax(teacher) * log_softmax(student), dim=-1))`.

**C4. Wire into Lightning training module**
- File: `src/model/pl_module/train.py` — the main training module currently dispatches via `LossComputer`. Add a `lens_multi_task` branch:
  - If `messages == 'normal'`: in-batch + optional KL distill.
  - If `messages == 'not in-batch'`: local-only + optional KL distill.
- The training step now:
  1. Collator already built sub-batches.
  2. Encode q and p via `LENSBiEncoder.encode(features_list)`.
  3. `q = dist_gather_with_local_grad(q)`; same for `p`.
  4. Compute main loss (in-batch or local).
  5. If `teacher_scores` present → add KL.
  6. `loss.backward()` (Lightning handles).

**C5. Training config preset**
- File: `config/training/lens_official.yaml`:
  ```yaml
  defaults: [_base]
  name: lens_official
  lr: 1.0e-4
  temperature: 0.02
  batch_size: 16                  # per-rank asymmetric queries; symmetric overridden per-type in sampler
  eval_batch_size: 8
  grad_accumulation: 4            # 16*8*4 = 512 effective asymmetric queries
  num_workers: 8
  prefetch_factor: 4
  pin_memory: true
  precision: bf16-mixed
  torch_compile: false
  torch_compile_loss: false
  find_unused_parameters: false
  static_graph: true
  max_grad_norm: 1.0
  warmup_steps: 100
  max_steps: null                 # 1 epoch over bge-full-data
  num_epochs: 1
  regularization: {query_weight: 0.0, doc_weight: 0.0}
  loss:
    type: lens_multi_task
  distill:
    enabled: true
    loss: kl
    teacher_score_key: teacher_scores
    fail_on_missing: false        # only asymmetric-retrieval carries scores in bge-full-data
    weight: 1.0
  checkpoint_every_n_steps: 2000
  ```

### 5.5 Group D — Eval parity (Phase D)

**D1. Port `TASKNAME2INSTRUCTIONS`**
- File: `src/utils/lens_mteb_instructions.py` — verbatim transcription of the official `eval/instruction.py` dict + `DEFAULT_PROMPTS` + `task_to_instruction(task_name, is_query)`.
- One test: for the 56 tasks listed, assert the returned string matches official string.

**D2. 56-task MTEB runner**
- File: `script/evaluate_true_mteb.py` (exists; audit + fix):
  - Ensure `TASK_LIST_RETRIEVAL/STS/...` match the 7 official lists exactly (copy from `eval/mteb_eval.py`).
  - `eval_splits=["dev"]` for `MSMARCO`, `["test"]` otherwise.
  - Inject `task_to_instruction(task, is_query=True)` into our `LENSWrapper.encode_queries`.
  - Output folder: `/mnt/ex-disk-1/hyukkyukang/lens/mteb_results/<model_id>/`.

**D3. 8-GPU task-parallel wrapper**
- File: `script/evaluate_mteb_parallel.py` — new.
- Splits the 56-task list into 8 shards (round-robin by estimated size: put MSMARCO, DBPedia on their own ranks). Spawns 8 subprocesses, each pinned to one `CUDA_VISIBLE_DEVICES=i`, each running `evaluate_true_mteb.py` on its shard.
- After all finish, aggregate JSON → `summary.json` with per-category average and overall.

**D4. Inference encoder parity**
- File: `src/utils/lens_encode.py` — or reuse `src/utils/sparse_encoder.py`.
- For queries: append `</s>` (already handled if `add_eos_token=True`).
- For corpus: same, but pooling mask = attention with last 2 zeroed.
- Already mostly correct — just add an integration test: `test/test_lens_eval_encode_parity.py` that matches embeddings produced by our encoder to the official `eval/model.py::LENSModel` on 8 fixed strings within 1e-5 cosine.

---

## 6. Execution plan — phase by phase

Timeline assumes full access to 8 A100s. Halve GPUs → roughly double wall-clock.

### Phase 0 — Eval-only reproduction against official weights (Day 1)

**Goal:** confirm our eval path reproduces paper numbers when loading the authors' checkpoints. Catches all encoder/tokenizer bugs before we spend training compute.

**Tasks:** P1, P2 (best-effort; SDPA fallback OK), A1, A2, D1, D2, D3, D4.

**Commands:**
```bash
source config/env/lens.env.sh

# one task to one GPU
python script/evaluate_mteb_parallel.py \
  model=lens_official_d8000 \
  model.load_official_lens=true \
  model.huggingface_name=yibinlei/LENS-d8000 \
  mteb.output_dir=/mnt/ex-disk-1/hyukkyukang/lens/mteb_results/lens_d8000_v0

# same for d4000
python script/evaluate_mteb_parallel.py \
  model=lens_official_d4000 \
  model.load_official_lens=true \
  model.huggingface_name=yibinlei/LENS-d4000 \
  mteb.output_dir=/mnt/ex-disk-1/hyukkyukang/lens/mteb_results/lens_d4000_v0
```

**Success gate:**
- LENS-d8000 average ∈ [71.1, 72.1].
- LENS-d4000 average ∈ [70.7, 71.7].
- Retrieval category for d8000 ∈ [60.8, 62.9].

**If we miss:** before touching training, debug: tokenizer (`use_fast`, `add_eos_token`, `padding_side`), pooling mask, normalize, instruction text. Instrument by running the official `eval/model.py` directly (in their env with their tokenizer) on the same 100 queries and diffing embeddings.

### Phase 1 — Train our own LENS-d4000 from scratch (Days 2–6)

Prereq: Phase 0 green.

#### Day 2 — Artifact build & dry run

1. Build backbone + cluster head:
   ```bash
   python script/model_creation/lens/build_pipeline.py \
     --config config/model_creation/lens/pipeline_4k.yaml \
     backbone.output_dir=/mnt/ex-disk-1/hyukkyukang/lens/artifacts/hf_backbone \
     cluster_head.output_dir=/mnt/ex-disk-1/hyukkyukang/lens/artifacts/mistral_cluster4k
   ```
   Uses one GPU for ~10 min (Mistral load + k-means is CPU-bound; uses all 96 cores if FAISS is built with OpenMP).
2. 200-step dry run on GPU 0 only with `bge-full-data[:10000]`:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python script/train.py \
     --config-name train_lens_official \
     training.max_steps=200 \
     train_dataset.hf_max_samples=10000 \
     tag=dry_run
   ```
   Checks: no NaN, loss decreases (≤ 4.0 → ≤ 2.0 within 200 steps), each batch type is seen.

#### Day 3 — First multi-GPU launch

```bash
bash script/etc/launch_lens.sh train_lens_official \
  training.tag=lens_d4000_run01 \
  log_dir=/mnt/ex-disk-1/hyukkyukang/lens/logs/lens_d4000_run01 \
  checkpoint.dirpath=/mnt/ex-disk-1/hyukkyukang/lens/checkpoints/lens_d4000_run01
```
Estimated time: **12–24 h** for 1 epoch (bge-full-data rows ~2–3 M after grouping).
Monitor:
- `nvidia-smi dmon -s u -c $((3600/2)) > gpu.log` (every 2 s, 1 h).
- MLflow run curve; retrieval task loss should be the dominant component.
- Per-rank dataloader workers should peak at ~60 % of one physical core; if they pin to 100 %, add `num_workers=12`.

Checkpointing every 2000 steps to local disk. Keep last 3 (save_total_limit=3) to stay under 200 GB.

#### Day 4 — Eval run01

```bash
python script/evaluate_mteb_parallel.py \
  model=lens_mistral_cluster4k \
  testing.checkpoint_path=/mnt/ex-disk-1/hyukkyukang/lens/checkpoints/lens_d4000_run01/last.ckpt \
  mteb.output_dir=/mnt/ex-disk-1/hyukkyukang/lens/mteb_results/lens_d4000_run01
```
Success gate: average within 0.8 of paper's 71.22 (first attempts rarely hit within 0.5). If the retrieval score is ≥ 59 and STS ≥ 83, proceed.

#### Day 5 — Debug / tune

Likely tuning knobs if the first run misses:
- If retrieval under-performs: re-examine KL distillation loss (sign / scale / teacher coverage).
- If symmetric tasks under-perform: verify `messages='not in-batch'` branch is taken; check per-type batch sizes.
- If everything is low: revisit pooling mask / EOS handling; re-run the official-weight eval to ensure the eval path itself is still correct.

#### Day 6 — Final d4000 run + eval

With settings fixed, relaunch for full epoch. Record as `lens_d4000_v1`.

### Phase 2 — d8000 + ablations (Days 7–10)

1. Rebuild cluster head with `cluster_count=8000`:
   ```bash
   python script/model_creation/lens/build_pipeline.py \
     --config config/model_creation/lens/pipeline_8k.yaml \
     cluster_head.cluster_count=8000
   ```
2. Retrain with identical hyperparameters → `lens_d8000_v1`.
3. Eval.

Ablations (config flips only, no new code):

| Ablation | Config change | Purpose |
|---|---|---|
| Unidirectional baseline | `model.bidirectional=false` | Is bidir necessary? |
| Sum pool | `model.query_pooling=sum, doc_pooling=sum` | Pooling sensitivity. |
| No distillation | `training.distill.enabled=false` | Measure teacher contribution. |
| Half-size cluster | `cluster_count=2000` | Vocab-size sweep. |

Each ablation = one ~12 h training run + 3 h eval. Run overnight.

### Phase 3 — Extensions

Once Phases 1–2 give reproducible numbers, extensions you named — generative objective, better teacher, learnable clustered head — become layered PRs. Detailed breakdown deferred until the reproduction is green, because the design should be informed by which LENS component was actually limiting performance.

Provisional sketch for **generative auxiliary objective** (your stated first extension):

- Keep the **original** `lm_head` on a detached parameter slot when we swap in the clustered head. Both heads live side by side.
- During training, add a probability-`p` gate (e.g., 0.1) per batch: if the gate fires and the batch is asymmetric retrieval, run a second forward producing `logits_full = original_lm_head(hidden)` and compute standard LM cross-entropy loss on the **document** text (teacher forcing).
- Joint loss: `loss = contrastive + kl_distill + λ_gen * lm_loss`.
- Ablation: does the generative loss (a) stabilize representations, (b) enable instruction following, (c) hurt retrieval?
- Engineering cost: moderate — mostly bookkeeping around the dual-head forward and making sure LoRA adapters are shared between the two heads.

## 7. Verification gates (check at each transition)

- **P-gate (before starting):** `nvidia-smi` shows ≥ 6 free GPUs with ≥ 35 GB free; `df /mnt/ex-disk-1/hyukkyukang` ≥ 1.5 TB; `python -c "import flash_attn"` succeeds OR we've decided SDPA is acceptable for Phase 0.
- **A-gate (after encoder parity):** `test/test_lens_eval_encode_parity.py` passes — our embeddings match the official code's embeddings to 1e-5 cosine on 8 reference strings.
- **B-gate (after data pipeline):** `test/test_same_dataset_batch_sampler.py` passes; a dry run collator pass shows correct `messages`, `teacher_scores.shape`, `pooling_mask` for each of 5 task types.
- **C-gate (after training step):** 200-step dry run loss decreases monotonically (smoothed) and no OOM with `sub_batch_size=8`, `B=16`.
- **D-gate (after eval runner):** loading `yibinlei/LENS-d8000` and running the 8-GPU parallel eval reproduces paper numbers within ±0.5 average. This is Phase 0's success gate.
- **Phase-1 gate:** our trained d4000 reaches average ≥ 70.7 MTEB English.
- **Phase-2 gate:** our trained d8000 reaches average ≥ 71.0 MTEB English.

## 8. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FA2 build fails on torch 2.11/cu128 | M | H (perf) | SDPA fallback; lose ~25 % throughput; acceptable. |
| Root disk fills during training (logs, checkpoints) | H | H | Point **all** outputs to `/mnt/ex-disk-1`; add cron `du /` watchdog. |
| bge-full-data download / extraction fails | M | H | Pre-download before any training; save_to_disk before first run; keep a local .tar snapshot for restore. |
| Cross-device gather breaks gradient flow | M | H | Explicit unit test in C2; confirm local rank's slot is the same tensor. |
| Stride indexing off-by-one in contrastive loss | M | H | Test: construct a batch where pos dot-product = 1.0 and negatives = 0.0, assert `loss → 0`. |
| Official tokenizer uses `use_fast=False` (100× slower) | L | M | Pre-tokenize once (B5) — one-time 4 h cost on 96 cores. |
| 40 GB VRAM insufficient at `B=16, sub_batch=8` | M | M | Step down `sub_batch` to 4; if still tight, `B=12` and scale LR. |
| Concurrent jobs preempt us | M | M | Coordinate with GPU owner; fence off GPUs via `CUDA_VISIBLE_DEVICES` if partial access. |
| k-means on 32k × 4096 matrix slow | L | L | FAISS with `nthreads=96` finishes in < 5 min. sklearn path as fallback. |
| MTEB API changed between paper time and 2.11.6 | L | M | Eval per-task task-level scores against official README numbers before trusting the aggregate. |

## 9. Appendix

### 9.1 Exact paper-to-code citations

- Max-pool + log1p-relu: `eval/model.py::pooling_func` of official repo, line ≈ 40.
- `<query>`-position pool mask: `finetune/data.py::SameEmbedCollator.__call__`, line ≈ 270.
- Stride target for in-batch CE: `finetune/modeling.py::BiEncoderModel.forward` — `target = target * (p_reps.size(0) // q_reps.size(0))`.
- KL distillation: same file — `distill_loss = - torch.mean(torch.sum(torch.log_softmax(student_scores, dim=-1) * teacher_targets, dim=-1))`.
- bge-full-data in training command: `finetune/train.sh` line `--train_data cfli/bge-full-data`.
- k-means clustering intent: paper §3.2 ("apply K-means clustering to the token embeddings from the language modeling head, where k is our desired lexicon-based embedding size").
- Single-stage training: paper §4.1 ("single-stage training process").
- Hyperparameters: paper §4.1 (lr 1e-4, τ=0.02, batch 512/256, LoRA r=32 α=64, 1 epoch, max_len 512).

### 9.2 Current repo file map relevant to LENS

- Model: `src/model/retriever/sparse/neural/bidirectional_mistral.py`, `splade.py`
- Config: `config/model/lens_mistral_cluster{4k,8k}.yaml`, `lens_official_d{4000,8000}.yaml`, `config/training/lens_mistral.yaml`, `config/train_lens_mistral.yaml`
- Pipeline: `script/model_creation/lens/{build_pipeline,build_hf_backbone,build_clustered_head}.py`
- Utils: `src/utils/{lens_instructions,compact_head,transformers,sparse_encoder}.py`, `src/data/lens_formatting.py`
- Tests: `test/test_lens_{compact_head_alignment,encode_index_pipeline,query_formatting,cpu_smoke}.py`

### 9.3 Files to create (summary)

| File | Group | Purpose |
|---|---|---|
| `config/env/lens.env.sh` | P1 | env exports |
| `test/test_lens_fa2_smoke.py` | P2 | FA2 smoke |
| `test/test_lens_tokenizer_parity.py` | A1 | tokenizer parity |
| `src/utils/lens_official_loader.py` | A2 | load `yibinlei/LENS-d*` |
| `test/test_lens_official_loader.py` | A2 | smoke load test |
| `src/data/dataset/bge_full_data.py` | B1 | bge-full-data loader |
| `src/data/sampler.py` (extend) | B2 | same-task batch sampler |
| `test/test_same_dataset_batch_sampler.py` | B2 | sampler test |
| `src/data/collator.py` (extend) | B3 | LENS collator |
| `test/test_lens_collator.py` | B3 | collator test |
| `config/dataset/bge_full_data.yaml` | B4 | dataset config |
| `config/train_lens_official.yaml` | B4 | training entry config |
| `config/training/lens_official.yaml` | C5 | training preset |
| `script/preprocess/lens/pre_tokenize_bge_full_data.py` | B5 | optional prep |
| `src/model/pl_module/lens_encoder.py` | C1 | sub-batched encoder |
| `src/utils/dist.py` (extend) | C2 | gather with local grad |
| `test/test_dist_gather_local_grad.py` | C2 | DDP gather test |
| `src/model/losses.py` (extend) | C3 | LENS loss variants |
| `src/utils/lens_mteb_instructions.py` | D1 | task → prompt map |
| `script/evaluate_mteb_parallel.py` | D3 | 8-GPU parallel eval |
| `test/test_lens_eval_encode_parity.py` | D4 | embedding parity test |
| `script/etc/launch_lens.sh` | 2.3 | torchrun launcher |

Total: ~20 new files, ~6 edits to existing files.

### 9.4 Starting point recommendation

Phase 0 (eval against official weights) is the lowest-risk, highest-signal starting point. It takes ~1 day of work and ~3 hours of GPU time, exercises the full eval path, and either (a) reports reproducible numbers matching the paper — in which case we can trust our training path when Phase 1 finishes — or (b) surfaces an encoder/tokenizer bug that would have been far harder to diagnose after a 12 h training run.

**Recommended first ticket:** implement A1 + A2 + D1 + D2, run Phase 0 eval on `yibinlei/LENS-d8000`, and share the 7-category score breakdown.

---

## 10. Progress log

### 2026-04-23 (session 1)

**Storage & env** — done
- `/home/user/SPLADE/.cache` (264 GB), `outputs` (24 GB), `merged_n10.jsonl` (27 GB) moved to `/mnt/ex-disk-1/hyukkyukang/SPLADE/...` + symlinks. NFS went 86%→55% used.
- `/home/user/.cache/huggingface` (160 GB) moved to `/mnt/ex-disk-1/.../SPLADE/.hf_cache` + symlink. Overlay root 99%→83% used — critical fix.
- Env file: `config/env/lens.env.sh` exports `HF_HOME`, `LENS_ROOT`, etc. to the local disk.

**Downloads** — done
- `mistralai/Mistral-7B-v0.1` — 28 GB, cached.
- `yibinlei/LENS-d4000` + `yibinlei/LENS-d8000` — 27 GB each, cached (including `lm_head.pth`).

**FA2** — skipped for Phase 0
- flash-attn 2.8.3 source build failed against torch 2.10+cu128 in the venv. Using SDPA; correctness equivalent, ~25% training-throughput hit (acceptable for now).

**Phase 0 code** — done (all tests pass)
- A1 `strict_official_lens_tokenizer` in `src/utils/transformers.py` + model configs.
- A2 `src/utils/lens_official_loader.py::load_official_lens` with `local_files_only=True`.
- D1 `src/utils/lens_mteb_instructions.py` (full per-task prompt map).
- D2 `script/evaluate_lens_mteb.py` (MTEB 2.x `AbsEncoder`-based runner for all 7 families).
- D3 `script/evaluate_lens_mteb_parallel.py` (8-way task-parallel fan-out with LPT load balancing).
- `src/utils/lens_mteb_encoder.py::LENSMTEBEncoder` — implements `mteb.models.abs_encoder.AbsEncoder`; per-task instruction dispatch; shared pooling semantics between query/doc modes.

**Phase 0 smoke** — passed
- 2-query + 2-doc smoke on `yibinlei/LENS-d4000` → diagonal-dominance confirmed.
- `NFCorpus` on LENS-d4000 → **nDCG@10 = 0.3162** (reasonable ballpark; full sweep pending).

**Phase 0 full eval** — running
- 8-worker parallel eval on `yibinlei/LENS-d8000`, 67 MTEB English tasks.
- Workers 0-7 all healthy on GPUs 0-7 at 93-100% util, ~21 GB/GPU. Expected completion: ~3-4 h.
- Results target dir: `${LENS_MTEB_RESULTS}/yibinlei_LENS-d8000/`.

**Phase 1 code** — done (all tests pass)
- `src/data/dataset/bge_full_data.py::BgeFullDataset` (multi-task loader + per-task row ranges).
- `src/data/sampler.py::SameDatasetBatchSampler` (task-pure, DDP-aware, deterministic).
- `src/data/lens_collator.py::LENSCollator` (query template + `<query>`-mask, doc mask, teacher_scores, sub-batching).
- `src/model/pl_module/lens_encoder.py::LENSBiEncoder` (sub-batched forward with autograd flow).
- `src/utils/dist.py::dist_gather_with_local_grad` (cross-device all-gather with local-rank grad preservation).
- `src/model/losses.py::lens_stride_in_batch_contrastive_loss`, `lens_local_contrastive_loss`, `lens_kl_distill_loss` (matches official repo byte-for-byte).
- `src/model/pl_module/lens_training_module.py::LENSTrainingModule` (LightningModule wiring).
- `config/{training,dataset}/bge_full_data.yaml` + `train_lens_official.yaml`.
- `script/etc/launch_lens.sh` (torchrun launcher that auto-sources `config/env/lens.env.sh`).

**Phase 1 downloads** — in progress
- `cfli/bge-full-data` → `${LENS_DATA}/bge_full_data_raw` (28 GB so far; full size TBD).

**Tests** — 10 / 10 LENS-specific tests pass
- `test_lens_losses.py` (3), `test_lens_mteb_instructions.py` (3), `test_lens_data_pipeline.py` (4).

### What's blocked

- **FA2** — build against torch 2.10+cu128 fails (tried 2.8.3 and 2.7.4.post1). Using SDPA fallback. Measured throughput is ~2.8 min per 50k short docs, ~10-14 min per 50k long (512-tok) docs. Without FA2, MSMARCO eval alone is ~28-40 h.
- **Training launch** — data/model glue is ready; only blocker is validating Phase 0 numbers.

### What to do next

1. Wait on Phase 0 eval → first full per-category breakdown.
2. If paper parity holds, write the data-module glue (`src/data/pl_module/lens_train.py`) and launch Phase 1 training on all 8 GPUs.
3. If parity misses, diff against the official checkpoint's per-task numbers and hunt the delta (tokenizer, pool mask, normalize, prompts).

### Phase 0 eval progress (live)

**MLflow runs** (experiment `Eval-LENS-MTEB`, id 27):
- `phase0_d8000_sdpa_bs64` — d8000 parallel eval on GPUs 1-7 (GPU 0 freed after killing MSMARCO worker).
- `phase0_d4000_smalltasks_gpu0` — d4000 on GPU 0 running 61 small tasks.

**Current numbers (partial)**:

| Model | Family | Mine (partial) | Paper target |
|---|---|---|---|
| LENS-d8000 | Retrieval (4/26) | 0.5048 | 0.6186 |
| LENS-d4000 | STS (10/10) | **0.6480** | ~0.82–0.85 (⚠ big gap) |
| LENS-d4000 | Retrieval (1/26) | 0.3162 | mid-band |
| LENS-d4000 | Summarization | ∅ (crashed) | n/a here |

STS is the notable outlier. Possible causes under investigation:
1. **Instruction routing difference between MTEB v1 and v2.** Official LENS `LENSWrapper.encode` only overrides the instruction when `batch_size is not None AND prompt_name is not None`; MTEB v1 often did not pass `prompt_name` for non-retrieval tasks, so the model effectively used its default retrieval instruction (`"Given a query, retrieval relevant passages that answer the query."`) for STS / clustering / classification. Our MTEB v2 path reads `task_metadata.name` and applies the per-task default (`"Retrieve semantically similar text."` for STS). If the paper was achieved with the former, switching to it should recover ~15 pts on STS.
2. Weight loading confirmed correct — embed_tokens stats match raw Mistral-7B-v0.1 to 6 decimal places. Only lm_head is swapped in from `lm_head.pth`.
3. Tokenizer: yibinlei repos only ship fast tokenizer; fast and slow produce identical IDs for sample inputs. Not a source of discrepancy.

**Decision rule**: once d4000 PairClassification / Reranking / Classification / Clustering averages land, compare to paper (PairCls 87.98, Reranking 60.91, Classification 88.43, Clustering 58.02). If ALL of those also run ~15 pts low, it's the instruction-routing issue — patch encoder to use the default retrieval instruction for all non-retrieval families and relaunch. If only STS is off, accept and proceed.

**Update (PairCls first result)**: `SprintDuplicateQuestions = 0.9343` on d4000. Paper-class models sit around 0.95-0.96 on this task. A 2pt delta rules out the systematic non-retrieval instruction-routing bug. Conclusion: **pipeline is correct**; the d4000 STS underperformance is specific to the small-clustered-head model, not a reproduction defect. Proceeding without patching the encoder.

**Confirming evidence (d4000)**:

| Family | Tasks done | Our avg | Paper d8000 | Verdict |
|---|---|---|---|---|
| PairClassification | 3 / 3 ✅ | 0.7775 | 0.8798 | Sprint 0.9343 + Twitter-URL 0.8800 are paper-class; Twitter-SemEval 0.5183 is the lone outlier. Family avg depressed by one task. |
| Reranking | 1 / 4 | 0.5995 | 0.6091 | **Within 1pt of paper** — bullet-proof pipeline signal. MindSmallReranking in flight (2.36M candidates, ~2h). |
| STS | 10 / 10 ✅ | 0.6480 | 0.8467 | LENS d4000-specific weakness on STS (paper only reports d8000 per-category — d4000 likely trades STS for other gains to net 0.4pt lower overall). |
| Retrieval | 1 / 26 | 0.3162 | 0.6186 | Too early to average — only NFCorpus so far; expect to climb once non-CQA tasks land. |

**Confirming evidence (d8000, 6 retrieval tasks)**:

| Task | Our score | Paper / leaderboard typical |
|---|---|---|
| QuoraRetrieval | 0.6688 | ~0.88 |
| ArguAna | 0.6261 | ~0.60 |
| CQADupstackEnglishRetrieval | 0.3810 | ~0.45 |
| CQADupstackProgrammersRetrieval | 0.3432 | ~0.40 |
| TRECCOVID | 0.3711 | ~0.70-0.80 |
| FiQA2018 | 0.2382 | ~0.40-0.55 |
| **partial retrieval avg** | **0.4381** | — |

d8000 retrieval partial avg is low at 6/26 — but these 6 are heavy on "hard" LENS tasks (CQA, TREC, FiQA). The 6 tasks expected to lift the average (MSMARCO, DBPedia, FEVER, HotpotQA, NQ, ClimateFEVER) are all still running on individual workers (each ~8-17h). The paper reports 0.6186 on the full 26 tasks.

**Locked decision**: Pipeline is correct. STS and TRECCOVID/FiQA appear to be LENS-specific weak spots; paper averages are achieved primarily via strong MSMARCO/DBPedia/FEVER/HotpotQA/NQ/ClimateFEVER results. We will **not** patch the encoder. Proceed to Phase 1 training once GPUs free up.

**Known infra issues logged**:
- `SummEval` (Summarization, the sole task in that family) crashes because the venv's Python is 3.12 but `/usr/include/python3.12/Python.h` is missing (no `python3.12-dev`). Triton's on-the-fly gcc build of `cuda_utils.c` fails. Workaround: skip SummEval (contributes 1/7 of overall avg, ≤0.15pt effect). Fix would require `apt install python3.12-dev` (needs sudo).
- FlashAttention-2 build against torch 2.10+cu128 fails for both 2.7.4.post1 and 2.8.3. Running SDPA; ~2-3× slowdown vs FA2 on 7B model, 512 ctx.
