#!/usr/bin/env bash
# Phase-2 LENS training launcher.
#
# Phase 1 (phase1_d4000_LR1e5_20260429_1005) ran with batch sizes shrunk
# 4-32x below the paper-faithful lens_official defaults. Per-step Python /
# DDP orchestration dominated, so the 8 GPUs averaged ~30-50% util while
# all 8 rank processes pegged at 100% CPU. Phase 2 raises batches back
# toward the lens_official defaults (still capped for 8x A100-40GB) and
# halves grad_accumulation so each optimizer step does proportionally
# more GPU work per Python iteration.
#
# Compute-rate diff vs Phase 1:
#   - per-rank queries  : 1   ->  4
#   - sub_batch_size    : 2   ->  4    (activation chunk size)
#   - symmetric_batch   : 8   -> 64    (bge symmetric task batch)
#   - sym_train_group   : 2   ->  4    (positives/negatives per query)
#   - train_group_size  : 2   ->  4
#   - grad_accumulation : 8   ->  4    (back to lens_official default)
#
# Effective global batch grows ~8x; expected step throughput ~3-5x.
#
# torch.compile is left OFF by default because lens_official.yaml turned
# it off explicitly (compile + LoRA + MistralBi has been flaky in this
# tree). Set ENABLE_COMPILE=1 to flip it on; first run pays a 10-20 min
# inductor warmup, subsequent runs reuse $TRITON_CACHE_DIR.
#
# Usage:
#   bash script/etc/launch_lens_phase2.sh                       # default tag
#   TAG=phase2_d4000_run01 bash script/etc/launch_lens_phase2.sh
#   ENABLE_COMPILE=1 bash script/etc/launch_lens_phase2.sh
#   MAX_STEPS=10000 bash script/etc/launch_lens_phase2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- run identity ----------------------------------------------------------
DATE_TAG="$(date +%Y%m%d_%H%M)"
TAG="${TAG:-phase2_d4000_LR1e5_${DATE_TAG}}"

# --- training schedule -----------------------------------------------------
MAX_STEPS="${MAX_STEPS:-5000}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-500}"

# --- batch geometry (the actual fix) ---------------------------------------
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-4}"
SYMMETRIC_BATCH_SIZE="${SYMMETRIC_BATCH_SIZE:-64}"
SYMMETRIC_TRAIN_GROUP_SIZE="${SYMMETRIC_TRAIN_GROUP_SIZE:-4}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-4}"
SUB_BATCH_SIZE="${SUB_BATCH_SIZE:-4}"

# --- compile toggle --------------------------------------------------------
ENABLE_COMPILE="${ENABLE_COMPILE:-0}"
COMPILE_OVERRIDES=()
if [[ "${ENABLE_COMPILE}" == "1" ]]; then
  COMPILE_OVERRIDES+=(
    "training.torch_compile=true"
    "training.torch_compile_loss=true"
    "training.torch_compile_mode=default"
  )
fi

# --- model artifact (matches phase 1) --------------------------------------
MODEL_HF_NAME="${MODEL_HF_NAME:-/mnt/ex-disk-1/hyukkyukang/SPLADE/lens/artifacts/mistral_cluster4k}"

echo "=== LENS Phase 2 launch ==="
echo "  tag                       = ${TAG}"
echo "  max_steps                 = ${MAX_STEPS}"
echo "  lr                        = ${LR}"
echo "  batch_size (per rank)     = ${BATCH_SIZE}"
echo "  grad_accumulation         = ${GRAD_ACCUMULATION}"
echo "  symmetric_batch_size      = ${SYMMETRIC_BATCH_SIZE}"
echo "  symmetric_train_group     = ${SYMMETRIC_TRAIN_GROUP_SIZE}"
echo "  train_group_size          = ${TRAIN_GROUP_SIZE}"
echo "  sub_batch_size            = ${SUB_BATCH_SIZE}"
echo "  torch_compile             = $([[ "${ENABLE_COMPILE}" == "1" ]] && echo on || echo off)"
echo "==========================="

exec bash "${SCRIPT_DIR}/launch_lens.sh" train_lens_official \
  "tag=${TAG}" \
  "model.huggingface_name=${MODEL_HF_NAME}" \
  "training.max_steps=${MAX_STEPS}" \
  "training.lr=${LR}" \
  "training.warmup_steps=${WARMUP_STEPS}" \
  "training.batch_size=${BATCH_SIZE}" \
  "training.grad_accumulation=${GRAD_ACCUMULATION}" \
  "training.lens_multi_task.symmetric_batch_size=${SYMMETRIC_BATCH_SIZE}" \
  "training.lens_multi_task.symmetric_train_group_size=${SYMMETRIC_TRAIN_GROUP_SIZE}" \
  "training.lens_multi_task.train_group_size=${TRAIN_GROUP_SIZE}" \
  "training.lens_multi_task.sub_batch_size=${SUB_BATCH_SIZE}" \
  "+training.checkpoint_every_n_steps=${CHECKPOINT_EVERY_N_STEPS}" \
  "${COMPILE_OVERRIDES[@]}" \
  "$@"
