#!/usr/bin/env bash
# Phase-2 LENS training launcher.
#
# Defaults reflect the empirical sweep on top of phase 1
# (phase1_d4000_LR1e5_20260429_1005) using PyTorchProfiler traces:
#
#   sub_batch_size  | avg step ms | wall vs phase 1
#   ---------------- + ----------- + ---------------
#   2 (phase 1)     |     879     |     baseline
#   3               |     838     |   -5%
#   4               |     750     |  -15%
#   8               |     708     |  -19%   <- sweet spot, ships by default
#   16              |     735     |  -16%   (chunks too big, GEMM regression)
#
# Why sub_batch_size matters: in eager mode each ATen op carries ~155us
# CPU dispatch overhead. Phase 1 launched ~62k aten::linear calls per
# 35s window (1.77k/s), saturating the launch rate while the GPU sat
# 85% idle. Bigger sub-batches mean fewer-but-bigger GEMM kernels --
# same GPU work, fewer Python/dispatch round-trips.
#
# Why we keep phase-1 outer batch geometry (BATCH=1, SYMM=8, TG=2):
# Saved gradient-checkpointing activations live until backward
# completes, scaling linearly with TOTAL docs encoded per step (regard-
# less of sub_batch_size). The "paper-faithful" outer batch (SYMM=64,
# TG=4, BATCH=4) increases that ~64x and OOMs even at sub=8 in current
# memory. Achieving paper-faithful effective batch in this environment
# requires either FSDP, or freeing the ~6GB orphaned CUDA context that
# is currently held per-GPU by a defunct PID (host-side issue, not
# fixable from inside this container).
#
# torch.compile is left OFF by default after empirical investigation.
# The Python.h toolchain blocker that originally made compile crash is
# fixed (apt-get install python3.12-dev), and LENS_COMPILE=1 now wires
# torch.compile cleanly with dynamic=True. But measured wall time was
# 977-1370 ms/step (vs 708 ms eager) because three structural
# incompatibilities create a recompile storm:
#   1. PEFT's enable_input_require_grads() forward hook flips the
#      embedding output's requires_grad mid-forward. Gradient
#      checkpointing recompute then sees a mismatched guard and
#      forces a recompile per step.
#   2. The HF @can_return_tuple wrapper in transformers/utils/generic.py
#      uses *args/**kwargs, which dynamo can't trace through cleanly.
#   3. max_padding=False yields per-sub-batch shape variation that
#      dynamic=True only partially mitigates.
# The trace showed ~30 root regions and one region recompiling 128
# times. Making compile a net win on this stack would require either:
#   - Replacing PEFT's hook with a compile-friendly equivalent, or
#   - Running with max_padding=True + mode="reduce-overhead" + CUDA
#     graphs for fully static shapes, or
#   - Monkey-patching HF to remove the can_return_tuple wrapper.
# All three are weeks-long projects, not flag flips. Set
# ENABLE_COMPILE=1 if the upstream stack changes and re-measure.
#
# Usage:
#   bash script/etc/launch_lens_phase2.sh                       # default tag
#   TAG=phase2_d4000_run01 bash script/etc/launch_lens_phase2.sh
#   ENABLE_COMPILE=1 bash script/etc/launch_lens_phase2.sh
#   MAX_STEPS=10000 bash script/etc/launch_lens_phase2.sh
#
# Try the "paper-faithful" outer batch (will OOM today; here for when
# memory is freed up):
#   BATCH_SIZE=4 GRAD_ACCUMULATION=4 \
#   SYMMETRIC_BATCH_SIZE=64 SYMMETRIC_TRAIN_GROUP_SIZE=4 \
#   TRAIN_GROUP_SIZE=4 \
#   bash script/etc/launch_lens_phase2.sh
#
# Diagnostic mode (PyTorch profiler, ~25-step trace -> ${log_dir}/profile/):
#   LENS_PROFILE=1 MAX_STEPS=50 bash script/etc/launch_lens_phase2.sh
#   # window controlled by LENS_PROFILE_WAIT/_WARMUP/_ACTIVE (defaults 1/4/20).

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

# --- batch geometry --------------------------------------------------------
# Outer batch matches phase 1 (proven memory-fit). The win lives in
# SUB_BATCH_SIZE=8 -- see header comment for the ablation.
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-8}"
SYMMETRIC_BATCH_SIZE="${SYMMETRIC_BATCH_SIZE:-8}"
SYMMETRIC_TRAIN_GROUP_SIZE="${SYMMETRIC_TRAIN_GROUP_SIZE:-2}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-2}"
SUB_BATCH_SIZE="${SUB_BATCH_SIZE:-8}"

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

# --- DeepSpeed ZeRO toggle -------------------------------------------------
# ENABLE_DEEPSPEED=1 switches the Lightning strategy to DeepSpeedStrategy.
# DEEPSPEED_STAGE selects ZeRO stage (1, 2, or 3; default 3).
# DEEPSPEED_OFFLOAD_OPTIMIZER=1 / DEEPSPEED_OFFLOAD_PARAMS=1 page state to
# CPU at a throughput cost (use only when GPU memory still doesn't fit).
# DEEPSPEED_CONFIG_PATH points at a custom JSON; null = Lightning defaults.
ENABLE_DEEPSPEED="${ENABLE_DEEPSPEED:-0}"
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-3}"
DEEPSPEED_OFFLOAD_OPTIMIZER="${DEEPSPEED_OFFLOAD_OPTIMIZER:-0}"
DEEPSPEED_OFFLOAD_PARAMS="${DEEPSPEED_OFFLOAD_PARAMS:-0}"
DEEPSPEED_CONFIG_PATH="${DEEPSPEED_CONFIG_PATH:-}"
DEEPSPEED_OVERRIDES=()
if [[ "${ENABLE_DEEPSPEED}" == "1" ]]; then
  # ``++key=value`` set-or-add; works whether ``training.deepspeed`` is
  # present in ``config/training/_base.yaml`` or not.
  DEEPSPEED_OVERRIDES+=(
    "++training.deepspeed.enabled=true"
    "++training.deepspeed.stage=${DEEPSPEED_STAGE}"
    "++training.deepspeed.offload_optimizer=$([[ "${DEEPSPEED_OFFLOAD_OPTIMIZER}" == "1" ]] && echo true || echo false)"
    "++training.deepspeed.offload_params=$([[ "${DEEPSPEED_OFFLOAD_PARAMS}" == "1" ]] && echo true || echo false)"
  )
  if [[ -n "${DEEPSPEED_CONFIG_PATH}" ]]; then
    DEEPSPEED_OVERRIDES+=("++training.deepspeed.config_path=${DEEPSPEED_CONFIG_PATH}")
  fi
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
echo "  pytorch_profiler          = $([[ "${LENS_PROFILE:-0}" == "1" ]] && echo on || echo off)"
if [[ "${ENABLE_DEEPSPEED}" == "1" ]]; then
  echo "  deepspeed                 = stage=${DEEPSPEED_STAGE}, offload_optim=${DEEPSPEED_OFFLOAD_OPTIMIZER}, offload_params=${DEEPSPEED_OFFLOAD_PARAMS}"
else
  echo "  deepspeed                 = off"
fi
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
  "${DEEPSPEED_OVERRIDES[@]}" \
  "$@"
