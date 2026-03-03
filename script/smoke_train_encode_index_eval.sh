#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TAG="${1:-smoke_$(date +%Y%m%d_%H%M%S)}"
TRAIN_MODEL_NAME="splade_v2_pp"
CHECKPOINT_PATH="log/train/${TRAIN_MODEL_NAME}/${TAG}/checkpoints/last.ckpt"
SMOKE_ENCODE_DIR="data/embed_smoke"
SMOKE_INDEX_DIR="data/index_smoke"

echo "[smoke] tag=${TAG}"
echo "[smoke] step=1 train"
python script/train.py \
  model=splade_v2_pp \
  training=splade_v2_pp \
  tag="${TAG}" \
  training.max_steps=2 \
  training.val_check_interval=1.0 \
  training.limit_val_batches=0.0 \
  training.use_cpu=true \
  training.num_devices=1 \
  training.strategy=auto \
  training.num_workers=0 \
  training.torch_compile=false \
  training.torch_compile_loss=false \
  training.mlflow.enabled=false \
  nanobeir.enabled=false \
  train_dataset.hf_max_samples=128 \
  val_dataset.hf_max_samples=64 \
  train_dataset.pretokenize.enabled=false \
  val_dataset.pretokenize.enabled=false

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[smoke] missing checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

echo "[smoke] step=2 encode"
python script/encode.py \
  model=splade_v2_pp \
  dataset=beir/nfcorpus \
  tag="${TAG}" \
  encoding.checkpoint_path="${CHECKPOINT_PATH}" \
  encoding.encode_dir="${SMOKE_ENCODE_DIR}" \
  encoding.index_tag="${TAG}" \
  encoding.use_cpu=true \
  encoding.num_devices=1 \
  encoding.strategy=auto \
  encoding.num_workers=0 \
  encoding.batch_size=32 \
  encoding.torch_compile=false \
  dataset.hf_max_samples=512

echo "[smoke] step=3 index"
python script/index.py \
  model=splade_v2_pp \
  dataset=beir/nfcorpus \
  tag="${TAG}" \
  encoding.encode_dir="${SMOKE_ENCODE_DIR}" \
  encoding.index_dir="${SMOKE_INDEX_DIR}" \
  encoding.index_tag="${TAG}"

echo "[smoke] step=4 retrieval evaluation"
python script/evaluation.py \
  model=splade_v2_pp \
  dataset=beir/nfcorpus \
  tag="${TAG}" \
  testing.checkpoint_path="${CHECKPOINT_PATH}" \
  encoding.index_dir="${SMOKE_INDEX_DIR}" \
  encoding.index_tag="${TAG}" \
  testing.use_cpu=true \
  testing.num_devices=1 \
  testing.strategy=auto

echo "[smoke] completed"
echo "[smoke] checkpoint=${CHECKPOINT_PATH}"
echo "[smoke] index_dir=${SMOKE_INDEX_DIR}/${TRAIN_MODEL_NAME}/${TAG}"
