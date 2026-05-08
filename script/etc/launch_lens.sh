#!/usr/bin/env bash
# torchrun launcher for LENS training. Sources the env file so HF/torch caches
# live on /mnt/ex-disk-1 rather than the root overlay.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/config/env/lens.env.sh"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONFAULTHANDLER=1

CONFIG_NAME="${1:-train_lens_official}"
shift || true

VENV_PY="${VENV_PY:-$REPO_ROOT/.venv312eval/bin/python3}"
if [ ! -x "$VENV_PY" ]; then
  VENV_PY="$(command -v python3)"
fi

NPROC="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
if [ "$NPROC" -le 0 ]; then NPROC=1; fi

cd "$REPO_ROOT"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-script/train_lens.py}"
exec "$VENV_PY" -m torch.distributed.run \
  --standalone \
  --nnodes 1 \
  --nproc_per_node "$NPROC" \
  "$TRAIN_SCRIPT" \
  --config-name "$CONFIG_NAME" \
  "$@"
