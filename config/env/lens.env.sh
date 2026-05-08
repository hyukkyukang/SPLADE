# Source this file before any LENS training / eval invocation.
# All large artifacts live on /mnt/ex-disk-1/hyukkyukang — the only filesystem
# with enough free space (overlay root has <20G free; NFS has ~140G free).
export LENS_ROOT=/mnt/ex-disk-1/hyukkyukang/SPLADE

export HF_HOME="${LENS_ROOT}/hf"
export HF_HUB_CACHE="${LENS_ROOT}/hf/hub"
export HF_DATASETS_CACHE="${LENS_ROOT}/hf/datasets"
export TRANSFORMERS_CACHE="${LENS_ROOT}/hf/models"
export TORCH_HOME="${LENS_ROOT}/torch"
export TRITON_CACHE_DIR="${LENS_ROOT}/triton"
export MPLCONFIGDIR="${LENS_ROOT}/matplotlib"

# LENS-specific staging
export LENS_ARTIFACTS="${LENS_ROOT}/lens/artifacts"
export LENS_DATA="${LENS_ROOT}/lens/data"
export LENS_CHECKPOINTS="${LENS_ROOT}/lens/checkpoints"
export LENS_LOGS="${LENS_ROOT}/lens/logs"
export LENS_MTEB_RESULTS="${LENS_ROOT}/lens/mteb_results"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
    "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}" \
    "${LENS_ARTIFACTS}" "${LENS_DATA}" "${LENS_CHECKPOINTS}" \
    "${LENS_LOGS}" "${LENS_MTEB_RESULTS}"

# NCCL / DDP defaults tuned for this box (8×A100-SXM4 on NVSwitch, no IB).
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONFAULTHANDLER=1
