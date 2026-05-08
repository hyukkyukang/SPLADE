#!/usr/bin/env bash
set -euo pipefail

SHARD_COUNT="${SHARD_COUNT:-48}"
DOCUMENT_BATCH_SIZE="${DOCUMENT_BATCH_SIZE:-64}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DOCUMENT_ENCODING_MODE="${DOCUMENT_ENCODING_MODE:-split_fields_windowed}"
TERM_OUTPUT_MODE="${TERM_OUTPUT_MODE:-flat_terms}"

BASE_MODEL="${BASE_MODEL:-/home/user/.cache/huggingface/hub/models--naver--splade-v3/snapshots/fdfeceb91d7b9de7985b38addd3ba9f53a59a355}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/user/SPLADE/log/train/splade_v3_naver/patent_train_8gpu_bs15_ga2_ddpsafe_epochval/checkpoints/stepstep=6290-val_MRR_10val_MRR_10=0.9777.ckpt}"
CORPUS_PATH="${CORPUS_PATH:-/home/user/SPLADE/.cache/hf/patent-us-corpus}"

RUN_DIR="${RUN_DIR:-outputs/patent/electronic_untruncated_val_mrr_09777_cpu_run}"
SHARD_DIR="${SHARD_DIR:-${RUN_DIR}/shards}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
FINAL_OUTPUT="${FINAL_OUTPUT:-outputs/patent/electronic_splade_terms_all_untruncated_val_mrr_09777.json}"

DATASET_JSONS=(
  "data/testset/electronic/new_us_Electrical_Electronics_testset_1_list.json"
  "data/testset/electronic/new_us_Electrical_Electronics_testset_2_list.json"
  "data/testset/electronic/new_us_Electrical_Electronics_testset_3_list.json"
)

mkdir -p "${RUN_DIR}" "${SHARD_DIR}" "${LOG_DIR}"

echo "[INFO] shard_count=${SHARD_COUNT}"
echo "[INFO] document_batch_size=${DOCUMENT_BATCH_SIZE}"
echo "[INFO] encode_batch_size=${ENCODE_BATCH_SIZE}"
echo "[INFO] dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
echo "[INFO] document_encoding_mode=${DOCUMENT_ENCODING_MODE}"
echo "[INFO] term_output_mode=${TERM_OUTPUT_MODE}"
echo "[INFO] base_model=${BASE_MODEL}"
echo "[INFO] checkpoint_path=${CHECKPOINT_PATH}"
echo "[INFO] corpus_path=${CORPUS_PATH}"
echo "[INFO] run_dir=${RUN_DIR}"
echo "[INFO] shard_dir=${SHARD_DIR}"
echo "[INFO] log_dir=${LOG_DIR}"
echo "[INFO] final_output=${FINAL_OUTPUT}"

pids=()
for i in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_id=$(printf "%02d" "${i}")
  shard_total=$(printf "%02d" "${SHARD_COUNT}")
  shard_path="${SHARD_DIR}/electronic_untruncated_shard${shard_id}of${shard_total}.jsonl"
  log_path="${LOG_DIR}/electronic_untruncated_shard${shard_id}of${shard_total}.log"

  if [[ -s "${shard_path}" ]]; then
    echo "[INFO] skipping existing shard ${shard_path}"
    continue
  fi

  echo "[INFO] launching shard ${i}/${SHARD_COUNT} -> ${shard_path}"
  python script/preprocess/patent/export_patent_splade_terms.py \
    --dataset-json "${DATASET_JSONS[@]}" \
    --output-path "${shard_path}" \
    --model-name "${BASE_MODEL}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --patent-source huggingface \
    --corpus-path "${CORPUS_PATH}" \
    --document-batch-size "${DOCUMENT_BATCH_SIZE}" \
    --encode-batch-size "${ENCODE_BATCH_SIZE}" \
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
    --document-encoding-mode "${DOCUMENT_ENCODING_MODE}" \
    --term-output-mode "${TERM_OUTPUT_MODE}" \
    --shard-index "${i}" \
    --shard-count "${SHARD_COUNT}" \
    --output-format jsonl \
    --use-cpu \
    --no-progress \
    >"${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[ERROR] One or more shard workers failed. Check ${LOG_DIR}/electronic_untruncated_shard*.log"
  exit 1
fi

python - <<'PY' "${SHARD_DIR}" "${FINAL_OUTPUT}" "${SHARD_COUNT}" "${BASE_MODEL}" "${CHECKPOINT_PATH}" "${DOCUMENT_BATCH_SIZE}" "${ENCODE_BATCH_SIZE}" "${DATALOADER_NUM_WORKERS}" "${DOCUMENT_ENCODING_MODE}" "${TERM_OUTPUT_MODE}"
import json
import sys
from pathlib import Path

from src.preprocess.patent_splade_terms import merge_patent_term_shards

shard_dir = Path(sys.argv[1])
final_output = Path(sys.argv[2])
shard_count = int(sys.argv[3])
base_model = sys.argv[4]
checkpoint_path = sys.argv[5]
document_batch_size = int(sys.argv[6])
encode_batch_size = int(sys.argv[7])
dataloader_num_workers = int(sys.argv[8])
document_encoding_mode = sys.argv[9]
term_output_mode = sys.argv[10]

shards = sorted(shard_dir.glob("electronic_untruncated_shard*.jsonl"))
if len(shards) != shard_count:
    raise RuntimeError(
        f"Expected {shard_count} shard files in {shard_dir}, found {len(shards)}."
    )

merge_patent_term_shards(
    shard_paths=shards,
    output_path=final_output,
    shard_format="jsonl",
)

manifest = {
    "final_output": str(final_output),
    "shard_dir": str(shard_dir),
    "shard_count": shard_count,
    "base_model": base_model,
    "checkpoint_path": checkpoint_path,
    "document_batch_size": document_batch_size,
    "encode_batch_size": encode_batch_size,
    "dataloader_num_workers": dataloader_num_workers,
    "document_encoding_mode": document_encoding_mode,
    "term_output_mode": term_output_mode,
    "dataset_jsons": [
        "data/testset/electronic/new_us_Electrical_Electronics_testset_1_list.json",
        "data/testset/electronic/new_us_Electrical_Electronics_testset_2_list.json",
        "data/testset/electronic/new_us_Electrical_Electronics_testset_3_list.json",
    ],
    "top_k": None,
    "patent_source": "huggingface",
}
manifest_path = final_output.with_suffix(".manifest.json")
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"[INFO] merged_output={final_output}")
print(f"[INFO] manifest={manifest_path}")
PY

echo "[INFO] Completed."
