#!/usr/bin/env bash
set -euo pipefail

GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"

SHARD_COUNT="${SHARD_COUNT:-${#GPU_IDS[@]}}"
DOCUMENT_BATCH_SIZE="${DOCUMENT_BATCH_SIZE:-128}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
DOCUMENT_ENCODING_MODE="${DOCUMENT_ENCODING_MODE:-combined_fields_truncate_head}"
TERM_OUTPUT_MODE="${TERM_OUTPUT_MODE:-flat_terms}"

BASE_MODEL="${BASE_MODEL:-/home/user/.cache/huggingface/hub/models--naver--splade-v3/snapshots/fdfeceb91d7b9de7985b38addd3ba9f53a59a355}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/user/SPLADE/log/train/splade_v3_naver/no_tag/checkpoints/last.ckpt}"

OPENSEARCH_URL="${OPENSEARCH_URL:-http://10.4.43.27:9200}"
OPENSEARCH_INDEX="${OPENSEARCH_INDEX:-patent_search_index_260504}"
OPENSEARCH_FIELD_PREFIX="${OPENSEARCH_FIELD_PREFIX:-EP}"

RUN_DIR="${RUN_DIR:-outputs/patent/epo_electronics_run}"
SHARD_DIR="${SHARD_DIR:-${RUN_DIR}/shards}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
FINAL_OUTPUT="${FINAL_OUTPUT:-outputs/patent/epo_electronics_splade_terms.json}"

DATASET_JSONS=(
  "data/testset/epo/ep_electronics_testset_questions_only.json"
)

mkdir -p "${RUN_DIR}" "${SHARD_DIR}" "${LOG_DIR}"

if [[ "${SHARD_COUNT}" -ne "${#GPU_IDS[@]}" ]]; then
  echo "[ERROR] SHARD_COUNT must match the number of GPU ids."
  exit 1
fi

pids=()
for i in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_id=$(printf "%02d" "${i}")
  shard_total=$(printf "%02d" "${SHARD_COUNT}")
  gpu_id="${GPU_IDS[$i]}"
  shard_path="${SHARD_DIR}/epo_electronics_shard${shard_id}of${shard_total}.jsonl"
  log_path="${LOG_DIR}/epo_electronics_shard${shard_id}of${shard_total}.log"

  if [[ -s "${shard_path}" ]]; then
    echo "[INFO] skipping existing shard ${shard_path}"
    continue
  fi

  echo "[INFO] launching shard ${i}/${SHARD_COUNT} on gpu=${gpu_id} -> ${shard_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" OMP_NUM_THREADS=4 python script/preprocess/patent/export_patent_splade_terms.py \
    --dataset-json "${DATASET_JSONS[@]}" \
    --output-path "${shard_path}" \
    --model-name "${BASE_MODEL}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --patent-source opensearch \
    --opensearch-url "${OPENSEARCH_URL}" \
    --opensearch-index "${OPENSEARCH_INDEX}" \
    --opensearch-field-prefix "${OPENSEARCH_FIELD_PREFIX}" \
    --document-batch-size "${DOCUMENT_BATCH_SIZE}" \
    --encode-batch-size "${ENCODE_BATCH_SIZE}" \
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
    --document-encoding-mode "${DOCUMENT_ENCODING_MODE}" \
    --term-output-mode "${TERM_OUTPUT_MODE}" \
    --shard-index "${i}" \
    --shard-count "${SHARD_COUNT}" \
    --output-format jsonl \
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
  echo "[ERROR] One or more shard workers failed. Check ${LOG_DIR}/epo_electronics_shard*.log"
  exit 1
fi

# Merge shards (keyed by appl_id) then remap to publ_id.
PUBL_MAP="${PUBL_MAP:-data/testset/epo/ep_electronics_publ_to_appl.json}"

python - <<'PY' "${SHARD_DIR}" "${FINAL_OUTPUT}" "${SHARD_COUNT}" "${PUBL_MAP}" "${BASE_MODEL}" "${CHECKPOINT_PATH}" "${DOCUMENT_BATCH_SIZE}" "${ENCODE_BATCH_SIZE}" "${DATALOADER_NUM_WORKERS}" "${GPU_IDS_CSV}" "${DOCUMENT_ENCODING_MODE}" "${TERM_OUTPUT_MODE}" "${OPENSEARCH_INDEX}" "${OPENSEARCH_FIELD_PREFIX}"
import json
import sys
from pathlib import Path

from src.preprocess.patent_splade_terms import merge_patent_term_shards

shard_dir = Path(sys.argv[1])
final_output = Path(sys.argv[2])
shard_count = int(sys.argv[3])
publ_map_path = Path(sys.argv[4])

shards = sorted(shard_dir.glob("epo_electronics_shard*.jsonl"))
if len(shards) != shard_count:
    raise RuntimeError(
        f"Expected {shard_count} shard files in {shard_dir}, found {len(shards)}."
    )

appl_keyed_path = final_output.with_name(final_output.stem + "_by_appl.json")
merge_patent_term_shards(
    shard_paths=shards,
    output_path=appl_keyed_path,
    shard_format="jsonl",
)

publ_to_appl = json.loads(publ_map_path.read_text(encoding="utf-8"))
appl_terms = json.loads(appl_keyed_path.read_text(encoding="utf-8"))
remapped = {}
missing = []
for publ_id, appl_id in publ_to_appl.items():
    payload = appl_terms.get(appl_id)
    if payload is None:
        missing.append((publ_id, appl_id))
        continue
    remapped[publ_id] = payload
final_output.write_text(json.dumps(remapped, ensure_ascii=False), encoding="utf-8")

manifest = {
    "final_output": str(final_output),
    "appl_keyed_intermediate": str(appl_keyed_path),
    "publ_to_appl_map": str(publ_map_path),
    "shard_dir": str(shard_dir),
    "shard_count": shard_count,
    "gpu_ids": sys.argv[10],
    "base_model": sys.argv[5],
    "checkpoint_path": sys.argv[6],
    "document_batch_size": int(sys.argv[7]),
    "encode_batch_size": int(sys.argv[8]),
    "dataloader_num_workers": int(sys.argv[9]),
    "document_encoding_mode": sys.argv[11],
    "term_output_mode": sys.argv[12],
    "opensearch_index": sys.argv[13],
    "opensearch_field_prefix": sys.argv[14],
    "dataset_jsons": [
        "data/testset/epo/ep_electronics_testset_questions_only.json"
    ],
    "patent_source": "opensearch",
    "publ_id_total": len(publ_to_appl),
    "publ_id_resolved": len(remapped),
    "publ_id_missing": len(missing),
    "publ_id_missing_sample": missing[:20],
}
manifest_path = final_output.with_suffix(".manifest.json")
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"[INFO] merged_output={final_output}")
print(f"[INFO] appl_keyed_intermediate={appl_keyed_path}")
print(f"[INFO] manifest={manifest_path}")
print(f"[INFO] publ_id_resolved={len(remapped)}/{len(publ_to_appl)} (missing={len(missing)})")
PY

echo "[INFO] Completed."
