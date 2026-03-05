#!/usr/bin/env bash
set -euo pipefail

ORG_CODE="${ORG_CODE:-US}"
NUM_SLICES="${NUM_SLICES:-24}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/patent/hf_${ORG_CODE,,}_patent_docs_${RUN_ID}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-patent_${ORG_CODE,,}_docs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] org_code=${ORG_CODE}"
echo "[INFO] num_slices=${NUM_SLICES}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] output_prefix=${OUTPUT_PREFIX}"

pids=()
for i in $(seq 0 $((NUM_SLICES - 1))); do
  log_file="${OUTPUT_DIR}/slice_$(printf "%02d" "${i}").log"
  python script/preprocess/patent/export_patent_documents_for_hf.py \
    --org-code "${ORG_CODE}" \
    --num-slices "${NUM_SLICES}" \
    --slice-id "${i}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --no-progress \
    ${EXTRA_ARGS} \
    >"${log_file}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[ERROR] Some slice workers failed. Check ${OUTPUT_DIR}/slice_*.log"
  exit 1
fi

MANIFEST_PATH="${OUTPUT_DIR}/${OUTPUT_PREFIX}_export_manifest.json"

python - <<'PY' "${OUTPUT_DIR}" "${OUTPUT_PREFIX}" "${NUM_SLICES}" "${ORG_CODE}" "${MANIFEST_PATH}"
import glob
import json
import pathlib
import sys

from opensearchpy import OpenSearch

output_dir = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
num_slices = int(sys.argv[3])
org_code = sys.argv[4]
manifest_path = pathlib.Path(sys.argv[5])

stats_paths = sorted(output_dir.glob(f"{prefix}_slice*of*_stats.json"))
if len(stats_paths) != num_slices:
    raise RuntimeError(
        f"Expected {num_slices} slice stats files, found {len(stats_paths)}"
    )

processed_docs = 0
emitted_docs = 0
missing_value_counts: dict[str, int] = {}
source_missing_field_counts: dict[str, int] = {}
target_columns = None
source_fields_used = None
compression = None
shard_size = None
shard_count = 0
shard_paths: list[str] = []

for path in stats_paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    processed_docs += int(data.get("processed_docs", 0))
    emitted_docs += int(data.get("emitted_docs", 0))

    for key, value in data.get("missing_value_counts", {}).items():
        missing_value_counts[key] = missing_value_counts.get(key, 0) + int(value)
    for key, value in data.get("source_missing_field_counts", {}).items():
        source_missing_field_counts[key] = source_missing_field_counts.get(key, 0) + int(value)

    parquet_info = data.get("parquet", {})
    shard_count += int(parquet_info.get("shard_count", 0))
    shard_paths.extend(parquet_info.get("shard_paths", []))
    if target_columns is None:
        target_columns = data.get("target_columns")
    if source_fields_used is None:
        source_fields_used = data.get("source_fields_used")
    if compression is None:
        compression = parquet_info.get("compression")
    if shard_size is None:
        shard_size = parquet_info.get("shard_size")

client = OpenSearch(
    hosts=[{"host": "10.4.43.27", "port": "9200"}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    timeout=120,
)
total_matching_docs = client.count(
    index="patent_search",
    body={"query": {"term": {"org_code": org_code}}},
)["count"]

manifest = {
    "source_index": "patent_search",
    "filters": {"org_code": org_code},
    "num_slices": num_slices,
    "target_columns": target_columns,
    "source_fields_used": source_fields_used,
    "total_matching_docs": total_matching_docs,
    "processed_docs": processed_docs,
    "emitted_docs": emitted_docs,
    "missing_value_counts": missing_value_counts,
    "source_missing_field_counts": source_missing_field_counts,
    "parquet": {
        "compression": compression,
        "shard_size": shard_size,
        "shard_count": shard_count,
        "shard_glob": f"{output_dir.as_posix()}/{prefix}_slice*of*-*.parquet",
        "shard_paths": sorted(shard_paths),
    },
    "hf_load_example": (
        f"from datasets import load_dataset\\n"
        f"ds = load_dataset('parquet', data_files='{output_dir.as_posix()}/{prefix}_slice*of*-*.parquet', split='train')"
    ),
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

echo "[INFO] Completed."
echo "[INFO] manifest=${MANIFEST_PATH}"
