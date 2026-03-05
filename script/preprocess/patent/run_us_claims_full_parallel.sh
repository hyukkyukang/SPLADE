#!/usr/bin/env bash
set -euo pipefail

NUM_SLICES="${NUM_SLICES:-24}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="outputs/patent/us_claims_corpus_full_${RUN_ID}"
FINAL_TSV="${OUT_DIR}/us_claims_corpus_full.tsv"
FINAL_STATS="${OUT_DIR}/us_claims_corpus_full_stats.json"

mkdir -p "${OUT_DIR}"

echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] num_slices=${NUM_SLICES}"

pids=()
for i in $(seq 0 $((NUM_SLICES - 1))); do
  slice_id=$(printf "%02d" "${i}")
  part_tsv="${OUT_DIR}/part_${slice_id}.tsv"
  part_stats="${OUT_DIR}/part_${slice_id}.stats.json"
  part_log="${OUT_DIR}/part_${slice_id}.log"

  python script/preprocess/patent/export_en_claims_corpus.py \
    --org-code US \
    --batch-size 10000 \
    --no-progress \
    --num-slices "${NUM_SLICES}" \
    --slice-id "${i}" \
    --output-tsv "${part_tsv}" \
    --stats-json "${part_stats}" \
    >"${part_log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[ERROR] One or more slice workers failed. Check ${OUT_DIR}/part_*.log"
  exit 1
fi

echo "[INFO] Merging TSV parts..."
cat "${OUT_DIR}"/part_*.tsv > "${FINAL_TSV}"

echo "[INFO] Building final stats manifest..."
python - <<'PY' "${OUT_DIR}" "${FINAL_TSV}" "${FINAL_STATS}" "${NUM_SLICES}"
import json
import pathlib
import sys

from opensearchpy import OpenSearch

out_dir = pathlib.Path(sys.argv[1])
final_tsv = pathlib.Path(sys.argv[2])
final_stats = pathlib.Path(sys.argv[3])
num_slices = int(sys.argv[4])

part_stats_paths = sorted(out_dir.glob("part_*.stats.json"))
if len(part_stats_paths) != num_slices:
    raise RuntimeError(
        f"Expected {num_slices} part stats files, found {len(part_stats_paths)}"
    )

processed_docs = 0
docs_with_claims = 0
claim_chunk_rows = 0
for path in part_stats_paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    processed_docs += int(data.get("processed_docs", 0))
    docs_with_claims += int(data.get("docs_with_non_empty_claims", 0))
    claim_chunk_rows += int(data.get("claim_chunk_rows", 0))

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
    body={"query": {"term": {"org_code": "US"}}},
)["count"]

stats = {
    "source_index": "patent_search",
    "filters": {"org_code": "US", "lang_type": None},
    "num_slices": num_slices,
    "total_matching_docs": total_matching_docs,
    "processed_docs": processed_docs,
    "docs_with_non_empty_claims": docs_with_claims,
    "claim_chunk_rows": claim_chunk_rows,
    "output_tsv": final_tsv.as_posix(),
    "parts_dir": out_dir.as_posix(),
}
final_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
PY

echo "[INFO] Completed."
echo "[INFO] final_tsv=${FINAL_TSV}"
echo "[INFO] final_stats=${FINAL_STATS}"
