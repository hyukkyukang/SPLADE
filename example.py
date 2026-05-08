from opensearchpy import OpenSearch, helpers
from tqdm import tqdm
from datetime import datetime
import time
import csv
import pickle
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# src_index = "patent_search_index_0630"
# src_index = "patent_search_index_0917"
src_index = "patent_search_index_1027"

scroll_time = "10m"  # scroll 유지 시간
batch_size = 10000    # 한 번에 가져올 문서 수
org_code="US" # or KR (scripts/opensearch/check_org.py)

# 중간과정 저장 경로
pickle_path = "/mnt/ex-disk-1/hansol_jang/patent_data/records_US_2512.pkl"
output_tsv_file = "/mnt/ex-disk-1/hansol_jang/patent_data/datapool_US_claims_2512.tsv"

def clean_text(text):
    text = text.replace('\n', ' ')
    words = text.split()
    return ' '.join(words)


# OpenSearch 클라이언트 설정
client = OpenSearch(
    hosts=[{'host': '10.4.43.27', 'port': '9200'}],
    http_compress=True,  # 요청 본문에 gzip 압축 사용
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    timeout=120,
    max_retries=10,
    retry_on_timeout=True
)


title_field = f"{org_code}_title"
abstract_field = f"{org_code}_abstract"
claims_field = f"{org_code}_claims"
# description_field = f"{org_code}_description"
# source_fields = ["publ_id", "appl_id", title_field, description_field, "org_code"]
source_fields = ["publ_id", "appl_id", title_field, abstract_field, claims_field, "org_code", "cpc"]

# # search 요청
response = client.search(
    index=src_index,
    scroll=scroll_time,
    size=batch_size,
    _source=source_fields,
    body={
        "track_total_hits": True,
        "query": {
            "term": {
                "org_code": org_code
            }
        }
    }
)

scroll_id = response['_scroll_id']
hits = response['hits']['hits']
total = response['hits']['total']['value']

# 수집 리스트
records = []
pbar = tqdm(total=total)
pbar.update(len(hits))

for doc in hits:
    doc_id = doc["_id"]
    source = doc["_source"]
    record = {
        "doc_id": doc_id,
        "publ_id": source.get("publ_id", ""),
        "appl_id": source.get("appl_id", ""),
        "cpc": source.get("cpc", ""),
        "title": source.get(title_field, ""),
        "abstract": source.get(abstract_field, ""),
        "claims": source.get(claims_field, "")
        # "description": source.get(description_field, "")
    }
    records.append(record)

# scroll 반복
while True:
    response = client.scroll(scroll_id=scroll_id, scroll=scroll_time)
    scroll_id = response["_scroll_id"]
    hits = response["hits"]["hits"]
    if not hits:
        break
    for doc in hits:
        doc_id = doc["_id"]
        source = doc["_source"]
        record = {
            "doc_id": doc_id,
            "publ_id": source.get("publ_id", ""),
            "appl_id": source.get("appl_id", ""),
            "title": source.get(title_field, ""),
            "abstract": source.get(abstract_field, ""),
            "claims": source.get(claims_field, "")
            # "description": source.get(description_field, "")
        }
        records.append(record)
    pbar.update(len(hits))

pbar.close()
print(f"✅ 총 수집된 {org_code} 문서 수: {len(records):,}")

# # 저장
with open(pickle_path, "wb") as f:
    pickle.dump(records, f)


# # for rec in tqdm(records):
# #     content = rec.get('title', '')+" "+rec.get('abstract', '')
# #     if content == "":
# #         print(rec["doc_id"])
# # 해야하는 것은 두 가지인데, 
# # dict가 하나 필요하다 (appl_id: [publ_id, publ_id, ...]) -> 실제로 이렇게 하면 점수 측정에는 도움이 되지만, 실제 서비스 상황과 거리가 있음. 
# # datapool abstract tsv 하나 필요하다. 
# # with open('/mnt/ex-disk-1/hansol_jang/patent_data/datapool_US_abstract.tsv', 'w', encoding='utf-8', newline='') as f:
# #     writer = csv.writer(f, delimiter='\t')
# # #     # 헤더 작성
# # #     writer.writerow(['_id', 'abstract', 'appl_id', 'publ_id']) -> 헤더는 필요없음. 
# # #     # 각 레코드 작성
# #     for rec in records:
# #         content = rec.get('title', '')+" "+rec.get('abstract', '')
# #         writer.writerow([
# #             rec.get('doc_id', ''),
# #             clean_text(content),
# #             rec.get('appl_id', ''),
# #             rec.get('publ_id', '')+"&&&abstract"
# #         ])


def split_into_chunks_by_sentence_nltk(text, max_words=300):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if word_count == 0:
            continue
        if current_len + word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_len = word_count
        else:
            current_chunk.append(sentence)
            current_len += word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    return chunks


# with open(output_tsv_file, 'w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f, delimiter='\t')
# #     # 헤더 작성
# #     writer.writerow(['_id', 'abstract', 'appl_id', 'publ_id']) -> 헤더는 필요없음. 
# #     # 각 레코드 작성
#     for rec in tqdm(records, desc="Writing description chunks"):
#         content = rec.get('claims', '')
#         if content:
#             content = clean_text(content)
#             chunks = split_into_chunks_by_sentence_nltk(content)
#             for i, chunk in enumerate(chunks):
#                 writer.writerow([
#                     rec.get('doc_id', ''),
#                     chunk,
#                     rec.get('appl_id', ''),
#                     rec.get('publ_id', '')+"&&&claim&&&"+str(i) ## KR 이기 때문
#                 ])
MAX_WORDS = 100

with open(output_tsv_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')

    for rec in tqdm(records, desc="Writing description chunks"):
        content = rec.get('claims', '')
        title = clean_text(rec.get('title', '') or "")

        if content:
            content = clean_text(content)
            chunks = split_into_chunks_by_sentence_nltk(content)

            for i, chunk in enumerate(chunks):
                chunk_clean = chunk.strip()
                chunk_words = chunk_clean.split()

                # 기본적으로 title 붙일 준비
                if title:
                    title_words = title.split()
                    merged_words = title_words + chunk_words

                    if len(merged_words) <= MAX_WORDS:
                        # ⬅ 전체가 100 words 이하 → title 전체 포함
                        final_words = merged_words
                    else:
                        # ⬅ 전체가 100 words 초과 → title을 잘라서 포함
                        # chunk는 최대한 유지하고, title에서 잘라내기
                        space_for_title = MAX_WORDS - len(chunk_words)
                        if space_for_title > 0:
                            trimmed_title_words = title_words[:space_for_title]
                            final_words = trimmed_title_words + chunk_words
                        else:
                            # title 넣을 공간이 아예 없는 경우 → chunk만 사용
                            final_words = chunk_words
                else:
                    final_words = chunk_words

                final_text = " ".join(final_words)

                writer.writerow([
                    rec.get('doc_id', ''),
                    final_text,
                    rec.get('appl_id', ''),
                    rec.get('publ_id', '') + "&&&claim&&&" + str(i)
                ])
 