# =============================================================================
# OpenSearch에서 특허 문서(claims)를 수집하고, 문장 단위로 청킹하여 TSV로 저장하는 스크립트
# =============================================================================

from opensearchpy import OpenSearch    # OpenSearch 클라이언트 라이브러리
from tqdm import tqdm                  # 진행률 표시 바
import csv                             # TSV/CSV 파일 입출력
import pickle                          # 중간 결과 직렬화 저장
import nltk
nltk.download('punkt')                 # 문장 분리(sent_tokenize)에 필요한 데이터 다운로드
from nltk.tokenize import sent_tokenize  # 텍스트를 문장 단위로 분리하는 함수



# ── 설정 값 ──────────────────────────────────────────────────────────────────
src_index = "patent_search"  # 검색 대상 OpenSearch 인덱스 이름

scroll_time = "10m"   # scroll API 유지 시간 (대량 데이터 페이징용)
batch_size = 10000    # 한 번의 scroll 요청으로 가져올 문서 수
org_code = "US"       # 수집할 국가 코드 (US 또는 KR, scripts/opensearch/check_org.py 참고)

# ── 파일 저장 경로 ────────────────────────────────────────────────────────────
pickle_path = "/mnt/ex-disk-1/hansol_jang/patent_data/records_US_2512.pkl"          # 수집된 원본 레코드를 pickle로 저장 (중간 결과물)
output_tsv_file = "/mnt/ex-disk-1/hansol_jang/patent_data/datapool_US_claims_2512.tsv"  # 청킹 완료된 최종 TSV 파일 경로

def clean_text(text):
    """텍스트에서 줄바꿈을 제거하고 연속된 공백을 하나로 정리하는 함수"""
    text = text.replace('\n', ' ')  # 줄바꿈 문자를 공백으로 치환
    words = text.split()            # 연속 공백 기준으로 단어 분리
    return ' '.join(words)          # 단어 사이를 공백 하나로 재결합


# ── OpenSearch 클라이언트 설정 ─────────────────────────────────────────────────
client = OpenSearch(
    hosts=[{'host': '10.4.43.27', 'port': '9200'}],
    http_compress=True,           # 요청/응답 본문에 gzip 압축 사용 (대용량 전송 최적화)
    use_ssl=False,                # SSL 비활성화 (내부망 통신)
    verify_certs=False,           # 인증서 검증 비활성화
    ssl_assert_hostname=False,    # 호스트명 검증 비활성화
    ssl_show_warn=False,          # SSL 관련 경고 메시지 숨김
    timeout=120,                  # 요청 타임아웃: 120초
    max_retries=10,               # 실패 시 최대 재시도 횟수
    retry_on_timeout=True         # 타임아웃 발생 시 자동 재시도
)


# ── 국가 코드에 따른 필드명 동적 생성 ──────────────────────────────────────────
title_field = f"{org_code}_title"       # 예: "US_title"
abstract_field = f"{org_code}_abstract" # 예: "US_abstract"
claims_field = f"{org_code}_claims"     # 예: "US_claims"
source_fields = ["publ_id", "appl_id", title_field, abstract_field, claims_field, "org_code", "cpc"]  # 가져올 필드 목록

# ── 최초 검색 요청 (scroll API 사용) ──────────────────────────────────────────
# scroll API를 사용하여 대량 문서를 페이지 단위로 순차 조회
response = client.search(
    index=src_index,
    scroll=scroll_time,       # scroll 세션 유지 시간
    size=batch_size,          # 첫 번째 배치 크기
    _source=source_fields,    # 응답에 포함할 필드 지정 (불필요한 필드 제외로 성능 향상)
    body={
        "track_total_hits": True,  # 전체 문서 수 정확히 추적
        "query": {
            "term": {
                "org_code": org_code   # org_code 필드가 정확히 일치하는 문서만 조회
            }
        }
    }
)

# ── 첫 번째 응답에서 scroll_id와 결과 추출 ────────────────────────────────────
scroll_id = response['_scroll_id']          # 다음 페이지 요청에 사용할 scroll ID
hits = response['hits']['hits']             # 첫 번째 배치의 문서 리스트
total = response['hits']['total']['value']  # 전체 매칭 문서 수

# ── 첫 번째 배치 레코드 수집 ──────────────────────────────────────────────────
records = []                  # 수집된 레코드를 저장할 리스트
pbar = tqdm(total=total)      # 전체 문서 수 기준 진행률 바 생성
pbar.update(len(hits))        # 첫 번째 배치만큼 진행률 업데이트

for doc in hits:
    doc_id = doc["_id"]       # OpenSearch 문서 고유 ID
    source = doc["_source"]   # 문서 본문 (요청한 필드만 포함)
    record = {
        "doc_id": doc_id,
        "publ_id": source.get("publ_id", ""),       # 공개번호
        "appl_id": source.get("appl_id", ""),       # 출원번호
        "cpc": source.get("cpc", ""),               # CPC 분류코드
        "title": source.get(title_field, ""),       # 특허 제목
        "abstract": source.get(abstract_field, ""), # 특허 초록
        "claims": source.get(claims_field, "")      # 특허 청구항
    }
    records.append(record)

# ── scroll 반복: 나머지 배치를 순차적으로 가져옴 ──────────────────────────────
while True:
    response = client.scroll(scroll_id=scroll_id, scroll=scroll_time)  # 다음 배치 요청
    scroll_id = response["_scroll_id"]  # scroll ID 갱신 (매 응답마다 변경될 수 있음)
    hits = response["hits"]["hits"]
    if not hits:  # 더 이상 가져올 문서가 없으면 종료
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
        }
        records.append(record)
    pbar.update(len(hits))  # 현재 배치 크기만큼 진행률 업데이트

pbar.close()
print(f"✅ 총 수집된 {org_code} 문서 수: {len(records):,}")  # 수집 완료 메시지 출력

# ── 수집된 레코드를 pickle 파일로 저장 (중간 결과 백업) ────────────────────────
with open(pickle_path, "wb") as f:
    pickle.dump(records, f)

def split_into_chunks_by_sentence_nltk(text, max_words=300):
    """
    텍스트를 문장 단위로 분리한 뒤, 최대 단어 수(max_words) 이하의 청크로 묶는 함수.
    - NLTK의 sent_tokenize를 사용하여 문장 경계를 감지
    - 각 청크는 max_words 단어를 초과하지 않도록 문장 단위로 그룹핑
    """
    sentences = sent_tokenize(text)  # 텍스트를 문장 리스트로 분리
    chunks = []           # 최종 청크 리스트
    current_chunk = []    # 현재 구성 중인 청크 (문장 리스트)
    current_len = 0       # 현재 청크의 누적 단어 수
    for sentence in sentences:
        word_count = len(sentence.split())  # 현재 문장의 단어 수
        if word_count == 0:
            continue  # 빈 문장은 건너뜀
        if current_len + word_count > max_words:
            # 현재 문장을 추가하면 max_words를 초과하는 경우
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())  # 기존 청크 저장
            current_chunk = [sentence]   # 새 청크 시작
            current_len = word_count
        else:
            # 아직 여유가 있으면 현재 청크에 문장 추가
            current_chunk.append(sentence)
            current_len += word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())  # 마지막 남은 청크 저장
    return chunks


# ── 청킹된 claims를 TSV 파일로 저장 ───────────────────────────────────────────
# 각 청크 앞에 title을 붙여서 최종 텍스트를 구성
# 최종 텍스트는 MAX_WORDS 이하가 되도록 title을 잘라냄
MAX_WORDS = 100  # 청크 + 제목을 합친 최대 단어 수 제한
with open(output_tsv_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')

    for rec in tqdm(records, desc="Writing chunks"):
        content = rec.get('claims', '')                  # 청구항 텍스트
        title = clean_text(rec.get('title', '') or "")   # 특허 제목 (정제)

        if content:
            content = clean_text(content)                            # 청구항 텍스트 정제
            chunks = split_into_chunks_by_sentence_nltk(content)     # 문장 단위 청킹

            for i, chunk in enumerate(chunks):
                chunk_clean = chunk.strip()
                chunk_words = chunk_clean.split()

                # 제목(title)을 청크 앞에 붙이는 로직
                if title:
                    title_words = title.split()
                    merged_words = title_words + chunk_words

                    if len(merged_words) <= MAX_WORDS:
                        # 제목 + 청크가 MAX_WORDS 이하 → 제목 전체 포함
                        final_words = merged_words
                    else:
                        # 제목 + 청크가 MAX_WORDS 초과 → 청크는 유지하고 제목을 잘라서 포함
                        space_for_title = MAX_WORDS - len(chunk_words)
                        if space_for_title > 0:
                            trimmed_title_words = title_words[:space_for_title]  # 제목을 남은 공간만큼만 사용
                            final_words = trimmed_title_words + chunk_words
                        else:
                            # 제목을 넣을 공간이 없는 경우 → 청크만 사용
                            final_words = chunk_words
                else:
                    # 제목이 없는 경우 → 청크만 사용
                    final_words = chunk_words

                final_text = " ".join(final_words)

                # TSV 행 구성: [문서ID, 최종텍스트, 출원번호, 공개번호&&&claim&&&청크인덱스]
                writer.writerow([
                    rec.get('doc_id', ''),
                    final_text,
                    rec.get('appl_id', ''),
                    rec.get('publ_id', '') + "&&&claim&&&" + str(i)
                ])
