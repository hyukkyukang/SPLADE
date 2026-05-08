# Reproducing Dense Retrieval Accuracy on Patent Search

- Owner: 강혁규
- Project: 특허 검색
- Status: 작성중
- Created: 2026-04-08
- Last updated: 2026-04-08
- TL;DR: 한솔님이 실험하신 Dense Retrieval 모델 성능을 현재 SPLADE 레포의 Dense evaluation 파이프라인으로 재현하는 방법을 정리

## Goal

- Reproduce the previous retrieval accuracy

|  | top1 | top5 | top10 | top16 | top32 | top64 | top150 | top1000 | top3000 | 비고 | index |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1013 진행, bridge에 negative 추가해서 학습 | 6.84% | 16.12% | 20.72% | 24.14% | 29.76% | 35.59% | 44.27% | 66.24% | 78.34% | `ckpt/officeaction_dpr_positive_neg1_train_kr_0916_dpr_neg1_en_query_train_claims_full_1013/dpr_biencoder.9.238` |  |

## Reference

- Scripts
  - `gs://hansol_lg_bucket/GC-DPR/generate_dense_embeddings_patent.py`
  - `gs://hansol_lg_bucket/GC-DPR/dense_retriever_patent.py`
- Model Checkpoint
  - `gs://hansol_lg_bucket/GC-DPR/ckpt/officeaction_dpr_positive_102103_en_85781_kr_2nd_all_bridges0830/dpr_biencoder.9.121`

## 1. Dataset

### 1.1 Evaluation dataset

- Corpus
  - `Hyukkyu/patent-us-corpus-small`
  - split: `train`
- Label / benchmark
  - `Hyukkyu/patent-us-small`
  - split: `test`

### 1.2 Retrieval unit

이 실험은 문서 단위 retrieval이 아니라 claim passage 단위 retrieval이다.

- 인덱싱 단위
  - `passage_id = {doc_id}&&&claim&&&{chunk_idx}`
- 정답 단위
  - `doc_id`
- 최종 평가 순서
  1. passage를 retrieval
  2. passage를 `parent_doc_id`로 collapse
  3. patent document 단위 metric 계산

즉 내부 검색 단위는 passage지만 최종 metric은 patent document 기준이다.

### 1.3 Metric definition

한솔님이 보시는 `top1 / top5 / top10 / ...` 숫자는 현재 레포 기준으로는 `Success@K` 에 해당한다.

- `Success@K`
  - top-K 안에 정답 patent가 하나라도 있으면 success
- 현재 레포의 `MRR`, `Recall` 과는 다른 metric임
- 한솔님 표를 맞추려면 아래처럼 설정
  - `testing.metric_families=[Success]`
  - `testing.k_list=[1,5,10,16,32,64,150,1000,3000]`

## 2. Model interpretation

### 2.1 Checkpoint structure

`dpr_biencoder.9.121` 는 Hugging Face model directory가 아니라 raw PyTorch checkpoint이다.

확인된 정보:

- top-level keys
  - `model_dict`
  - `optimizer_dict`
  - `scheduler_dict`
  - `offset`
  - `epoch`
  - `encoder_params`
- `encoder_params`
  - `encoder_model_type = bilingual_encoder`
  - `pretrained_model_cfg = bilingual_encoder`
  - `projection_dim = 0`
  - `sequence_length = 512`

### 2.2 Encoder usage

이 checkpoint는 asymmetric DPR bi-encoder 이다.

- query encoding
  - `question_model`
- document / passage encoding
  - `ctx_model`

중요:

- query/doc encoder가 weight sharing이 아니라 서로 다른 tower이다.
- pooling은 `pooler_output` 이 아니라 raw CLS 를 써야 한다.
- similarity는 dot product 를 사용한다.

즉 현재 레포에서는 아래 설정으로 맞춘다.

- `query_pooling=cls`
- `doc_pooling=cls`
- `similarity=dot`
- `normalize=false`

### 2.3 Tokenizer

이 checkpoint 자체에는 tokenizer가 포함되어 있지 않아서 tokenizer asset이 별도로 필요하다.

현재 로컬 실험에서는 아래 tokenizer를 사용했다.

- `data/model/bilingual_patent_checkpoint_28164/checkpoint-28164`

중요:

- `data/model/bilingual_patent_checkpoint_28164/` 루트가 아니라
- `data/model/bilingual_patent_checkpoint_28164/checkpoint-28164/` 를 써야 한다.

## 3. Encoding text format

### 3.1 Passage text format

Coworker script 기준으로 passage는 아래 순서로 생성한다.

1. 원본 patent row에서 `doc_id`, `title`, `claims` 사용
2. `claims` 를 sentence 기준으로 split
3. sentence들을 최대 `300 words` 까지 묶어서 claim chunk 생성
4. 각 chunk 앞에 `title` 을 prefix로 붙임
5. 최종 passage는 최대 `100 words` 로 제한
6. passage id는 `{doc_id}&&&claim&&&{chunk_idx}`

즉 encoder input은 `(title, claims)` 를 따로 넣는 것이 아니라, 최종적으로 합쳐진 한 개의 `text` string 이다.

### 3.2 Passage example

원본 patent row 예시:

```text
doc_id: US12345678
title: Method for controlling a semiconductor device
claims:
1. A method comprising receiving a control signal from a host device.
2. The method of claim 1, further comprising adjusting a gate voltage based on temperature information.
3. The method of claim 2, wherein the device enters a protection mode when an overcurrent event is detected.
```

생성되는 passage row 예시:

```text
passage_id: US12345678&&&claim&&&0
parent_doc_id: US12345678
text: Method for controlling a semiconductor device 1. A method comprising receiving a control signal from a host device. 2. The method of claim 1, further comprising adjusting a gate voltage based on temperature information. 3. The method of claim 2, wherein the device enters a protection mode when an overcurrent event is detected.
```

### 3.3 Real generated passage example

실제로 생성된 passage row 예시는 아래와 같다.

```text
passage_id: US05225450&&&claim&&&0
parent_doc_id: US05225450
text: 1. A safety helmet visor comprising a curved transparent faceshield having an arcuate slot extending therethrough adjacent to each end thereof, a headband provided with coacting securing means whereby the headband can be detachably mounted on a safety helmet, and a stationary capping member at each end of the faceshield extending from the external surface thereof into the arcuate slots, means securing each of the capping members to the headband to link the faceshield to the headband, the part of each capping member which extends into the arcuate slot comprising a rib extending substantially lengthwise of the arcuate slot and terminating at its ends in a pair of cam portions engaging with the walls of the arcuate slot, spring means being positioned between each of the ribs and a wall of each of the arcuate slots to provide for controlled arcuate movement of the faceshield relative to the capping members, said walls of each slot including at least two recesses formed therein facing said spring means, each spring means having a bowed portion cooperating and snap fitting in said recesses as the faceshield is moved relative to the capping members. 2. A safety helmet visor as claimed in claim 1, wherein each spring means is a leaf spring. 3. A safety helmet visor as claimed in claim 1, in which each capping member comprises a disc-like element having said rib projecting from one side thereof substantially along a diameter of the disc. 4. A safety helmet visor as claimed in claim 1, in which on each capping member said cam portions comprise a pair of cylindrical studs and the rib is a substantially linear rib therebetween. 5.
```

### 3.4 Query text format

이번 benchmark에서는 query도 자연어 질문이 아니라 patent document에서 재구성한다.

현재 dense passage 실험에서는 query를 `plain_claims` 로 만들었다.

- query id
  - benchmark의 `question_id`
- query text
  - 해당 patent의 `claims` field만 사용
  - `Title:`, `Claims:` 같은 label prefix는 붙이지 않음

예시:

```text
query_id: US17731340
text: 1. A method for controlling an autonomous vehicle, the method comprising: obtaining trip information ...
```

### 3.5 Real generated query example

실제로 생성된 query row 예시는 아래와 같다.

```text
query_id: US17731340
text: 1. A method for controlling an autonomous vehicle, the method comprising: obtaining trip information of a current trip of a user of the autonomous vehicle and historical traffic information related to the current trip; selecting an autonomous driving (AD) mode that is suitable for the current trip from a plurality of pre-defined AD modes; comparing time required for the autonomous vehicle to complete a portion of the current trip in the selected AD mode with historical average time to complete the portion of the current trip, the historical average time being acquired based on the historical traffic information; and dynamically adjusting driving of the autonomous vehicle based on the comparison to minimize the difference between the time required for the autonomous vehicle to complete the portion of the current trip in the selected AD mode and the historical average time.
```

### 3.6 Real qrels example

```text
query_id: US17731340
doc_id: US17672112
score: 1.0
```

### 3.7 Ranking logic with chunk-level retrieval

이 실험의 핵심은 passage를 검색한 뒤 바로 metric을 계산하는 것이 아니라, 먼저 patent document 단위로 결과를 다시 모으는 것이다.

순서는 아래와 같다.

1. query patent를 `question_model` 로 encode 한다.
2. claim passage corpus의 모든 passage embedding과 dot product 검색을 수행한다.
3. 검색 결과는 `passage_id` 기준으로 나온다.
4. 각 `passage_id` 를 `parent_doc_id` 로 변환한다.
5. 같은 `parent_doc_id` 에 속하는 여러 passage가 top-N 안에 있으면 하나의 patent로 collapse 한다.
6. collapse 시 patent score는 해당 patent에 속한 passage들 중 최고 score를 사용한다.
7. collapse 이후의 patent ranking을 기준으로 `Success@K` 를 계산한다.

중요:

- retrieval unit은 `passage` 이다.
- evaluation unit은 `patent document` 이다.
- 따라서 top-K 결과를 그대로 passage id로 평가하면 안 된다.
- 반드시 `parent_doc_id` 기준 dedup / collapse 이후에 metric을 계산해야 한다.

#### Ranking example

예를 들어 query `US17731340` 에 대해 dense search 결과가 아래처럼 나왔다고 가정한다.

```text
1. US17672112&&&claim&&&3   score=12.8
2. US17672112&&&claim&&&0   score=12.1
3. US18123456&&&claim&&&1   score=11.7
4. US19999999&&&claim&&&0   score=11.2
5. US18123456&&&claim&&&0   score=10.9
```

이를 `parent_doc_id` 로 collapse 하면 아래처럼 바뀐다.

```text
1. US17672112   score=12.8
2. US18123456   score=11.7
3. US19999999   score=11.2
```

즉 top-5 passage retrieval 결과가 top-3 patent ranking으로 바뀐다.

이때 정답 qrel이 아래와 같으면:

```text
query_id: US17731340
label_ids: [US17672112]
```

- `Success@1 = 1`
- `Success@5 = 1`
- `Success@10 = 1`

반대로 collapse 이전 passage ranking만 보고 평가하면 동일 patent의 여러 chunk가 중복 카운트되어 metric 해석이 틀어질 수 있다.

#### Why this differs from plain document retrieval

일반 document retrieval에서는 `1 document = 1 vector` 이므로 검색 결과를 바로 metric에 넣을 수 있다.

하지만 여기서는:

- `1 patent document = multiple passage vectors`

구조이기 때문에 다음 두 단계가 추가로 필요하다.

- passage-level nearest neighbor retrieval
- patent-level grouped ranking

이 grouped ranking이 현재 reproduction protocol에서 dense / sparse 공통으로 맞춰야 하는 핵심 로직이다.

## 4. Reproduction steps

### 4.1 Prepare files

#### Checkpoint

```bash
mkdir -p data/model/dpr_biencoder_officeaction_positive_102103_en_85781_kr_2nd_all_bridges0830
gcloud storage cp \
  gs://hansol_lg_bucket/GC-DPR/ckpt/officeaction_dpr_positive_102103_en_85781_kr_2nd_all_bridges0830/dpr_biencoder.9.121 \
  data/model/dpr_biencoder_officeaction_positive_102103_en_85781_kr_2nd_all_bridges0830/
```

#### Tokenizer

아래 tokenizer directory가 준비되어 있어야 한다.

```text
data/model/bilingual_patent_checkpoint_28164/checkpoint-28164/
```

#### HF access

- `.env` 에 Hugging Face token이 있어야 private / gated dataset 접근 가능
- 예:
  - `HF_TOKEN=...`
  - 또는 `HUGGINGFACE_HUB_TOKEN=...`

### 4.2 Build passage corpus

`Hyukkyu/patent-us-corpus-small` 를 claim-passage corpus로 변환한다.

```bash
python script/preprocess/patent/build_patent_claim_passages.py \
  --corpus-glob '.cache/hf/patent-us-corpus-small/data/*.parquet' \
  --output-path data/corpus/patent_us_claim_passages_small/passages.parquet \
  --max-claim-chunk-words 300 \
  --max-title-prefixed-words 100
```

생성 결과:

- `data/corpus/patent_us_claim_passages_small/passages.parquet`
- `data/corpus/patent_us_claim_passages_small/passages.metadata.json`

### 4.3 Build query / qrels artifacts

`Hyukkyu/patent-us-small` test split으로 query/qrels를 생성한다.

이번 dense passage 실험에서는 `plain_claims` 를 사용한다.

```bash
python script/preprocess/patent/build_patent_us_eval_artifacts.py \
  --benchmark-repo Hyukkyu/patent-us-small \
  --benchmark-split test \
  --corpus-glob '.cache/hf/patent-us-corpus-small/data/*.parquet' \
  --query-text-template plain_claims \
  --output-dir data/eval/patent_us_small_dpr_plain_claims
```

생성 결과:

- `data/eval/patent_us_small_dpr_plain_claims/queries.parquet`
- `data/eval/patent_us_small_dpr_plain_claims/qrels.parquet`
- `data/eval/patent_us_small_dpr_plain_claims/metadata.json`

### 4.4 Encode passage corpus

현재 레포에서는 `model=dpr_bilingual_negative1_ko_en` preset을 architecture wrapper 로만 재사용하고, 실제 weight는 `dpr_biencoder.9.121` 로 override 한다.

즉 `model.name` 은 `dpr_bilingual_negative1_ko_en` 이지만 실제 평가 weight는 `9.121` 이다.

```bash
python script/encode.py \
  model=dpr_bilingual_negative1_ko_en \
  dataset=patent_us_claim_passages_small \
  tag=dpr_officeaction_9_121_patent_passages_plain_claims_20260406_rerun2 \
  model.tokenizer_name=/home/user/SPLADE/data/model/bilingual_patent_checkpoint_28164/checkpoint-28164 \
  encoding.checkpoint_path=/home/user/SPLADE/data/model/dpr_biencoder_officeaction_positive_102103_en_85781_kr_2nd_all_bridges0830/dpr_biencoder.9.121 \
  encoding.batch_size=64 \
  encoding.num_workers=8 \
  encoding.prefetch_factor=4 \
  encoding.num_devices=8 \
  encoding.strategy=ddp \
  encoding.long_doc_strategy=truncate \
  encoding.value_dtype=float16 \
  encoding.index_tag=dpr_officeaction_9_121_patent_passages_plain_claims_20260406_rerun2
```

로컬에서 실제 생성된 encode config:

- `data/embed/dpr_bilingual_negative1_ko_en/dpr_officeaction_9_121_patent_passages_plain_claims_20260406_rerun2/config.yaml`

### 4.5 Build dense index

```bash
python script/index.py \
  model=dpr_bilingual_negative1_ko_en \
  tag=dpr_officeaction_9_121_patent_passages_plain_claims_20260406_rerun2 \
  encoding.index_dir=/mnt/ex-disk-1/hyukkyukang/SPLADE/data/index_user
```

생성 결과:

- `/mnt/ex-disk-1/hyukkyukang/SPLADE/data/index_user/dpr_bilingual_negative1_ko_en/dpr_officeaction_9_121_patent_passages_plain_claims_20260406_rerun2/faiss.index`
- `/mnt/ex-disk-1/hyukkyukang/SPLADE/data/index_user/dpr_bilingual_negative1_ko_en/dpr_officeaction_9_121_patent_passages_plain_claims_20260406_rerun2/metadata.json`

### 4.6 Evaluate

#### Exact reproduction for Hansol-style top-K accuracy

한솔님 표와 맞추려면 `Success@K` 만 계산하면 된다.

```bash
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 \
python script/evaluation.py \
  model=dpr_bilingual_negative1_ko_en \
  dataset=patent_us_small_eval_dpr \
  testing=patent_us_small_eval_dpr \
  tag=dpr_officeaction_9_121_patent_passages_plain_claims_eval \
  model.tokenizer_name=/home/user/SPLADE/data/model/bilingual_patent_checkpoint_28164/checkpoint-28164 \
  testing.checkpoint_path=/home/user/SPLADE/data/model/dpr_biencoder_officeaction_positive_102103_en_85781_kr_2nd_all_bridges0830/dpr_biencoder.9.121 \
  encoding.index_dir=/mnt/ex-disk-1/hyukkyukang/SPLADE/data/index_user \
  encoding.index_tag=dpr_officeaction_9_121_patent_passages_plain_claims_20260406_rerun2 \
  testing.use_cpu=true \
  testing.strategy=single \
  testing.num_devices=1 \
  testing.faiss_use_gpu=false \
  testing.torch_compile=false \
  model.use_fast_tokenizer=false \
  model.require_fast_tokenizer=false \
  'testing.metric_families=[Success]' \
  'testing.k_list=[1,5,10,16,32,64,150,1000,3000]'
```

#### If you also want MRR / Recall

```bash
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 \
python script/evaluation.py \
  model=dpr_bilingual_negative1_ko_en \
  dataset=patent_us_small_eval_dpr \
  testing=patent_us_small_eval_dpr \
  tag=dpr_officeaction_9_121_patent_passages_plain_claims_eval_full \
  model.tokenizer_name=/home/user/SPLADE/data/model/bilingual_patent_checkpoint_28164/checkpoint-28164 \
  testing.checkpoint_path=/home/user/SPLADE/data/model/dpr_biencoder_officeaction_positive_102103_en_85781_kr_2nd_all_bridges0830/dpr_biencoder.9.121 \
  encoding.index_dir=/mnt/ex-disk-1/hyukkyukang/SPLADE/data/index_user \
  encoding.index_tag=dpr_officeaction_9_121_patent_passages_plain_claims_20260406_rerun2 \
  testing.use_cpu=true \
  testing.strategy=single \
  testing.num_devices=1 \
  testing.faiss_use_gpu=false \
  testing.torch_compile=false \
  model.use_fast_tokenizer=false \
  model.require_fast_tokenizer=false \
  'testing.metric_families=[Success,MRR,Recall]' \
  'testing.k_list=[1,5,10,16,32,64,150,1000,3000]'
```

## 5. Current reproduction status

### 5.1 Officeaction checkpoint (`dpr_biencoder.9.121`)

We prepared and ran the pipeline for the following checkpoint:

- `gs://hansol_lg_bucket/GC-DPR/ckpt/officeaction_dpr_positive_102103_en_85781_kr_2nd_all_bridges0830/dpr_biencoder.9.121`

Current status:

- Passage corpus encode: 완료
- Dense index build: 완료
- Final evaluation: 미완료

Generated artifacts:

- Encode config
  - `data/embed/dpr_bilingual_negative1_ko_en/dpr_officeaction_9_121_patent_passages_plain_claims_20260406_rerun2/config.yaml`
- Built index
  - `/mnt/ex-disk-1/hyukkyukang/SPLADE/data/index_user/dpr_bilingual_negative1_ko_en/dpr_officeaction_9_121_patent_passages_plain_claims_20260406_rerun2/faiss.index`
  - `/mnt/ex-disk-1/hyukkyukang/SPLADE/data/index_user/dpr_bilingual_negative1_ko_en/dpr_officeaction_9_121_patent_passages_plain_claims_20260406_rerun2/metadata.json`

Index summary:

- `doc_count = 15,048,639`
- `dim = 768`
- `similarity = dot`
- `has_group_ids = true`

Note:

- Final `evaluation_metrics.json` for this checkpoint was not produced, so top-K accuracy for `9.121` is not available yet.

## 6. Completed evaluation results we already have

아래 결과는 동일한 passage-based patent retrieval protocol에서 이미 완료된 실험 결과이다.

즉:

- corpus는 claim passage 단위
- query는 `plain_claims`
- retrieval 후 `parent_doc_id` 기준으로 collapse
- metric은 patent document 기준

### 6.1 Dense baseline result (`dpr_biencoder.9.146`)

Checkpoint:

- `/home/user/SPLADE/data/model/dpr_biencoder_negative1_ko_en_20251202_filtered/dpr_biencoder.9.146`

Metric file:

- `/mnt/ex-disk-1/hyukkyukang/SPLADE/log/dpr_negative1_ko_en_9_146_patent_passages_plain_claims_success_cpu_20260406/evaluation_metrics.json`

`Success@K` result:

| Metric | Score | Score (%) |
| --- | ---: | ---: |
| Success@1 | 0.0144578312 | 1.45 |
| Success@5 | 0.0463186093 | 4.63 |
| Success@10 | 0.0674698800 | 6.75 |
| Success@16 | 0.0880856737 | 8.81 |
| Success@32 | 0.1298527420 | 12.99 |
| Success@63 | 0.1850066930 | 18.50 |
| Success@150 | 0.2653279901 | 26.53 |
| Success@300 | 0.3365461826 | 33.65 |

### 6.2 Sparse baseline result (SPLADE passage-grouped)

Checkpoint:

- `/home/user/SPLADE/log/train/splade_v3_naver/patent_train_20k_8gpu_bs15_ga2_spladev3_sparse_v1_20260401_125703/checkpoints/step1680_valMRR10_09755.ckpt`

Metric files:

- Success
  - `/mnt/ex-disk-1/hyukkyukang/SPLADE/log/patent_us_small_splade_passage_grouped_step1680_success_cpu_20260406/evaluation_metrics.json`
- MRR / Recall
  - `/mnt/ex-disk-1/hyukkyukang/SPLADE/log/patent_us_small_splade_passage_grouped_step1680/evaluation_metrics.json`

`Success@K` result:

| Metric | Score | Score (%) |
| --- | ---: | ---: |
| Success@1 | 0.0024096386 | 0.24 |
| Success@5 | 0.0069611780 | 0.70 |
| Success@10 | 0.0109772421 | 1.10 |
| Success@16 | 0.0152610438 | 1.53 |
| Success@32 | 0.0214190092 | 2.14 |
| Success@63 | 0.0321285129 | 3.21 |
| Success@150 | 0.0575635880 | 5.76 |
| Success@300 | 0.0789825991 | 7.90 |

Additional sparse metrics:

| Metric | Score |
| --- | ---: |
| MRR@1 | 0.0024096384 |
| MRR@5 | 0.0037037036 |
| MRR@10 | 0.0042390726 |
| MRR@16 | 0.0045662588 |
| MRR@32 | 0.0048225429 |
| MRR@63 | 0.0050556562 |
| MRR@150 | 0.0053054672 |
| MRR@300 | 0.0054165740 |
| Recall@1 | 0.0019634091 |
| Recall@5 | 0.0055332440 |
| Recall@10 | 0.0088353409 |
| Recall@16 | 0.0128514050 |
| Recall@32 | 0.0186523870 |
| Recall@63 | 0.0273761693 |
| Recall@150 | 0.0484381951 |
| Recall@300 | 0.0672021359 |

## 7. Important notes

- `question_model` 로 query encoding, `ctx_model` 로 passage encoding 한다.
- passage text는 `title + claim chunk` 가 이미 합쳐진 하나의 `text` string 이다.
- evaluation 시 retrieval 결과는 `passage_id` 로 나온 뒤 `parent_doc_id` 로 collapse 한다.
- 현재 benchmark (`Hyukkyu/patent-us-small`) 는 `doc_id` 기준 qrels 이므로 grouping도 `parent_doc_id == doc_id` 기준으로 한다.
- original GC-DPR script에서 일부 task는 `appl_id` 로 grouping하지만, 이 benchmark에서는 그렇게 하면 안 된다.
- pooling은 반드시 `cls` 를 사용해야 한다.
  - `pooler_output` 를 쓰면 성능이 크게 깨질 수 있다.
- full exact evaluation은 15M+ passage Flat index를 rank별 GPU로 clone하면 40GB A100 메모리를 넘길 수 있어서, 현재는 CPU FAISS exact search 로 돌리는 것이 안전하다.
  - encode / index는 GPU 사용
  - final exact search만 CPU 사용
- 현재 레포의 기본 `k_list` 는 `[1,5,10,16,32,63,150,300]` 이므로, 한솔님 표를 재현하려면 반드시 `64,1000,3000` 으로 override 해야 한다.

## 8. One-line summary

현재 레포에서 `officeaction_dpr_positive_102103_en_85781_kr_2nd_all_bridges0830/dpr_biencoder.9.121` checkpoint는 다음 프로토콜로 평가한다.

- corpus를 claim passage로 분해
- 각 passage를 `title + claim chunk` text로 `ctx_model` 로 encode
- query는 benchmark patent의 `claims` 만 사용하여 `question_model` 로 encode
- dense flat IP index에서 passage retrieval
- retrieved passage를 `parent_doc_id` 로 collapse
- `Success@K` 로 top-K accuracy 계산
