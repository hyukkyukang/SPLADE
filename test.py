import requests
import pickle, json
import time
import re
 
def getRQ(query):
    def split_korean_and_english(input_list):
        """
        입력 리스트에서 한글이 포함된 항목과 영어로 된 항목을 분리합니다.
        Args:
            input_list (list): 한글과 영어 문장이 혼합된 리스트.
        Returns:
            tuple: (ko_list, en_list)
                - ko_list: 한글만 포함된 리스트
                - en_list: 영어만 포함된 리스트
        """
        korean_pattern = re.compile('[가-힣]')  # 한글 유니코드 범위를 탐지
        ko_list, en_list = [], []
        for item in input_list:
            if korean_pattern.search(item):  # 한글이 포함되어 있으면
                ko_list.append(item)
            else:  # 영어로만 이루어진 경우
                en_list.append(item)
        return ko_list, en_list
 
    def getPrompt(query):
        example = [[{"role": "user", "content": query}] ]
        return example
    reformulated_query = {"orig": [], "trans": []}
    url = "http://10.4.43.14:8120/generate"
    data = {
        "conversations": getPrompt(query),  # pass only converations
        "do_translate": True # append translations to the original queries
    }
    
    rq_out, rq_trans_out = [],[]
    response = requests.post(url, json=data)
    if response.status_code == 200:
        res = response.json()["result"]
        if isinstance(res, list):
            for line in res:
                if  line[0]["name"] == "retrieval":
                    if "arguments" in line[0]:
                        if "query" in line[0]["arguments"]:
                            rqs = line[0]["arguments"]["query"] #한영RQ 동시에 들어옴
                            rqs = list(dict.fromkeys(rqs)) # 중복제거
                            ko_rqs, en_rqs = split_korean_and_english(rqs)
                            rq_out = ko_rqs
                            rq_trans_out = en_rqs
                        else:
                            print(f"@ no query {line[0]}")
                    else:
                        print(f"@ No arguments {line[0]}")
                else:
                    print(f"@ No retrieval query: {line}\n")
    else:
        print(f"{response.status_code}: {response.text}")
    reformulated_query = {"orig": rq_out, "trans": rq_trans_out}
    
    return reformulated_query
 
 
def get_abstract(_id:str):
    url = "http://10.4.43.27:9200/patent_search/_search" # 각자 테스트 서버 URL 주소
    headers={"Content-Type": "application/json"}
    search_data = {
        "_source": ["doc_id","US_abstract"],
        "query": {
            "bool": {
                "filter": [
                    {"term": {
                    "appl_id": _id
                    }}
                ]
            }
        }
    }
    responses = requests.get(url=url, data=json.dumps(search_data), headers=headers)
    result = responses.json()
    docs = result['hits']['hits'][0]
    abstract = docs['_source']['US_abstract']
        
    return abstract
 
 
 
if __name__ == "__main__":
 
    start = time.time()
    # url = "http://10.1.210.13:8326/api/platform/inference-template/base" # 각자 테스트 서버 URL 주소
    url = "http://10.1.210.13:6124/api/platform/inference-template/base"
 
    headers = {'X-request-id': None}
    data = {'params': '{"inputs_format": "json"}'}
 
 
    search_pass = 0 # 요약은 1
    
    
    # Reranker 학습용 hard-negative 만들기 위함
    files = {'request_type': 'search_patent',
        'question': {
            'query': ['홀로그래픽저장장치 [SEP] 본 발명은 홀로그래픽 저장장치에 관한 것으로, 광을 발생시키는 광원과, 광을 참조광과 물체광으로 분리하는 광분할기와, 참조광과 물체광이 함께 입사될 때 그 셀에 해당하는 정보를 기록하고 참조광만이 입사될 때 이전에 기록된 정보를 재생하는 홀로그램 메모리부와, 정보의 기록 및 재생을 위해 참조광이 홀로그램 메모리부로 임의의 각도를 가지고 입사되도록 그 참조광을 임의의 각도로 편향시키는 제 1 편향기와, 제 1 편향기에서 편향된 참조광을 수직으로 재편향시키는 제 2 편향기와, 제 2 편향기에서 재편향된 참조광을 홀로그램 메모리부로 입사시키는 적어도 하나 이상의 복합 HOE를 갖는 텔레스코우프를 포함하는 제 1 광로변경부와, 정보의 기록을 위해 물체광이 홀로그램 메모리부로 입사되도록 그 물체광의 광로를 조절하는 제 2 광로변경부로 구성함으로써, 기록 용량을 증가시킬 수 있고, 생산성 및 제조원가 측면에서 유리하다.'],
            'date': '',
            'applicant': [],
            'country': [],
            'ipc': []
            },
        'question_id': '',
        'reformulated_query':
            {'orig': [], 'trans': []},
        'rerank': 1,
        'top_n': 150,
        'fids': [],
        'search_pass': 0
    }
 
 
 
    files = [('inputs', ('sample', json.dumps(files).encode()))]
    responses = requests.post(url=url, files=files, data=data, headers=headers)
 
    print("Response:", type(responses), responses)
    responses_body = json.loads(responses.content)
    end = time.time()
    
    #decoded = responses_body.outputs[0].decode()
    decoded = responses_body['outputs'][0]
    list_res = json.loads(decoded)
    print(f"##### list_res length: {len(list_res)}")
    if isinstance(list_res, list):
        print("list_res:", list_res[0:1], "\n")
    elif isinstance(list_res, dict):
        print("list_res:", list_res, "\n")
 
    print(f"{end - start:.5f} sec")
 
    with open('test_search_output_rerank_dqa.txt', "w", encoding="utf-8") as f:
        for line in list_res:
            f.write(str(line)+"\n")
