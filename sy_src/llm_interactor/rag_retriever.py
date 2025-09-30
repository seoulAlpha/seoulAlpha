# rag_retriever.py

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# --- 설정 ---
# .env 파일 로드
load_dotenv()

# os.environ['HF_HOME'] = 'D:/huggingface_cache'

# 모델 및 파일 경로 정의
MODEL_NAME = 'jhgan/ko-sbert-nli'
#LLM_MODEL_NAME = 'gpt-4o-mini'
LLM_MODEL_NAME = 'gpt-3.5-turbo'
OUTPUT_DIR = 'data/faiss/faiss_merged_output'
INDEX_FILE = f'{OUTPUT_DIR}/merged.index'
METADATA_FILE = f'{OUTPUT_DIR}/merged_metadata.jsonl'
TOP_K = 10  # 검색할 결과의 수

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("API_KEY"))

# --- 리소스 로딩 ---
def _load_resources():
    """모듈 로딩 시 검색에 필요한 리소스를 미리 불러옵니다."""
    try:
        print("1. RAG 리소스를 로딩합니다...")
        model = SentenceTransformer(MODEL_NAME)
        index = faiss.read_index(INDEX_FILE)
        metadata_map = {}
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                meta = json.loads(line)
                metadata_map[meta['vector_id']] = meta
        print("RAG 리소스 로딩 완료!")
        return model, index, metadata_map
    except Exception as e:
        print(f"RAG 리소스 로딩에 실패했습니다: {e}")
        return None, None, None

# 모듈이 임포트될 때 리소스를 한 번만 로드합니다.
embedding_model, faiss_index, meta_map = _load_resources()


def filter_by_regions_get_ids(meta_map, region_list):
    """
    meta_map을 순회하며 주어진 region_list에 해당하는 벡터의 ID(인덱스) 목록을 반환하는 함수
    """
    ids = []
    for vector_id, meta in meta_map.items():
        for region in region_list:
            # region_l2 값이 없으면 region_l1만 비교
            if region['region_l2'] == "":
                if meta['region_l1'] == region['region_l1']:
                    ids.append(vector_id)
                    break
            else:
                if (meta['region_l1'] == region['region_l1'] and
                    meta.get('region_l2') == region['region_l2']):
                    ids.append(vector_id)
                    break
    return ids

"""
def _retrieve_places(query, k, region_list=None):
    # 1. meta_map 필터링
    region_list = [ {"region_l1": "전북특별자치도", "region_l2": ""}
                   , {"region_l1": "광주광역시", "region_l2": ""}
                   , {"region_l1": "전라남도", "region_l2": ""} ]
    if region_list:
        search_space = filter_by_regions(meta_map, region_list)
    else:
        search_space = meta_map

    if not search_space:
        return []

    # 2. 쿼리 임베딩
    query_vector = embedding_model.encode([query])
    distances, ids = faiss_index.search(query_vector.astype("float32"), k)

    # 3. 결과 반환 (필터링된 대상만)
    results = []
    for vector_id in ids[0]:
        if vector_id in search_space:
            results.append(search_space[vector_id])
    return results
"""

def _retrieve_places(query, k, region_list=None):
    """
    쿼리를 기반으로 '지정된 지역 내에서' 유사한 장소를 효율적으로 검색합니다.
    """
    # 1. meta_map에서 검색 대상이 될 벡터 ID 목록 필터링
    region_list = [ {"region_l1": "전북특별자치도", "region_l2": ""}
                   , {"region_l1": "광주광역시", "region_l2": ""}
                   , {"region_l1": "전라남도", "region_l2": ""} ]
    
    # search_space는 이제 데이터 자체가 아닌, 검색 대상이 될 'ID의 리스트'가 되어야 합니다.
    # filter_by_regions가 해당 지역에 속하는 벡터의 ID(인덱스) 리스트를 반환한다고 가정합니다.
    if region_list:
        search_ids = filter_by_regions_get_ids(meta_map, region_list) 
    else:
        # region_list가 없으면 전체 ID를 대상으로 검색
        search_ids = list(range(faiss_index.ntotal))

    if not search_ids:
        print("해당 지역에 검색 대상이 없습니다.")
        return [], []

    # 2. 쿼리 임베딩
    query_vector = embedding_model.encode([query])
    search_ids_np = np.array(search_ids, dtype='int64')
    selector = faiss.IDSelectorArray(search_ids_np)
    params = faiss.SearchParametersIVF()
    params.sel = selector
    params.nprobe = 10
    distances, ids = faiss_index.search(query_vector.astype("float32"), k, params=params)

    results = []
    valid_ids = ids[0][ids[0] != -1]
    
    for vector_id in valid_ids:
        results.append(meta_map[vector_id])
        
    return results

def _generate_answer_with_llm(query, retrieved_places):
    """내부 함수: 검색된 정보를 바탕으로 LLM 답변을 생성합니다."""
    context = ""
    for i, place in enumerate(retrieved_places): # 상위 5개 정보만 사용
        context += f"--- 장소 정보 {i+1} ---\n"
        context += f"이름: {place.get('name', '정보 없음')}\n"
        context += f"주소: {place.get('address', '정보 없음')}\n"  # <--- 1. '주소' 정보 추가
        context += f"AI 요약: {place.get('ai_summary', '정보 없음')}\n"
        processed_sentences = place.get('processed_sentences', [])
        context += "주요 특징 및 후기:\n"
        for sentence in processed_sentences:
            context += f"- {sentence}\n"
        context += "\n"

    system_prompt = "당신은 사용자의 질문에 가장 적합한 장소를 추천해주는 유용한 어시스턴트입니다."
    # <--- 2. 지시사항 수정
    user_prompt = f"""
    아래 '장소 정보'만을 바탕으로 사용자의 질문에 대한 답변을 생성해 주세요.

    [지시사항]
    1. 검색된 장소 중에서 질문과 가장 관련성이 높은 2~3곳을 추천해 주세요.
    2. 각 장소를 추천할 때, 반드시 '이름'과 '주소'를 명확하게 함께 표시해주세요.
    3. 각 장소를 추천하는 이유를 'AI 요약'과 '주요 특징 및 후기'를 근거로 구체적으로 설명해 주세요.
    4. 'processed_sentences'에 있는 실제 후기를 인용하여 답변하면 신뢰도를 높일 수 있습니다.
    5. 친절하고 자연스러운 말투로 답변해 주세요.

    --- 장소 정보 ---
    {context}
    --- 사용자의 질문 ---
    {query}
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM 답변 생성 중 오류가 발생했습니다: {e}"

# --- 대표 실행 함수 ---
# search_query 외에 region_keywords를 인자로 받도록 변경
def get_rag_recommendation(search_query, region_keywords):
    """
    검색 쿼리와 지역 키워드를 받아 RAG 시스템을 통해 최종 추천 답변을 반환합니다.
    """
    if not all([embedding_model, faiss_index, meta_map]):
        return "RAG 시스템이 준비되지 않아 추천을 생성할 수 없습니다."
        
    # 1. 장소 검색
    print("\n[RAG] 의미적으로 유사한 장소를 검색합니다...")
    top_places = _retrieve_places(search_query, k=5)
    
    if not top_places:
        return "관련된 장소를 찾지 못했습니다."

    # 3. LLM으로 답변 생성
    print("[RAG] 필터링된 정보를 바탕으로 최종 답변을 생성합니다...")
    final_answer = _generate_answer_with_llm(search_query, top_places)
    #final_answer = ""
    return final_answer