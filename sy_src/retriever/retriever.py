import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Hugging Face 캐시 경로 설정 (필요시 주석 해제)
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# --- 설정 ---
# 임베딩 모델
MODEL_NAME = 'jhgan/ko-sbert-nli'
# 답변 생성 LLM 모델
LLM_MODEL_NAME = 'gpt-4o-mini'

# 파일 경로
OUTPUT_DIR = 'data/faiss/faiss_merged_output'
INDEX_FILE = f'{OUTPUT_DIR}/merged.index'
METADATA_FILE = f'{OUTPUT_DIR}/merged_metadata.jsonl'

# 검색할 결과의 수
TOP_K = 10

import os
from dotenv import load_dotenv

# 스크립트 시작 부분에서 .env 파일 로드
load_dotenv()

# --- 설정 끝 ---


def load_resources():
    """
    검색에 필요한 모델, 인덱스, 메타데이터를 로드하는 함수
    """
    print("1. 리소스를 로딩합니다...")
    
    print(f"  - 임베딩 모델({MODEL_NAME}) 로딩 중...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"  - FAISS 인덱스({INDEX_FILE}) 로딩 중...")
    index = faiss.read_index(INDEX_FILE)
    
    print(f"  - 메타데이터({METADATA_FILE}) 로딩 및 매핑 중...")
    metadata_map = {}
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            meta = json.loads(line)
            metadata_map[meta['vector_id']] = meta
            
    print("✅ 리소스 로딩 완료!")
    return model, index, metadata_map


def retrieve_places(query, model, index, metadata_map, k):
    """
    주어진 쿼리를 기반으로 가장 유사한 장소 K개를 검색하는 함수
    """
    print("\n2. 쿼리를 벡터로 변환합니다...")
    query_vector = model.encode([query])
    
    print(f"\n3. FAISS 인덱스에서 유사도 높은 Top {k}개를 검색합니다...")
    distances, ids = index.search(query_vector.astype('float32'), k)
    
    retrieved_ids = ids[0]
    
    print("\n4. 검색된 ID를 바탕으로 메타데이터를 조회합니다...")
    results = []
    for vector_id in retrieved_ids:
        if vector_id in metadata_map:
            results.append(metadata_map[vector_id])
        else:
            print(f"  - ⚠️ 경고: ID {vector_id}에 해당하는 메타데이터를 찾을 수 없습니다.")

    return results


# ====================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼ LLM 답변 생성 함수 추가 ▼▼▼▼▼▼▼▼▼▼▼▼▼
# ====================================================

def generate_answer_with_llm(query, retrieved_places):
    """
    검색된 장소 정보를 기반으로 LLM을 사용해 자연스러운 답변을 생성합니다.
    """
    print("\n5. LLM으로 최종 답변을 생성합니다...")

    # 1. LLM에게 전달할 정보(Context)를 정리합니다.
    context = ""
    for i, place in enumerate(retrieved_places[:5]): # 너무 많은 정보를 주지 않기 위해 상위 5개만 사용
        context += f"--- 장소 정보 {i+1} ---\n"
        context += f"이름: {place.get('name', '정보 없음')}\n"
        context += f"AI 요약: {place.get('ai_summary', '정보 없음')}\n"
        processed_sentences = place.get('processed_sentences', [])
        context += "주요 특징 및 후기:\n"
        for sentence in processed_sentences:
            context += f"- {sentence}\n"
        context += "\n"

    # 2. LLM에 전달할 프롬프트 구성
    system_prompt = "당신은 사용자의 질문에 가장 적합한 장소를 추천해주는 유용한 어시스턴트입니다."
    
    user_prompt = f"""
    아래 '장소 정보'만을 바탕으로 사용자의 질문에 대한 답변을 생성해 주세요.

    [지시사항]
    1. 검색된 장소 중에서 질문과 가장 관련성이 높은 2~3곳을 추천해 주세요.
    2. 각 장소를 추천하는 이유를 'AI 요약'과 '주요 특징 및 후기'를 근거로 구체적으로 설명해 주세요.
    3. 'processed_sentences'에 있는 실제 후기를 인용하여 답변하면 신뢰도를 높일 수 있습니다.
    4. 친절하고 자연스러운 말투로 답변해 주세요.

    --- 장소 정보 ---
    {context}
    --- 사용자의 질문 ---
    {query}
    """

    try:
        # 3. OpenAI API 호출
        client = OpenAI() # 환경변수(OPENAI_API_KEY)를 자동으로 읽음
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
        return f"❌ LLM 답변 생성 중 오류가 발생했습니다: {e}"


if __name__ == '__main__':
    # 1~4단계: 리소스 로딩 및 장소 검색
    embedding_model, faiss_index, meta_map = load_resources()

    # 2. 한국어 질문 리스트
    queries = [
        "전통적인 한국의 미를 느낄 수 있는 장소를 찾고 있어요. 고궁이나 민속촌 같은 곳이면 좋겠어요.",
        "요즘 인스타그램에서 유행하는 힙한 카페 좀 추천해 줄래?",
        "아름다운 자연을 즐기고 싶어요. 유명한 산이나 해변이 있나요?",
        "전형적인 관광지 말고, 좀 독특하고 특별한 경험을 할 수 있는 곳은 어디일까요?",
        "커피 한잔하면서 야경 감상하기 좋은 장소 추천해 줘."
    ]

    # 3. 질문 리스트를 순회하며 전체 RAG 파이프라인 실행
    for user_query in queries:
        print("\n" + "#"*70)
        print(f"🗣️  NEW QUERY: {user_query}")
        print("#"*70)

        # 1단계: 장소 검색
        top_places = retrieve_places(
            query=user_query,
            model=embedding_model,
            index=faiss_index,
            metadata_map=meta_map,
            k=TOP_K
        )

        # 검색된 장소 목록 간략히 출력
        print("\n" + "="*50)
        print(f"✅ 검색된 Top {len(top_places)} 장소 목록 (의미순)")
        print("="*50)
        for i, place in enumerate(top_places):
            name = place.get('name', 'N/A')
            address = place.get('address', 'N/A')
            print(f"  RANK {i+1}: {name} | {address}")
        print("="*50)

        # 2단계: LLM으로 답변 생성
        llm_answer = generate_answer_with_llm(user_query, top_places)
        
        # 최종 답변 출력
        print("\n" + "="*50)
        print(f"🤖 LLM 추천 요약: '{user_query}'")
        print("="*50)
        print(llm_answer)
        print("="*50 + "\n\n")