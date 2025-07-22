import json
import numpy as np
import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'

import faiss
import torch
from sentence_transformers import SentenceTransformer

# --- 1. 설정 ---
# 이전에 생성된 DB 파일들이 저장된 디렉터리
DB_DIR = 'faiss_integrated_dbs'
# DB 생성 시 사용했던 것과 "반드시 동일한" 모델을 지정해야 합니다.
MODEL_NAME = 'jhgan/ko-sbert-nli'

# 로드할 DB의 지역과 카테고리 (파일명에 맞춰 영문으로)
PROVINCE_ASCII = 'gwangjugwangyeogsi'
CATEGORY_ASCII = 'kape'


if __name__ == "__main__":
    # --- 2. 임베딩 모델 및 DB 파일 로드 ---
    print("FAISS DB와 모델을 로드합니다...")
    
    # 모델 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # 파일 경로 조합
    index_path = os.path.join(DB_DIR, f"faiss_{PROVINCE_ASCII}_{CATEGORY_ASCII}.index")
    metadata_path = os.path.join(DB_DIR, f"metadata_{PROVINCE_ASCII}_{CATEGORY_ASCII}.json")

    # 파일 존재 여부 확인
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print(f"오류: '{index_path}' 또는 '{metadata_path}' 파일을 찾을 수 없습니다.")
        print("DB 구축 스크립트를 먼저 실행했는지, 파일명이 올바른지 확인하세요.")
        exit()

    # FAISS 인덱스 로드
    index = faiss.read_index(index_path)
    
    # 메타데이터 로드
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"로드 완료! '{PROVINCE_ASCII}_{CATEGORY_ASCII}' DB에는 총 {index.ntotal}개의 장소가 있습니다.")
    
    
    # --- 3. 검색 실행 ---
    # 사용자 질문(쿼리)
    query = "조용하고 공부하기 좋은 곳"
    
    # 상위 몇 개를 가져올지 결정
    k = 20
    
    print(f"\n[검색어]: '{query}'")
    
    # 1. 쿼리를 벡터로 변환
    #    검색할 벡터는 2차원 배열이어야 하므로 reshape
    query_vector = model.encode([query], convert_to_numpy=True, device=device)
    
    # 2. FAISS 인덱스에서 검색
    #    D: 각 벡터까지의 거리(distance), I: 벡터의 인덱스(ID)
    distances, indices = index.search(query_vector, k)
    
    
    # --- 4. 검색 결과 출력 ---
    print("\n[검색 결과]:")
    if not metadata:
        print("메타데이터가 비어있어 결과를 표시할 수 없습니다.")
    else:
        for i in range(k):
            # 검색된 벡터의 인덱스(ID)
            result_index = indices[0][i]
            # 해당 인덱스의 원본 데이터
            result_data = metadata[result_index]
            # 쿼리 벡터와의 거리 (L2 Distance)
            distance = distances[0][i]
            
            print(f"\n--- {i+1}위 (Distance: {distance:.4f}) ---")
            print(f"  이름: {result_data['name']}")
            print(f"  주소: {result_data['address']}")
            print(f"  지역: {result_data['region_l1']} {result_data['region_l2']}")
            # 후기 중 처음 2개만 샘플로 출력
            comments_sample = result_data.get('processed_sentences', [])[:5]
            print(f"  후기 샘플: {comments_sample}")
            ai_summary = result_data.get('ai_summary', [])
            print(f"  ai 요약: {ai_summary}")