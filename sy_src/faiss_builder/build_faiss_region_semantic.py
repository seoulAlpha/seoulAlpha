import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# --- 설정 ---
# 기존 장소 임베딩과 동일한 모델 사용
MODEL_NAME = 'jhgan/ko-sbert-nli'

# 원본 데이터 경로
INPUT_METADATA_FILE = 'data/faiss/faiss_merged_output/merged_metadata.jsonl'

# 생성될 결과물 경로
OUTPUT_DIR = 'data/faiss/region_db'
OUTPUT_INDEX_FILE = f'{OUTPUT_DIR}/faiss_region_semantic.index'
OUTPUT_METADATA_FILE = f'{OUTPUT_DIR}/metadata_region_semantic.jsonl'

# --- 설정 끝 ---

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_region_documents():
    """원본 메타데이터에서 지역별로 문서를 그룹화합니다."""
    print("1. 지역별 대표 문서 생성을 시작합니다...")
    region_docs = {}
    
    with open(INPUT_METADATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            meta = json.loads(line)
            region_name = meta.get('region_l2')
            
            if not region_name:
                continue
            
            # 지역별로 텍스트를 담을 리스트 초기화
            if region_name not in region_docs:
                region_docs[region_name] = []
            
            # ai_summary와 processed_sentences 텍스트를 추가
            if meta.get('ai_summary'):
                region_docs[region_name].append(meta['ai_summary'])
            if meta.get('processed_sentences'):
                region_docs[region_name].extend(meta['processed_sentences'])
                
    print(f"✅ 총 {len(region_docs)}개 지역의 대표 문서를 그룹화했습니다.")
    return region_docs


def main():
    region_docs = create_region_documents()
    
    print("\n2. 임베딩 모델을 로드합니다...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("\n3. 각 지역 대표 문서를 임베딩합니다...")
    region_names = []
    region_vectors = []
    
    for region_name, texts in region_docs.items():
        # 한 지역의 모든 텍스트를 하나의 큰 문자열로 합침
        full_document = ". ".join(texts)
        if not full_document:
            continue
        
        print(f"  - '{region_name}' 임베딩 중...")
        vector = model.encode(full_document)
        
        region_names.append(region_name)
        region_vectors.append(vector)
    
    # 리스트를 numpy 배열로 변환
    region_vectors = np.array(region_vectors).astype('float32')
    
    # FAISS 인덱스 생성 및 벡터 추가
    print("\n4. FAISS 인덱스를 생성하고 저장합니다...")
    dimension = region_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(region_vectors)
    
    faiss.write_index(index, OUTPUT_INDEX_FILE)
    print(f"  - FAISS 인덱스 저장 완료: {OUTPUT_INDEX_FILE}")
    
    # 메타데이터 생성 및 저장
    print("\n5. 지역 메타데이터를 생성하고 저장합니다...")
    with open(OUTPUT_METADATA_FILE, 'w', encoding='utf-8') as f:
        for i, region_name in enumerate(region_names):
            # 인덱스 순서(i)가 곧 벡터 ID가 됨
            meta = {'region_id': i, 'region_name': region_name}
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
            
    print(f"  - 지역 메타데이터 저장 완료: {OUTPUT_METADATA_FILE}")
    print("\n🎉 의미 기반 '유사 지역' DB 생성이 완료되었습니다!")


if __name__ == '__main__':
    main()