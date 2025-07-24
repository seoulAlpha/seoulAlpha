import os
import json

os.environ['HF_HOME'] = 'D:/huggingface_cache'

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from unidecode import unidecode


# --- 1. 설정 ---
# 원본 JSON 파일들이 저장된 디렉터리
JSON_DIR = 'data/json'
# 사용할 한국어 임베딩 모델
MODEL_NAME = 'jhgan/ko-sbert-nli'
# 최종 결과물(FAISS DB)이 저장될 폴더 경로
OUTPUT_DIR = 'faiss_integrated_dbs'

# 스크래핑한 행정구역 및 카테고리 정보 (스크래핑 스크립트와 동일하게 유지)
CATEGORIES = ['카페', '식당', '관광명소', '문화재', '레저', '쇼핑', '숙박']
CATEGORY_TRANSLATIONS = {
    "카페": "cafe",
    "식당": "restaurant",
    "숙박": "accommodation",
    "관광명소": "attraction",
    "문화재": "heritage",
    "레저": "leisure",
    "쇼핑": "shopping"
}


ADMIN_AREAS = {
    # Batch1
    "서울특별시": ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"],
    "부산광역시": ["강서구", "금정구", "기장군", "남구", "동구", "동래구", "부산진구", "북구", "사상구", "사하구", "서구", "수영구", "연제구", "영도구", "중구", "해운대구"],
    "대구광역시": ["군위군", "남구", "달서구", "달성군", "동구", "북구", "서구", "수성구", "중구"],
    "인천광역시": ["강화군", "계양구", "남동구", "동구", "미추홀구", "부평구", "서구", "연수구", "옹진군", "중구"],

    # Batch2
    "광주광역시": ["광산구", "남구", "동구", "북구", "서구"],
    "대전광역시": ["대덕구", "동구", "서구", "유성구", "중구"],
    "울산광역시": ["남구", "동구", "북구", "울주군", "중구"],
    "세종특별자치시": ["세종시"],
    "경기도": ["수원시", "용인시", "고양시", "성남시", "화성시", "부천시", "남양주시", "안산시", "평택시", "안양시", "시흥시", "파주시", "김포시", "의정부시", "광주시", "하남시", "오산시", "이천시", "안성시", "의왕시", "양주시", "구리시", "포천시", "동두천시", "과천시", "여주시", "양평군", "가평군", "연천군"],
    "강원특별자치도": ["춘천시", "원주시", "강릉시", "동해시", "태백시", "속초시", "삼척시", "홍천군", "횡성군", "영월군", "평창군", "정선군", "철원군", "화천군", "양구군", "인제군", "고성군", "양양군"],

    # Batch3
    "충청북도": ["청주시", "충주시", "제천시", "보은군", "옥천군", "영동군", "증평군", "진천군", "괴산군", "음성군", "단양군"],
    "충청남도": ["천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시", "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군"],
    "전북특별자치도": ["전주시", "익산시", "군산시", "정읍시", "남원시", "김제시", "완주군", "진안군", "무주군", "장수군", "임실군", "순창군", "고창군", "부안군"],
    "전라남도": ["목포시", "여수시", "순천시", "나주시", "광양시", "담양군", "곡성군", "구례군", "고흥군", "보성군", "화순군", "장흥군", "강진군", "해남군", "영암군", "무안군", "함평군", "영광군", "장성군", "완도군", "진도군", "신안군"],
    "경상북도": ["포항시", "경주시", "김천시", "안동시", "구미시", "영주시", "영천시", "상주시", "문경시", "경산시", "의성군", "청송군", "영양군", "영덕군", "청도군", "고령군", "성주군", "칠곡군", "예천군", "봉화군", "울진군", "울릉군"],
    "경상남도": ["창원시", "진주시", "통영시", "사천시", "김해시", "밀양시", "거제시", "양산시", "의령군", "함안군", "창녕군", "고성군", "남해군", "하동군", "산청군", "함양군", "거창군", "합천군"],
    "제주특별자치도": ["제주시", "서귀포시"]
}

def create_faiss_db_for_province(province, province_ascii, cities, category, category_ascii, model, device):
    print(f"\n{'='*20}\n[시작] {province} - {category}\n{'='*20}")
    
    province_category_data = []
    province_short = province.replace("특별시", "").replace("광역시", "").replace("특별자치도", "").replace("특별자치시", "")

    for city in cities:
        filename = f"{city}_{category}.jsonl"
        filepath = os.path.join(f'{JSON_DIR}/{province_short}', filename)

        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                # 1. 빈 리스트를 먼저 만듭니다.
                data = []
                # 2. 파일의 각 줄을 순회합니다.
                for line in f:
                    # 3. 각 줄(JSON 문자열)을 딕셔너리로 변환하여 리스트에 추가합니다.
                    #    json.load()가 아닌 json.loads() (s가 붙음)를 사용합니다.
                    data.append(json.loads(line))


                for place in data:
                    place['region_l1'] = province
                    place['region_l2'] = city
                province_category_data.extend(data)
    
    if not province_category_data:
        print(f"-> 처리할 데이터 없음. 건너뜁니다.")
        return

    print(f"-> 총 {len(province_category_data)}개의 장소 데이터 로드 완료.")

    corpus = []
    for place in province_category_data:
        comments_text = " ".join(place.get('processed_sentences', []))
        ai_summary_text = place.get('ai_summary', '')

        # '이름', '요약', '후기' 정보를 모두 합쳐 최종 텍스트를 만듭니다.
        text_to_embed = f"이름: {place.get('name', '')}\n요약: {ai_summary_text}\n후기: {comments_text}"
        #text_to_embed = f"이름: {place.get('name', '')}\n후기: {comments_text}"
        corpus.append(text_to_embed)

    print(f"-> 텍스트 벡터 변환 중...")
    vectors = model.encode(corpus, batch_size=32, show_progress_bar=True, convert_to_numpy=True, device=device)

    if vectors.shape[0] == 0:
        print(f"-> 임베딩 결과 없음. 건너뜁니다.")
        return

    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors)

    # [수정] 파일명 생성 시 ASCII 변환된 이름 사용
    faiss_filename = f"faiss_{category_ascii}.index"
    faiss_path = os.path.join(f'{OUTPUT_DIR}/{province_ascii}', faiss_filename)

    faiss.write_index(index, faiss_path)

    metadata_filename = f"metadata_{category_ascii}.json"
    metadata_path = os.path.join(f'{OUTPUT_DIR}/{province_ascii}', metadata_filename)
    with open(metadata_path.replace(".json", ".jsonl"), 'w', encoding='utf-8') as f:
        for item in province_category_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
    
    print(f"-> [완료] {province} - {category} DB 구축 완료!")
    print(f"   - Index: {faiss_filename} ({index.ntotal}개 벡터)")
    print(f"   - Metadata: {metadata_filename}")


if __name__ == "__main__":
    print("FAISS DB 구축 파이프라인을 시작합니다.")
    
    print(f"임베딩 모델 '{MODEL_NAME}' 로딩 중...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 장치: {device}")
    embedding_model = SentenceTransformer(MODEL_NAME, device=device)

    for province_name, city_list in tqdm(ADMIN_AREAS.items(), desc="전체 진행률"):
        for category_name in CATEGORIES:
            # [수정] unidecode를 사용하여 파일명에 쓸 ASCII 문자열 자동 생성
            province_ascii = unidecode(province_name).lower().replace(" ", "")
            category_ascii = CATEGORY_TRANSLATIONS.get(category_name, unidecode(category_name).lower().replace(" ", ""))
            
            if not os.path.exists(f'{OUTPUT_DIR}/{province_ascii}'):
                os.makedirs(f'{OUTPUT_DIR}/{province_ascii}')
                print(f"결과물 저장 폴더 '{OUTPUT_DIR}/{province_ascii}' 생성 완료.")
            

            create_faiss_db_for_province(
                province_name, province_ascii, 
                city_list, 
                category_name, category_ascii, 
                embedding_model, device
            )

    print("\n\n🎉🎉🎉 모든 DB 구축 작업이 완료되었습니다. 🎉🎉🎉")