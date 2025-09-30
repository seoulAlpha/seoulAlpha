from dotenv import load_dotenv
import random
from openai import OpenAI
import os
import json
from kiwipiepy import Kiwi

# --- 초기 설정 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))
kiwi = Kiwi()

# 지역 이름 불러오기
file_path = "./data/korean_regions.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    regions = [line.strip() for line in f if line.strip()]

def extract_region_from_query(user_query, simple=True):
    """
    사용자 질문에서 LLM을 사용해 지역명 키워드 리스트를 추출합니다.
    """
    if simple:
        print("[Keyword] 사용자 쿼리에서 지역명 키워드를 추출합니다...")
        found_regions = set()

        if "서울 근교" in user_query or "수도권" in user_query:
            found_regions.update(["경기", "인천"])
        if "경기도" in user_query:
            found_regions.update(["경기", "인천"])
        if "경상도" in user_query:
            found_regions.update(["경북", "경남", "부산", "대구", "울산"])
        if "전라도" in user_query:
            found_regions.update(["전북", "전남", "광주"])
        if "충청도" in user_query:
            found_regions.update(["충북", "충남", "대전", "세종"])
        
        query_nouns = [token.form for token in kiwi.tokenize(user_query) 
            if token.tag in ['NNG', 'NNP']]
        print(query_nouns)
        for region in regions:
            if region in query_nouns:
                found_regions.add(region)
        
        return list(found_regions)
    
    else:
        print("[LLM] 사용자 쿼리에서 지역명 키워드를 추출합니다...")
        system_prompt = """
        당신은 사용자의 여행 관련 질문에서 '대한민국 행정구역' 키워드를 추출하는 AI 어시턴트입니다.
        사용자의 질문을 분석하여, 주소 필터링에 사용할 수 있는 키워드 목록을 JSON 형식으로 반환해 주세요.
        결과는 반드시 {"regions": ["키워드1", "키워드2", ...]} 형태여야 합니다.
        - "전라도"는 "전북", "전남", "광주"로 해석합니다.
        - "경상도"는 "경북", "경남", "부산", "대구", "울산"으로 해석합니다.
        - "충청도"는 "충북", "충남", "대전", "세종"으로 해석합니다.
        - "서울 근교"는 "경기", "인천"으로 해석합니다.
        - 언급된 지역이 없으면 빈 리스트 []를 반환합니다.
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
            if 'regions' in result and isinstance(result['regions'], list):
                return result['regions']
            else:
                return []
        except Exception as e:
            print(f"[LLM] 지역명 추출 중 오류 발생: {e}")
            return []

def update_region_keywords(query, state):
    prev_regions = state.get("region_keywords", [])
    new_regions = extract_region_from_query(query, simple=True)

    if new_regions:
        # 새 지역이 발견되면 업데이트
        if set(new_regions) != set(prev_regions):
            state["region_keywords"] = new_regions
    # 새 지역이 없으면 그대로 유지
    return state