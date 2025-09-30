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
file_path = "./data/korean_regions.json"
with open(file_path, 'r', encoding='utf-8') as f:
    regions = json.load(f)

# 예외 매핑 룰
except_mapping_rules = {
    "전라도": {"전북", "전남", "광주"},
    "경상도": {"경북", "경남", "부산", "대구", "울산"},
    "충청도": {"충북", "충남", "대전", "세종"},
    "수도권": {"경기", "인천"},
    "서울 근교": {"경기", "인천"},
    "경기도": {"경기", "인천"}
}

def extract_region_from_query(user_query, simple=True):
    """
    사용자 질문에서 LLM을 사용해 지역명 키워드 리스트를 추출합니다.
    """
    if simple:
        print("[Keyword] 사용자 쿼리에서 지역명 키워드를 추출합니다...")        
        query_nouns = [token.form for token in kiwi.tokenize(user_query) 
            if token.tag in ['NNG', 'NNP']]
        print(query_nouns)
        specific_regions = {region for region in regions if region in query_nouns}

        # 2. '전라도', '경상도' 등 광역 키워드로 인해 추가될 대분류 지역을 찾습니다.
        potential_broad_regions = set()
        for broad_keyword, provinces in except_mapping_rules.items():
            if broad_keyword in user_query:
                potential_broad_regions.update(provinces)

        # 3. 추출된 소분류 지역이 어떤 대분류에 속하는지 확인하여, 제외할 대분류를 결정합니다.
        broad_regions_to_discard = set()
        for region in specific_regions:
            province = regions.get(region) # ex: '순천' -> '전남'
            if not province:
                continue

            # 이 소분류(region)가 속한 대분류(province)가 광역 키워드로 인해 추가될 예정이었다면,
            # 해당 광역 키워드에 해당하는 모든 대분류를 제외 목록에 추가합니다.
            for broad_keyword, provinces_in_map in except_mapping_rules.items():
                if province in provinces_in_map and broad_keyword in user_query:
                    broad_regions_to_discard.update(provinces_in_map)
                    
        # 4. 최종 결과를 조합합니다.
        # (추출된 모든 지역 + 광역 키워드 지역) - 제외할 광역 지역
        final_regions = (specific_regions.union(potential_broad_regions)) - broad_regions_to_discard
        
        return list(final_regions)
    
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