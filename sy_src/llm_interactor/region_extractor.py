from dotenv import load_dotenv
import random
from openai import OpenAI
import os
import json

# --- 초기 설정 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))


def extract_region_from_query(user_query):
    """
    사용자 질문에서 LLM을 사용해 지역명 키워드 리스트를 추출합니다.
    """
    print("[LLM] 사용자 쿼리에서 지역명 키워드를 추출합니다...")
    
    # LLM에게 역할을 부여하고, 예시(Few-shot)를 통해 원하는 결과 형식을 명확히 알려줍니다.
    system_prompt = """
    당신은 사용자의 여행 관련 질문에서 '대한민국 행정구역' 키워드를 추출하는 AI 어시스턴트입니다.
    사용자의 질문을 분석하여, 주소 필터링에 사용할 수 있는 키워드 목록을 JSON 형식으로 반환해 주세요.
    결과는 반드시 {"regions": ["키워드1", "키워드2", ...]} 형태여야 합니다.
    
    - "전라도"는 "전북", "전남", "광주"로 해석합니다.
    - "경상도"는 "경북", "경남", "부산", "대구", "울산"으로 해석합니다.
    - "충청도"는 "충북", "충남", "대전", "세종"으로 해석합니다.
    - "서울 근교"는 "경기", "인천"으로 해석합니다.
    - 언급된 지역이 없으면 빈 리스트 []를 반환합니다.
    
    # 예시1
    사용자: "전라도 쪽으로 맛집 투어 가고 싶어"
    AI: {"regions": ["전북", "전남", "광주"]}

    # 예시2
    사용자: "강원도나 경주 쪽 바다 보고 싶다"
    AI: {"regions": ["강원", "경주"]}
    
    # 예시3
    사용자: "그냥 조용한 곳이면 돼"
    AI: {"regions": []}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        
        # 'regions' 키가 있고, 그 값이 리스트인지 확인
        if 'regions' in result and isinstance(result['regions'], list):
            #print(f"[LLM] 추출된 지역 키워드: {result['regions']}")
            return result['regions']
        else:
            #print("[LLM] 'regions' 키를 찾지 못했거나 형식이 리스트가 아닙니다.")
            return []
            
    except Exception as e:
        #print(f"[LLM] 지역명 추출 중 오류 발생: {e}")
        return []
