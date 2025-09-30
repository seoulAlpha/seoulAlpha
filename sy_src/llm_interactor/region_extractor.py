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

EXCEPT_RULE_MAP = {
    "전라도": ["전북특별자치도", "전라남도", "광주광역시"],
    "경상도": ["경상북도", "경상남도", "부산광역시", "대구광역시", "울산광역시"],
    "충청도": ["충청북도", "충청남도", "대전광역시", "세종특별자치시"],
    "수도권": ["경기도", "인천광역시"],
    "서울 근교": ["경기도", "인천광역시"],
    "경기도": ["경기도", "인천광역시"]
}

PROVINCE_DISTRICT_MAP = {
    "서울특별시": ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"],
    "부산광역시": ["강서구", "금정구", "기장군", "남구", "동구", "동래구", "부산진구", "북구", "사상구", "사하구", "서구", "수영구", "연제구", "영도구", "중구", "해운대구"],
    "대구광역시": ["군위군", "남구", "달서구", "달성군", "동구", "북구", "서구", "수성구", "중구"],
    "인천광역시": ["강화군", "계양구", "남동구", "동구", "미추홀구", "부평구", "서구", "연수구", "옹진군", "중구"],
    "광주광역시": ["광산구", "남구", "동구", "북구", "서구"],
    "대전광역시": ["대덕구", "동구", "서구", "유성구", "중구"],
    "울산광역시": ["남구", "동구", "북구", "울주군", "중구"],
    "세종특별자치시": [], 
    "경기도": ["수원시", "용인시", "고양시", "성남시", "화성시", "부천시", "남양주시", "안산시", "평택시", "안양시", "시흥시", "파주시", "김포시", "의정부시", "광주시", "하남시", "오산시", "이천시", "안성시", "의왕시", "양주시", "구리시", "포천시", "동두천시", "과천시", "여주시", "양평군", "가평군", "연천군"],
    "강원특별자치도": ["춘천시", "원주시", "강릉시", "동해시", "태백시", "속초시", "삼척시", "홍천군", "횡성군", "영월군", "평창군", "정선군", "철원군", "화천군", "양구군", "인제군", "고성군", "양양군"],
    "충청북도": ["청주시", "충주시", "제천시", "보은군", "옥천군", "영동군", "증평군", "진천군", "괴산군", "음성군", "단양군"],
    "충청남도": ["천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시", "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군"],
    "전북특별자치도": ["전주시", "익산시", "군산시", "정읍시", "남원시", "김제시", "완주군", "진안군", "무주군", "장수군", "임실군", "순창군", "고창군", "부안군"],
    "전라남도": ["목포시", "여수시", "순천시", "나주시", "광양시", "담양군", "곡성군", "구례군", "고흥군", "보성군", "화순군", "장흥군", "강진군", "해남군", "영암군", "무안군", "함평군", "영광군", "장성군", "완도군", "진도군", "신안군"],
    "경상북도": ["포항시", "경주시", "김천시", "안동시", "구미시", "영주시", "영천시", "상주시", "문경시", "경산시", "의성군", "청송군", "영양군", "영덕군", "청도군", "고령군", "성주군", "칠곡군", "예천군", "봉화군", "울진군", "울릉군"],
    "경상남도": ["창원시", "진주시", "통영시", "사천시", "김해시", "밀양시", "거제시", "양산시", "의령군", "함안군", "창녕군", "고성군", "남해군", "하동군", "산청군", "함양군", "거창군", "합천군"],
    "제주특별자치도": ["제주시", "서귀포시"]
}

SHORT_TO_FULL_PROVINCE_NAME = {
    "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시",
    "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
    "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도",
    "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
    "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도",
    "경남": "경상남도", "제주": "제주특별자치도"
}

# 1-3. 시/군/구 이름으로 시/도를 빠르게 찾기 위한 역방향 맵 (코드 실행 시 자동으로 생성됨)
DISTRICT_TO_PROVINCE_MAP = {
    district: province
    for province, districts in PROVINCE_DISTRICT_MAP.items()
    for district in districts
}

ALL_KNOWN_REGIONS = set(PROVINCE_DISTRICT_MAP.keys()) | set(DISTRICT_TO_PROVINCE_MAP.keys())

def extract_region_from_query(user_query):
    """
    사용자 쿼리에서 지역 정보를 추출합니다.
    """
    print("[Keyword] 사용자 쿼리에서 지역명 키워드를 추출합니다...")
    query_nouns = {token.form for token in kiwi.tokenize(user_query) if token.tag in ['NNG', 'NNP']}
    
    found_regions = set()
    for noun in query_nouns:  
        for known_region in ALL_KNOWN_REGIONS:
            if noun in known_region:
                found_regions.add(known_region)

    specific_regions_found = list(found_regions)
    # 구체적인 지역 명사가 있으면 상세 분석
    if specific_regions_found:
        return format_regions(specific_regions_found)
    
    # 구체적인 지역 명사가 없으면 포괄적인 키워드 검색
    else:
        print("-> 구체적인 지역이 없어 포괄 키워드를 검색합니다.")
        for keyword, provinces in EXCEPT_RULE_MAP.items():
            if keyword in user_query:
                return [{"region_l1": province, "region_l2": ""} for province in provinces]
    # 경상도 맛집을 추천해줘. 첫 여행이야
    
def format_regions(region_list):
    province = None
    district = None
    for region in region_list:
        if region.endswith(('구', '시', '군')):
            if region in DISTRICT_TO_PROVINCE_MAP:
                district = region
                province = DISTRICT_TO_PROVINCE_MAP[region]
        full_name = SHORT_TO_FULL_PROVINCE_NAME.get(region)
        if full_name and full_name in PROVINCE_DISTRICT_MAP:
            if province is None:
                province = full_name
    result_dict = {}
    if province:
        result_dict["region_l1"] = province
    if district:
        result_dict["region_l2"] = district
    return result_dict

def update_region_keywords(query, state):
    prev_regions = state.get("region_keywords", [])
    new_regions = extract_region_from_query(query)
    new_list = []
    prev_list= []
    for i in new_regions:
        for a,b in i.items():
            new_list.append(a)
            new_list.append(b)
    print(new_list)

    for i in prev_regions:
        for a,b in i.items():
            prev_list.append(a)
            prev_list.append(b)
    print(prev_list)

    if new_regions:
    # 새 지역이 발견되면 업데이트
        if set(new_list) != set(prev_list):
            state["region_keywords"] = new_regions
    # 새 지역이 없으면 그대로 유지
    return state