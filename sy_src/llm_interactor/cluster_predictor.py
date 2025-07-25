# cluster_predictor.py

import joblib
import pandas as pd
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# --- 초기 설정 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

# 미리 정의된 클러스터별 프로필 (실제 내용으로 채워야 합니다)
CLUSTER_PROFILES = {
    0: "문화, 역사, 자연 탐방을 주목적으로 가을에 한국을 재방문하는 중장년층 여성 여행객. 긴 체류 기간 동안 서울과 여러 지방(경기, 강원, 경상)을 함께 방문하며, 매우 알뜰하게 소비하는 경향이 있음.",
    1: "한국을 처음 방문한 20-30대 영어권/유럽 남성 여행객. 짧은 기간 동안 서울에만 머무르며 음식과 미식 탐방에 가장 큰 관심을 두고 여행함. 숙박비에 비교적 높은 예산을 사용함.",
    2: "한국을 처음 방문한 20대 젊은 여성 여행객. 짧은 기간 서울에 머물며 음식, 쇼핑 등 모든 분야에서 압도적인 소비력을 보여주는 럭셔리 여행을 즐김.",
    3: "쇼핑과 맛집 탐방을 목적으로 서울을 자주 재방문하는 동아시아 여성 여행객. 매우 짧은 기간 머물며 여행 목적을 집중적으로 달성하고, 식비에 지출 비중이 매우 높음. 문화나 자연보다 쇼핑과 미식에 관심이 집중됨.",
    4: "한국 여행 경험이 풍부한 30-50대 남성 재방문객. 서울뿐만 아니라 전국을 여행하며, 특히 다양한 지역의 음식을 즐기는 미식 활동에 관심이 매우 높음.",
    5: "한국을 처음 방문하는 서구권 및 동남아 출신 남성. 긴 기간 동안 머무르며 서울을 넘어 지방, 특히 경상도 지역의 자연 경관과 문화 유산을 깊이 있게 탐험하는 것에 관심이 압도적으로 높음. 예산은 비교적 적게 사용함.",
    6: "한국을 처음 방문하는 서구권 및 동남아 출신 여성. 긴 기간 동안 지방, 특히 경상도를 여행하며 한국의 자연 경관과 문화 유산에 매우 높은 만족도와 깊은 감명을 느낌. 재방문 의향도 높은 이상적인 탐방형 여행객."
}

# --- 모델 및 데이터 로드 ---
try:
    preprocessor = joblib.load('./models/preprocessor.joblib')
    pca = joblib.load('./models/pca.joblib')
    kmeans = joblib.load('./models/kmeans.joblib')
    imputation_base_data = pd.read_csv('./models/imputation_base_data.csv', encoding='utf-8-sig')
except FileNotFoundError:
    print("모델 파일이 없습니다. train_model.py를 먼저 실행해주세요.")
    preprocessor, pca, kmeans, imputation_base_data = None, None, None, None

# 변수 정의
categorical_cols = ['country', 'gender', 'age', 'revisit_indicator', 'visit_local_indicator', 'planned_activity']
numerical_cols = ['stay_duration', 'accommodation_percent', 'food_percent', 'shopping_percent', 'food', 'landscape', 'heritage', 'language', 'safety', 'budget', 'accommodation', 'transport', 'navigation']
used_variables = categorical_cols + numerical_cols


def query_llm_for_variables(user_query, use_prompt=True, use_fewshot=True):
    prompt_parts = []

    if use_prompt:
        with open("custom_prompt.txt", "r", encoding="utf-8") as f:
            custom_prompt = f.read()
            prompt_parts.append(custom_prompt)

    if use_fewshot:
        with open("custom_few_shot_learning.txt", "r", encoding="utf-8") as f:
            few_shot_examples = f.read()
            prompt_parts.append(few_shot_examples)
        
    full_prompt = "\n\n".join(prompt_parts)

    messages = [
        {"role": "system", "content": full_prompt},
        {"role": "user", "content": user_query}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_format={"type": "json_object"} # tsy 추가: JSON 응답 형식을 강제
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print("[파싱 실패]", e)
        return {}

def impute_with_user_subgroup(user_input_dict, df_base=imputation_base_data):
    known_info = {k: v for k, v in user_input_dict.items() if v is not None}
    filtered_df = df_base.copy()
    for key, val in known_info.items():
        if key in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[key].astype(str) == str(val)]
    imputed = {}
    for var in used_variables:
        if user_input_dict.get(var) is not None:
            imputed[var] = user_input_dict[var]
        else:
            if not filtered_df.empty:
                if var in numerical_cols: imputed[var] = filtered_df[var].mean()
                elif var in categorical_cols: imputed[var] = filtered_df[var].mode().iloc[0]
            else:
                if var in numerical_cols: imputed[var] = df_base[var].mean()
                elif var in categorical_cols: imputed[var] = df_base[var].mode().iloc[0]
    return imputed

def predict_cluster_from_query(user_query):
    variable_dict = query_llm_for_variables(user_query, use_prompt=True, use_fewshot=True)
    if not variable_dict: return None
    completed_input = impute_with_user_subgroup(variable_dict)
    df = pd.DataFrame([completed_input])
    for col in categorical_cols:
        if col in df.columns: df[col] = df[col].astype(str)
    for col in numerical_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    try:
        X_processed = preprocessor.transform(df)
        X_pca = pca.transform(X_processed)
        return kmeans.predict(X_pca)[0]
    except Exception as e:
        print(f"[클러스터 예측 실패] {e}")
        return None

# --- 대표 실행 함수 ---
def get_user_cluster(user_query):
    # 각 모델이 None인지, DataFrame이 비어있는지를 명시적으로 확인
    if preprocessor is None or pca is None or kmeans is None or imputation_base_data.empty:
        return None, None
        
    cluster_label = predict_cluster_from_query(user_query)
    
    if cluster_label is not None:
        profile = CLUSTER_PROFILES.get(cluster_label, "정의되지 않은 클러스터입니다.")
        return cluster_label, profile
    else:
        return None, None
    
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
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        
        # 'regions' 키가 있고, 그 값이 리스트인지 확인
        if 'regions' in result and isinstance(result['regions'], list):
            print(f"✅ [LLM] 추출된 지역 키워드: {result['regions']}")
            return result['regions']
        else:
            print("⚠️ [LLM] 'regions' 키를 찾지 못했거나 형식이 리스트가 아닙니다.")
            return []
            
    except Exception as e:
        print(f"❌ [LLM] 지역명 추출 중 오류 발생: {e}")
        return []
