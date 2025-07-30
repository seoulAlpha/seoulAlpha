# cluster_predictor.py

import joblib
import pandas as pd
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import random

# --- 초기 설정 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

CLUSTER_PROFILES = {
    0: "문화, 역사, 자연 탐방을 주목적으로 가을에 한국을 재방문하는 여행객. 긴 체류 기간 동안 서울과 여러 지방(경기, 강원, 경상)을 함께 방문하며, 매우 알뜰하게 소비하는 경향이 있음.",
    1: "한국을 처음 방문한 여행객. 짧은 기간 동안 서울에만 머무르며 음식과 미식 탐방에 가장 큰 관심을 두고 여행함. 숙박비에 비교적 높은 예산을 사용함.",
    2: "한국을 처음 방문한 여행객. 짧은 기간 서울에 머물며 음식, 쇼핑 등 모든 분야에서 압도적인 소비력을 보여주는 럭셔리 여행을 즐김.",
    3: "쇼핑과 맛집 탐방을 목적으로 서울을 자주 재방문하는 여행객. 매우 짧은 기간 머물며 여행 목적을 집중적으로 달성하고, 식비에 지출 비중이 매우 높음. 문화나 자연보다 쇼핑과 미식에 관심이 집중됨.",
    4: "한국 여행 경험이 풍부한 재방문객. 서울뿐만 아니라 전국을 여행하며, 특히 다양한 지역의 음식을 즐기는 미식 활동에 관심이 매우 높음.",
    5: "한국을 처음 방문하는 여행객. 긴 기간 동안 머무르며 서울을 넘어 지방, 특히 경상도 지역의 자연 경관과 문화 유산을 깊이 있게 탐험하는 것에 관심이 압도적으로 높음. 예산은 비교적 적게 사용함.",
    6: "한국을 처음 방문하는 여행객. 긴 기간 동안 지방, 특히 경상도를 여행하며 한국의 자연 경관과 문화 유산에 매우 높은 만족도와 깊은 감명을 느낌. 재방문 의향도 높은 이상적인 탐방형 여행객."
}
# --- 모델 및 데이터 로드 ---
try:
    preprocessor = joblib.load('./models/preprocessor.joblib')
    pca = joblib.load('./models/pca.joblib')
    kmeans = joblib.load('./models/kmeans.joblib')
    imputation_base_data = pd.read_csv('./models/imputation_base_data.csv', encoding='utf-8-sig')
    with open('./models/variable_weights.json', 'r', encoding='utf-8') as f:
        VARIABLE_WEIGHTS = json.load(f)
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
        with open("data/prompt/custom_prompt_eng.txt", "r", encoding="utf-8") as f:
            custom_prompt = f.read()
            prompt_parts.append(custom_prompt)

    if use_fewshot:
        with open("data/prompt/custom_few_shot_learning_multi_language.txt", "r", encoding="utf-8") as f:
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

def predict_cluster_from_query(variable_dict: dict):
    # 이 함수는 더 이상 LLM을 호출하지 않고, 주어진 정보로 예측만 수행
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

# ==================== 신규 추가: 헬퍼 함수 ====================

def _calculate_info_score(extracted_vars):
    """추출된 변수들의 가중치 합으로 정보 충분도 점수를 계산합니다."""
    if not VARIABLE_WEIGHTS: return 0.0
    current_score = sum(VARIABLE_WEIGHTS.get(var, 0) for var, value in extracted_vars.items() if value is not None)
    print(f"정보 충분도 점수: {current_score:.4f}")
    return current_score

def _generate_clarifying_question(user_query, context):
    variable_map = {
        'revisit_indicator': '이번이 한국 첫 방문인지, 혹은 이전에 한국을 방문한 적이 있는지',
        'visit_local_indicator': '수도권(서울/경기/인천) 외 다른 지역을 방문할 계획이 있는지',
        'stay_duration': '한국 여행 기간',
        'planned_activity': '한국 여행을 하기위해 계획한 활동'
    }
    
    missing_vars = []
    if VARIABLE_WEIGHTS:
        sorted_vars = sorted(VARIABLE_WEIGHTS.keys(), key=lambda k: VARIABLE_WEIGHTS[k], reverse=True)
        for var in sorted_vars:
            if context.get(var) is None and var in variable_map:
                missing_vars.append(variable_map[var])
    
    if not missing_vars:
        return "여행에 대해 조금만 더 자세히 말씀해주시겠어요?"

    question_prompt = f"""당신은 친절한 여행 플래너입니다.
        사용자가 아래와 같이 질문했습니다.
        사용자 질문: "{user_query}"

        사용자 맞춤 추천을 위해 '{', '.join(missing_vars[:2])}' 정보가 필요합니다.
        사용자의 질문 맥락에 맞춰 자연스럽게 질문을 한 문장으로 만들어주세요."""
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": question_prompt}])
        return response.choices[0].message.content
    except Exception:
        return f"혹시 계획 중인 {missing_vars[0]}에 대해 조금 더 알려주실 수 있나요?"



# --- 대표 실행 함수 (재설계) ---
def get_user_cluster(user_query: str, previous_context: dict = None):
    if preprocessor is None or pca is None or kmeans is None or imputation_base_data.empty:
        return None, None

    #if not all([preprocessor, pca, kmeans, imputation_base_data, VARIABLE_WEIGHTS]):
    #    return "FAIL", "필수 모델/데이터 파일이 로드되지 않았습니다."

    newly_extracted_vars = query_llm_for_variables(user_query)
    current_context = previous_context.copy() if previous_context else {}
    current_context.update({k: v for k, v in newly_extracted_vars.items() if v is not None})
    
    score = _calculate_info_score(current_context)

    if score > 0.50:
        #print("✅ 정보가 충분하여 클러스터링을 진행합니다.")
        cluster_label = predict_cluster_from_query(current_context)
        if cluster_label is not None:
            profile = CLUSTER_PROFILES.get(cluster_label, "정의되지 않은 클러스터입니다.")
            return "SUCCESS", (cluster_label, profile)
        else:
            return "FAIL", "클러스터 예측에 실패했습니다."
    else:
        #print("⚠️ 정보가 불충분하여 사용자에게 재질의합니다.")
        question = _generate_clarifying_question(user_query, current_context)
        return "RETRY_WITH_QUESTION", (question, current_context)
