import subprocess
import sys
import os
import json

# 설치가 필요한 라이브러리 없는 경우 자동 설치
def install_required_packages(package, import_name=None):
    try:
        if import_name:
            __import__(import_name)
        else:
            __import__(package)
    except ImportError:
        print(f"📦 {package} 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 설치 완료!")

required_packages = [
    ("scikit-learn", "sklearn"),
    ("tqdm", "tqdm"),
    ("openai", "openai"),
    ("pandas", "pandas"),
    ("numpy", "numpy")
]

for pkg, imp in required_packages:
    install_required_packages(pkg, imp)

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
import numpy as np

# OpenAI API 키 설정
USER_KEY = ""
client = OpenAI(api_key=USER_KEY)

# 데이터 준비
preprocessed_data = pd.read_csv('./관광데이터.csv', encoding='cp949')

# 변수 구분
categorical_cols = ['country', 'gender', 'age',
                    'revisit_indicator', 'visit_local_indicator', 'planned_activity']

numerical_cols = [
    'stay_duration', 'accommodation_percent', 'food_percent', 'shopping_percent', 'food',
    'landscape', 'heritage', 'language', 'safety', 'budget',
    'accommodation', 'transport', 'navigation'
]
used_variables = categorical_cols + numerical_cols

for col in categorical_cols:
    preprocessed_data[col] = preprocessed_data[col].astype(str)
preprocessed_data_clean = preprocessed_data.dropna(subset=used_variables).copy()

# 전처리 파이프라인 정의
# 수치형 파이프라인: 평균 대체 + 정규화
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# 범주형 파이프라인: 최빈값 대체 + 원핫인코딩
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_pipeline, categorical_cols),
    ('num', numeric_pipeline, numerical_cols)
])

# 학습: 전처리 + PCA + 클러스터링
X_preprocessed = preprocessor.fit_transform(preprocessed_data_clean)
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_preprocessed)

kmeans = KMeans(n_clusters=7, random_state=42)
preprocessed_data_clean['cluster'] = kmeans.fit_predict(X_reduced)

print("explained_variance_ratio:", pca.explained_variance_ratio_.sum())
print(f"Silhouette Score: {silhouette_score(X_reduced, preprocessed_data_clean['cluster']):.4f}")

# LLM 질의 = 변수 매핑 함수
def load_text_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[파일 로딩 실패] {filepath} - {e}")
        return ""

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
            messages=messages
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print("[파싱 실패]", e)
        return {}


def impute_with_user_subgroup(user_input_dict, df_base):
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
            if var in numerical_cols:
                imputed[var] = filtered_df[var].mean() if not filtered_df.empty else df_base[var].mean()
            elif var in categorical_cols:
                mode_series = filtered_df[var].mode() if not filtered_df.empty else df_base[var].mode()
                imputed[var] = mode_series.iloc[0] if not mode_series.empty else None
    return imputed

# 질의 = 예측 함수
def predict_cluster_from_query(user_query):
    variable_dict = query_llm_for_variables(user_query, use_prompt=True, use_fewshot=True)
    
    # null이 아닌 값만 필터링하여 출력
    filtered_dict = {k: v for k, v in variable_dict.items() if v is not None}
    print("⮕ LLM 추출 결과:", filtered_dict)

    # 결측 보완
    completed_input = impute_with_user_subgroup(variable_dict, preprocessed_data_clean)
    df = pd.DataFrame([completed_input])

    for col in categorical_cols:
        df[col] = df[col].astype(str)
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    try:
        X_processed = preprocessor.transform(df)
        X_pca = pca.transform(X_processed)
        cluster_label = kmeans.predict(X_pca)[0]
        return cluster_label
    except Exception as e:
        print("[예측 실패]", e)
        return None
    
# main block
if __name__ == "__main__":
    test_inputs = [
        "나는 50대 남성이고, 자연 풍경을 좋아해서 제주도에 4일 여행했어요",
        "저는 20대 여성이며 쇼핑을 좋아해요. 서울에서 3일간 머물렀어요",
        "나는 30대 남자고 한국 전통문화 체험이 좋아서 전주에 갔어요. 총 5일 있었어요",
        "저는 미국에서 왔고, 처음 방문했어요. 한국 음식에 관심이 많아 6일간 머물렀어요",
        "저는 일본 여성이고, 두 번째 방문입니다. 자연 풍경과 유적지를 보기 위해 강원도에 7일 머물렀어요"
    ]

    for i, user_input in enumerate(test_inputs, 1):
        cluster = predict_cluster_from_query(user_input)
        print(f"# 실행 예시 {i}")
        print(f"입력 문장: {user_input}")
        print(f"예측된 클러스터: {cluster}\n")