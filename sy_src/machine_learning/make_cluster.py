import subprocess
import sys
import os
import json
from dotenv import load_dotenv
import joblib  # 모델 저장을 위해 import

# --- 초기 설정 ---
# .env 파일에서 환경 변수 로드
load_dotenv()
USER_KEY = os.getenv("API_KEY")

# 필요한 라이브러리 자동 설치 함수
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

# 설치할 패키지 목록
required_packages = [
    ("scikit-learn", "sklearn"),
    ("tqdm", "tqdm"),
    ("openai", "openai"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("python-dotenv", "dotenv")
]

for pkg, imp in required_packages:
    install_required_packages(pkg, imp)

# 라이브러리 import
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# --- 데이터 및 변수 정의 ---
# 데이터 준비

def main():
    try:
        preprocessed_data = pd.read_csv('./data/관광데이터.csv', encoding='cp949')
    except FileNotFoundError:
        print("❌ '관광데이터.csv' 파일을 찾을 수 없습니다. 스크립트와 같은 위치에 파일을 놓아주세요.")
        sys.exit()

    # 변수 구분
    categorical_cols = ['country', 'gender', 'age', 'revisit_indicator', 'visit_local_indicator', 'planned_activity']
    numerical_cols = [
        'stay_duration', 'accommodation_percent', 'food_percent', 'shopping_percent', 'food',
        'landscape', 'heritage', 'language', 'safety', 'budget', 'accommodation', 'transport', 'navigation'
    ]
    used_variables = categorical_cols + numerical_cols

    # 데이터 타입 변환 및 결측치 처리
    for col in categorical_cols:
        preprocessed_data[col] = preprocessed_data[col].astype(str)
    preprocessed_data_clean = preprocessed_data.dropna(subset=used_variables).copy()

    # --- 전처리 파이프라인 정의 ---
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

    # 전처리기 결합
    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_pipeline, categorical_cols),
        ('num', numeric_pipeline, numerical_cols)
    ])

    # --- 모델 학습 ---
    print("🚀 모델 학습을 시작합니다...")

    # 1. 전처리기(preprocessor) 학습
    X_preprocessed = preprocessor.fit_transform(preprocessed_data_clean)

    # 2. PCA 모델 학습
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X_preprocessed)

    # 3. K-Means 모델 학습
    kmeans = KMeans(n_clusters=7, random_state=42)
    preprocessed_data_clean['cluster'] = kmeans.fit_predict(X_reduced)

    print("explained_variance_ratio:", pca.explained_variance_ratio_.sum())
    print(f"Silhouette Score: {silhouette_score(X_reduced, preprocessed_data_clean['cluster']):.4f}")

    # --- 학습된 모델 및 데이터 저장 ---
    # 저장할 폴더 생성
    if not os.path.exists('./models'):
        os.makedirs('./models')

    joblib.dump(preprocessor, './models/preprocessor.joblib')
    joblib.dump(pca, './models/pca.joblib')
    joblib.dump(kmeans, './models/kmeans.joblib')
    # 예측 시 결측치 보완을 위해 사용하는 데이터도 함께 저장
    preprocessed_data_clean.to_csv('./models/imputation_base_data.csv', index=False, encoding='utf-8-sig')

    print("\n학습된 모델과 데이터를 './models/' 폴더에 성공적으로 저장했습니다.")
    print("이제 predict_app.py 파일을 실행하여 실시간 예측을 할 수 있습니다.")

if __name__ == "__main__":
    main()