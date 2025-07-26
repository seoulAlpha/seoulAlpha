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
        print("'관광데이터.csv' 파일을 찾을 수 없습니다. 스크립트와 같은 위치에 파일을 놓아주세요.")
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
    print("모델 학습을 시작합니다...")

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

    # ==================== 변수 중요도 추출 및 가중치 파일 생성 ====================
    print("\n분류 모델을 사용해 변수 중요도 추출")

    # 1. 랜덤 포레스트 분류 모델 임포트
    from sklearn.ensemble import RandomForestClassifier
    import json # JSON 저장을 위해 import

    # 2. 모델 학습
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_preprocessed, kmeans.labels_)

    # 3. 변수 중요도 추출
    importances = rf_classifier.feature_importances_

    # 4. 중요도를 원래 변수명과 매핑
    feature_names = preprocessor.get_feature_names_out()
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

    # 5. 중요도 높은 순으로 전체 변수 정렬
    ranked_features_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print("클러스터링에 중요한 변수 전체 순위 (상위 10개):")
    print(ranked_features_df.head(10))

    # 6. 전체 순위가 담긴 DataFrame을 CSV 파일로 저장 (기존 코드)
    ranked_features_df.to_csv('./models/feature_importance_ranking.csv', index=False, encoding='utf-8-sig')
    print(f"\n전체 변수 중요도 순위를 './models/feature_importance_ranking.csv'에 저장했습니다.")


    # -------------------- 중요도 전처리 및 가중치 파일 생성 (추가된 부분) --------------------
    print("\n저장된 중요도를 가중치로 사용하기 위해 전처리를 시작합니다...")

    # 7. 원본 변수명 추출을 위한 수정된 함수
    def get_base_feature(feature_name: str, categorical_cols: list) -> str:
        """
        전체 변수 리스트를 기반으로 정확한 원본 변수명을 찾습니다.
        """
        # 'cat__', 'num__' 접두사 제거
        clean_name = feature_name.split('__')[1]
        
        # 범주형 변수 목록을 확인하여 일치하는 원본 변수명 반환
        for col in categorical_cols:
            if clean_name.startswith(col + '_'):
                return col
                
        # 범주형에 해당하지 않으면 수치형 변수이므로 그대로 반환
        return clean_name

    # 전역에 정의된 categorical_cols 리스트를 함수에 전달

    ranked_features_df['base_feature'] = ranked_features_df['feature'].apply(
        lambda x: get_base_feature(x, categorical_cols)
    )

    # 8. 원본 변수 기준으로 중요도 합산 및 정규화
    aggregated_importances = ranked_features_df.groupby('base_feature')['importance'].sum()

    # 8a. 제외할 변수 목록 정의
    EXCLUDE_VARS = ['gender', 'age', 'country']
    print(f"\n제외할 변수: {EXCLUDE_VARS}")

    # 8b. 해당 변수들을 중요도 목록에서 제거
    filtered_importances = aggregated_importances.drop(labels=EXCLUDE_VARS, errors='ignore')
    print("변수 제외 완료.")

    # 8c. 남은 변수들의 중요도 총합이 1.0이 되도록 재정규화
    renormalized_weights = filtered_importances / filtered_importances.sum()
    final_weights = renormalized_weights.sort_values(ascending=False)
    print("남은 변수들의 가중치를 재조정했습니다.")

    print("\n원본 변수별 최종 가중치:")
    print(final_weights)

    # 9. 최종 가중치를 JSON 파일로 저장하여 예측 시 사용
    final_weights.to_json('./models/variable_weights.json', orient='index', indent=4)
    print(f"\n최종 변수 가중치를 './models/  .json'에 저장했습니다.")
    # ------------------------------------------------------------------------------------


    print("\n학습된 모델과 데이터를 './models/' 폴더에 성공적으로 저장했습니다.")
    print("이제 predict_app.py 파일을 실행하여 실시간 예측을 할 수 있습니다.")

if __name__ == "__main__":
    main()