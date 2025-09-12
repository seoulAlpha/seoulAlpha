# main_tc.py

import os
import pandas as pd
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect   # ✅ 언어 자동 감지 추가

# 각 모듈에서 대표 함수들을 가져옵니다.
from cluster_predictor import get_user_cluster
from region_extractor import extract_region_from_query
from rag_retriever import get_rag_recommendation


# --- 초기 설정 ---
load_dotenv()

LANG_CODE_MAP = {
    "zh-cn": "zh-CN",   # 중국어 간체
    "zh-tw": "zh-TW",   # 중국어 번체
    "iw": "he",         # 히브리어 (langdetect 구버전 코드 → modern code)
    # 필요 시 추가 매핑 가능
}

def normalize_lang_code(code: str) -> str:
    """langdetect 코드 → deep_translator 지원 코드 변환"""
    code = code.lower()
    return LANG_CODE_MAP.get(code, code)

def chatbot_pipeline(user_query):
    """언어 감지 + 한국어 변환 + 답변을 입력언어로 다시 번역 (max_turn = 1)"""
    conversation_context = {}
    full_conversation = []
    cluster_info = None
    max_turns = 1

    # --- step1: 입력 언어 감지 ---
    try:
        input_lang = detect(user_query)   # 예: 'en', 'fr', 'ja', 'zh-cn', 'ko'
        input_lang = normalize_lang_code(input_lang)
    except Exception as e:
        return f"❌ 언어 감지 오류: {e}"

    # --- step2: 한국어로 번역 ---
    if input_lang != "ko":
        try:
            current_query = GoogleTranslator(source=input_lang, target="ko").translate(user_query)
        except Exception as e:
            return f"❌ 번역 오류: {e}"
    else:
        current_query = user_query

    # --- 기존 로직 실행 ---
    for turn in range(max_turns):
        full_conversation.append(current_query)
        status, data = get_user_cluster(current_query, conversation_context)

        if status == "SUCCESS":
            cluster_info = data
            break
        elif status == "RETRY_WITH_QUESTION":
            current_query = "추가 입력 없음"
        elif status == "FAIL":
            return "❌ 클러스터 분석 실패"

    if cluster_info:
        cluster_id, cluster_profile = cluster_info
        final_query_for_rag = " ".join(full_conversation)
        region_keywords = extract_region_from_query(final_query_for_rag)

        rag_query = f"{cluster_profile} 특징을 가진 여행객이 '{final_query_for_rag}'와 같은 여행을 할 때 가기 좋은 곳"
        final_answer_ko = get_rag_recommendation(rag_query, region_keywords)

        # --- step3: 결과를 입력 언어로 다시 번역 ---
        if input_lang != "ko":
            try:
                final_answer = GoogleTranslator(source="auto", target=input_lang).translate(final_answer_ko)
                return final_answer
            except Exception as e:
                return f"❌ 결과 번역 오류: {e}"
        else:
            return final_answer_ko
    else:
        return "❌ 추천 실패"


def run_test_cases(tc_path, output_path="data/TC/result_TC.xlsx"):
    df = pd.read_excel(tc_path)

    if "챗봇 답변(원문)" not in df.columns:
        df["챗봇 답변(원문)"] = None
    if "챗봇 답변(한국어 번역)" not in df.columns:
        df["챗봇 답변(한국어 번역)"] = None
    if "자동 감지 언어" not in df.columns:   # ✅ 감지된 언어 기록
        df["자동 감지 언어"] = None

    for idx, row in df.iterrows():
        id = str(row["ID"]).strip()

        if id != "9":
            continue
        user_query = str(row["원문 질문"]).strip()

        try:
            # --- 챗봇 실행 ---
            answer = chatbot_pipeline(user_query)
            df.at[idx, "챗봇 답변(원문)"] = answer

            # --- 감지된 언어 기록 ---
            try:
                detected_lang = detect(user_query)
                df.at[idx, "자동 감지 언어"] = detected_lang
            except:
                df.at[idx, "자동 감지 언어"] = "감지 실패"

            # --- 챗봇 답변 한국어 번역 ---
            if answer and not answer.startswith("❌"):
                try:
                    translated_a = GoogleTranslator(source="auto", target="ko").translate(answer)
                    df.at[idx, "챗봇 답변(한국어 번역)"] = translated_a
                except Exception as e:
                    df.at[idx, "챗봇 답변(한국어 번역)"] = f"번역 오류: {e}"
            else:
                df.at[idx, "챗봇 답변(한국어 번역)"] = answer

        except Exception as e:
            df.at[idx, "챗봇 답변(원문)"] = f"챗봇 오류: {e}"
            df.at[idx, "챗봇 답변(한국어 번역)"] = f"챗봇 오류: {e}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"✅ 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    run_test_cases("data/TC/TC_multilang.xlsx")
