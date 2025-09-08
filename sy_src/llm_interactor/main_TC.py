# main_tc.py

import os
import pandas as pd
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# 각 모듈에서 대표 함수들을 가져옵니다.
from cluster_predictor import get_user_cluster
from region_extractor import extract_region_from_query
from rag_retriever import get_rag_recommendation


# --- 초기 설정 ---
load_dotenv()

def chatbot_pipeline(user_query):
    """main.py 로직을 함수화 (max_turn = 1)"""
    conversation_context = {}
    full_conversation = []
    cluster_info = None
    max_turns = 1  # ✅ 한 번만 시도

    current_query = user_query

    for turn in range(max_turns):
        full_conversation.append(current_query)
        status, data = get_user_cluster(current_query, conversation_context)

        if status == "SUCCESS":
            cluster_info = data
            break
        elif status == "RETRY_WITH_QUESTION":
            # TC loop에서는 추가 입력을 받지 않고 기록만 남김
            current_query = "추가 입력 없음"
        elif status == "FAIL":
            return "❌ 클러스터 분석 실패"

    if cluster_info:
        cluster_id, cluster_profile = cluster_info
        final_query_for_rag = " ".join(full_conversation)
        region_keywords = extract_region_from_query(final_query_for_rag)

        rag_query = f"{cluster_profile} 특징을 가진 여행객이 '{final_query_for_rag}'와 같은 여행을 할 때 가기 좋은 곳"
        final_answer = get_rag_recommendation(rag_query, region_keywords)
        return final_answer
    else:
        return "❌ 추천 실패"

def run_test_cases(tc_path, output_path="data/TC/result_TC.xlsx"):
    df = pd.read_excel(tc_path)

    # 결과 컬럼 확장
    if "챗봇 답변(원문)" not in df.columns:
        df["챗봇 답변(원문)"] = None
    if "챗봇 답변(한국어 번역)" not in df.columns:
        df["챗봇 답변(한국어 번역)"] = None

    for idx, row in df.iterrows():
        user_query = str(row["원문 질문"]).strip()
        lang = str(row["언어"]).strip()

        # --- 질문 한국어 번역 ---
        if lang != "한국어":
            try:
                translated_q = GoogleTranslator(source="auto", target="ko").translate(user_query)
                df.at[idx, "한국어 번역"] = translated_q
            except Exception as e:
                df.at[idx, "한국어 번역"] = f"번역 오류: {e}"
        else:
            df.at[idx, "한국어 번역"] = user_query

        # --- 챗봇 실행 ---
        try:
            answer = chatbot_pipeline(user_query)
            df.at[idx, "챗봇 답변(원문)"] = answer

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

        # ✅ 어떤 답변이든 무조건 다음 질문으로 넘어감
        continue

    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"✅ 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    run_test_cases("data/TC/TC_multilang.xlsx")
