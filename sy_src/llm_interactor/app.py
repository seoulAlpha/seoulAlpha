# app.py

import os
from dotenv import load_dotenv
import gradio as gr

# 모듈 import
from cluster_predictor import get_user_cluster
from region_extractor import extract_region_from_query
from rag_retriever import get_rag_recommendation

# --- 초기 설정 ---
load_dotenv()

# --- Gradio용 대화 함수 ---
def chatbot_interface(user_input, history):
    if user_input.lower() in ["종료", "exit", "quit"]:
        return history + [[user_input, "프로그램을 종료합니다."]]

    conversation_context = {}
    full_conversation = []
    cluster_info = None
    max_turns = 3
    current_query = user_input

    # 클러스터 확정 루프
    for turn in range(max_turns):
        full_conversation.append(current_query)
        status, data = get_user_cluster(current_query, conversation_context)

        if status == "SUCCESS":
            cluster_info = data
            break
        elif status == "RETRY_WITH_QUESTION":
            question_to_user, updated_context = data
            conversation_context = updated_context
            return history + [[user_input, question_to_user]]
        elif status == "FAIL":
            return history + [[user_input, "최종 클러스터 분석에 실패했습니다."]]

    # RAG 실행
    if cluster_info:
        cluster_id, cluster_profile = cluster_info
        final_query_for_rag = " ".join(full_conversation)
        region_keywords = extract_region_from_query(final_query_for_rag)

        rag_query = f"{cluster_profile} 특징을 가진 여행객이 '{final_query_for_rag}'와 같은 여행을 할 때 가기 좋은 곳"
        final_answer = get_rag_recommendation(rag_query, region_keywords)
        return history + [[user_input, final_answer]]
    else:
        return history + [[user_input, "추천을 생성할 수 없습니다."]]


# --- Gradio UI 정의 ---
with gr.Blocks() as demo:
    gr.Markdown("## ✈️ 여행 추천 챗봇")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="사용자 입력")

    def respond(message, chat_history):
        response = chatbot_interface(message, chat_history)
        return "", response

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
