# app.py

import os
from dotenv import load_dotenv
import gradio as gr
from langdetect import detect
from deep_translator import GoogleTranslator

# 모듈 import
from cluster_predictor import get_user_cluster
from region_extractor import extract_region_from_query
from rag_retriever import get_rag_recommendation

# --- 초기 설정 ---
load_dotenv()

# 언어 코드 매핑 (deep_translator 호환)
LANG_CODE_MAP = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "iw": "he",
}
def normalize_lang_code(code: str) -> str:
    return LANG_CODE_MAP.get(code.lower(), code)


# --- Gradio용 대화 함수 ---
def chatbot_interface(user_input, history, state):
    if user_input.lower() in ["종료", "exit", "quit"]:
        return history + [[user_input, "프로그램을 종료합니다."]], state

    conversation_context = state.get("conversation_context", {})
    full_conversation = state.get("full_conversation", [])

    # --- Step1: 입력 언어 감지 & 한국어 번역 ---
    try:
        detected = detect(user_input)     # 'en', 'ja', 'fr', 'zh-cn' ...
        input_lang = normalize_lang_code(detected)
    except Exception as e:
        return history + [[user_input, f"❌ 언어 감지 오류: {e}"]], state

    if input_lang != "ko":
        try:
            current_query = GoogleTranslator(source=input_lang, target="ko").translate(user_input)
        except Exception as e:
            return history + [[user_input, f"❌ 번역 오류: {e}"]], state
    else:
        current_query = user_input

    cluster_info = None
    max_turns = 3

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

            # 질문도 입력 언어로 번역해서 사용자에게 보여줌
            if input_lang != "ko":
                try:
                    question_to_user = GoogleTranslator(source="ko", target=input_lang).translate(question_to_user)
                except:
                    pass

            # state 업데이트
            state["conversation_context"] = conversation_context
            state["full_conversation"] = full_conversation
            return history + [[user_input, question_to_user]], state

        elif status == "FAIL":
            fail_msg = "최종 클러스터 분석에 실패했습니다."
            if input_lang != "ko":
                try:
                    fail_msg = GoogleTranslator(source="ko", target=input_lang).translate(fail_msg)
                except:
                    pass
            return history + [[user_input, fail_msg]], state

    # RAG 실행
    if cluster_info:
        cluster_id, cluster_profile = cluster_info
        final_query_for_rag = " ".join(full_conversation)
        region_keywords = extract_region_from_query(final_query_for_rag)

        rag_query = f"{cluster_profile} 특징을 가진 여행객이 '{final_query_for_rag}'와 같은 여행을 할 때 가기 좋은 곳"
        final_answer_ko = get_rag_recommendation(rag_query, region_keywords)

        # 최종 답변도 입력 언어로 다시 번역
        final_answer = final_answer_ko
        if input_lang != "ko":
            try:
                final_answer = GoogleTranslator(source="ko", target=input_lang).translate(final_answer_ko)
            except:
                final_answer = f"❌ 결과 번역 오류: {final_answer_ko}"

        # state 업데이트
        state["conversation_context"] = conversation_context
        state["full_conversation"] = full_conversation
        return history + [[user_input, final_answer]], state

    else:
        fail_msg = "추천을 생성할 수 없습니다."
        if input_lang != "ko":
            try:
                fail_msg = GoogleTranslator(source="ko", target=input_lang).translate(fail_msg)
            except:
                pass
        return history + [[user_input, fail_msg]], state

# --- Gradio UI 정의 ---
with gr.Blocks() as demo:
    gr.Markdown("## ✈️ 여행 추천 챗봇")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="사용자 입력")
    state = gr.State({"conversation_context": {}, "full_conversation": []})

    def respond(message, chat_history, state):
        response, new_state = chatbot_interface(message, chat_history, state)
        return "", response, new_state

    msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])

if __name__ == "__main__":
    demo.launch()
