import os
from dotenv import load_dotenv
import gradio as gr

# utils에서 필요한 함수만 가져오기
from app_util import (
    handle_exit,
    enforce_turn_limit,
    detect_and_translate,
    resolve_cluster,
    detect_intent,
    generate_recommendation,
    update_region_keywords,  # region_extractor_sy에서 사용
    update_history_and_state,
)

# --- 초기 설정 ---
load_dotenv()

# ======================
# chatbot interface
# ======================
def chatbot_interface(user_input, history, state):
    if history is None:
        history = []

    if user_input.lower() in ["종료", "exit", "quit"]:
        return update_history_and_state(history, state, user_input,
                                    handle_exit(user_input, history, state)[0])

    # turn 제한
    history, state, exceeded = enforce_turn_limit(user_input, history, state)
    if exceeded:
        return update_history_and_state(history, state, user_input,
                                        "⚠️ 이번 세션에서는 최대 10번 대화가 가능합니다.")

    # Step1: 언어 감지
    try:
        input_lang, current_query = detect_and_translate(user_input)
    except RuntimeError as e:
        return update_history_and_state(history, state, user_input, str(e))

    # Step2: 클러스터
    _, question_or_msg, state = resolve_cluster(current_query, state, input_lang)
    if question_or_msg:
        return update_history_and_state(history, state, user_input, question_or_msg)
    
    # Step3: 추천 의도
    user_history = state.get("full_conversation", [])[-4:]
    has_intent, intent_msg = detect_intent(user_history, current_query, input_lang)
    if not has_intent:
        return update_history_and_state(history, state, user_input, intent_msg)

    # Step4: 지역 업데이트
    state = update_region_keywords(current_query, state)

    # Step5: 추천 생성
    final_answer, state = generate_recommendation(current_query, state, input_lang)
    return update_history_and_state(history, state, user_input, final_answer)


# ======================
# Gradio UI 정의
# ======================
with gr.Blocks() as demo:
    gr.Markdown("## ✈️ 여행 추천 챗봇")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="사용자 입력", interactive=True)
    state = gr.State({
        "conversation_context": {},
        "full_conversation": [],
        "turn_count": 0
    })

    def respond(message, chat_history, state):
        response, new_state = chatbot_interface(message, chat_history, state)
        if new_state.get("turn_count", 0) >= 10:
            return "", response, new_state, gr.update(interactive=False)
        return "", response, new_state, gr.update(interactive=True)

    msg.submit(
        respond,
        [msg, chatbot, state],
        [msg, chatbot, state, msg]
    )

    reset_btn = gr.Button("🔄 다시 대화 시작")

    def reset_chat():
        return "", [], {
            "conversation_context": {},
            "full_conversation": [],
            "turn_count": 0
        }, gr.update(interactive=True)

    reset_btn.click(
        reset_chat,
        inputs=[],
        outputs=[msg, chatbot, state, msg]
    )

if __name__ == "__main__":
    demo.launch(share=True)
