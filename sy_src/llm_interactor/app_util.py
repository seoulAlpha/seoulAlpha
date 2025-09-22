from langdetect import detect
from deep_translator import GoogleTranslator
from cluster_predictor import get_user_cluster
from region_extractor import update_region_keywords
from rag_retriever import get_rag_recommendation
from query_detector import classify_recommendation_intent

# --- 언어 코드 매핑 ---
LANG_CODE_MAP = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "iw": "he",
}

def normalize_lang_code(code: str) -> str:
    return LANG_CODE_MAP.get(code.lower(), code)

def translate_text(text, source_lang, target_lang):
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception:
        return text

def handle_exit(user_input, history, state):
    return history + [[user_input, "프로그램을 종료합니다."]], state

def enforce_turn_limit(user_input, history, state, max_turns=10):
    state["turn_count"] = state.get("turn_count", 0) + 1
    if state["turn_count"] > max_turns:
        msg = "⚠️ 이번 세션에서는 최대 10번의 대화만 가능합니다. 새로운 세션을 시작해주세요."
        return history + [[user_input, msg]], state, True
    return history, state, False

def detect_and_translate(user_input):
    try:
        detected = detect(user_input)
        input_lang = normalize_lang_code(detected)
    except Exception as e:
        raise RuntimeError(f"언어 감지 오류: {e}")

    if input_lang != "ko":
        return input_lang, translate_text(user_input, input_lang, "ko")
    return input_lang, user_input

def resolve_cluster(current_query, state, input_lang):
    cluster_info = state.get("cluster_info")
    if cluster_info:
        return cluster_info, None, state

    conversation_context = state.get("conversation_context", {})
    full_conversation = state.get("full_conversation", [])

    full_conversation.append(current_query)
    status, data = get_user_cluster(current_query, conversation_context)

    if status == "SUCCESS":
        state["cluster_info"] = data
        return data, None, state

    elif status == "RETRY_WITH_QUESTION":
        question_to_user, updated_context = data
        if input_lang != "ko":
            question_to_user = translate_text(question_to_user, "ko", input_lang)
        state.update({"conversation_context": updated_context, "full_conversation": full_conversation})
        return None, question_to_user, state

    elif status == "FAIL":
        msg = "최종 클러스터 분석에 실패했습니다."
        if input_lang != "ko":
            msg = translate_text(msg, "ko", input_lang)
        return None, msg, state

    return None, "클러스터 확인 실패", state

def generate_recommendation(current_query, state, input_lang):
    cluster_info = state.get("cluster_info")
    if not cluster_info:
        msg = "추천을 생성할 수 없습니다."
        if input_lang != "ko":
            msg = translate_text(msg, "ko", input_lang)
        return msg, state

    cluster_id, cluster_profile = cluster_info
    rag_query = f"{cluster_profile} 특징을 가진 여행객이 '{current_query}'와 같은 여행을 할 때 가기 좋은 곳"
    final_answer_ko = get_rag_recommendation(rag_query, state.get("region_keywords", []))
    final_answer = translate_text(final_answer_ko, "ko", input_lang) if input_lang != "ko" else final_answer_ko

    state["conversation_context"] = state.get("conversation_context", {})
    state["full_conversation"] = state.get("full_conversation", [])
    return final_answer, state

def detect_intent(user_history, current_query, input_lang):
    from query_detector import classify_recommendation_intent
    intent = classify_recommendation_intent(user_history, current_query)
    if intent == "X":
        msg = "여행 추천을 원하시면 지역이나 목적을 말씀해주세요."
        if input_lang != "ko":
            msg = translate_text(msg, "ko", input_lang)
        return False, msg
    return True, None

def update_history_and_state(history, state, user_input, answer):
    # full_conversation 누적
    state["full_conversation"].append((user_input, answer))
    # UI용 history 누적
    history.append([user_input, answer])
    return history, state