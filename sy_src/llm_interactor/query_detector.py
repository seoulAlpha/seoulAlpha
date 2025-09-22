from dotenv import load_dotenv
import random
from openai import OpenAI
import os
import json

# --- 초기 설정 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

def classify_recommendation_intent(history, current_query):
    """
    최근 대화와 현재 입력을 기반으로 추천 의도를 판별합니다.
    - history: 유저 발화 리스트 (str 리스트)
    - current_query: 현재 유저 입력 (str)
    return: "O" (추천 의도 있음) / "X" (추천 의도 없음)
    """
    flat_history = []
    for turn in history[-4:]:
        if isinstance(turn, (tuple, list)) and len(turn) == 2:
            flat_history.append(f"사용자: {turn[0]}\n챗봇: {turn[1]}")
        else:
            flat_history.append(str(turn))

    history_text = "\n".join(flat_history)

    prompt = f"""
        최근 대화 기록:
        나 제주도 여행지 추천좀
        나 첫방문이야. 그리고 미식 좋아해
        제주도 갈래

        현재 질문:
        다른데는 없어?

        판단: O

        ---

        최근 대화 기록:
        안녕
        나 한국 두번째 방문이야. 그리고 경상북도쪽 방문하고 싶은데

        현재 질문:
        오늘 날씨 어때

        판단: X

        ---

        최근 대화 기록:
        {history_text}

        현재 질문:
        {current_query}

        판단:
        - 현재 질문이 여행 추천(장소, 일정, 코스, 숙소, 맛집 등)에 관한 것이면 "O".
        - 현재 질문이 추천과 무관하다면 "X".
        답변은 반드시 O 또는 X 중 하나만 출력하세요.
        """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip().upper()

    # 보정 (혹시 다른 값 나오는 경우 X 처리)
    if "O" in answer:
        return "O"
    if "X" in answer:
        return "X"
    return "X"