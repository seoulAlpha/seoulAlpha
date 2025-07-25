# main.py

import os
from dotenv import load_dotenv

# 각 모듈에서 대표 함수들을 가져옵니다.
from cluster_predictor import get_user_cluster, extract_region_from_query
# rag_retriever에서 최종 답변을 생성하는 대표 함수를 가져옵니다.
from rag_retriever import get_rag_recommendation

# --- 초기 설정 ---
load_dotenv()



# --- 메인 실행 로직 ---
if __name__ == "__main__":
    # 1. 사용자 쿼리 입력
    test_queries = [
        # 서울 근교 시나리오
        "주말에 서울 근교로 당일치기 드라이브 가고 싶은데, 경치 좋은 곳 좀 추천해 줘.",
        #"대중교통으로 갈 수 있는 서울 근교 여행지 없을까? 예쁜 카페나 산책로가 있었으면 좋겠어.",
        "이번 주에 연차 썼는데, 서울 근교에서 하루 푹 쉬다 올 만한 한적한 장소 추천해 줘.",

        # 지방 장기 여행 시나리오
        #"시간 여유가 많아서 2주 정도 지방으로 푹 쉬러 떠나고 싶어. 서울이랑 완전 다른 느낌의 조용한 곳이면 좋겠어.",
        "전라도 쪽으로 일주일 정도 맛집 투어 여행을 계획 중이야. 꼭 가봐야 할 도시나 식당 위주로 알려줄래?",
        "사람들이 잘 모르는 국내 여행지 중에, 최소 5일 이상 머물면서 그 지역을 깊게 경험할 수 있는 곳 추천해 줘.",
        
        # "나 저번에 남산타워 갔었는데, 사람많아서 좀 별로 였거든? 유명하긴 한데 사람 덜 있는거 있어?"
    ]

    # 2. for 루프로 각 질문을 순서대로 처리
    for i, user_query in enumerate(test_queries):
        print(f"\n\n{'='*20} 🧪 테스트 {i+1} 시작 {'='*20}")
        print(f"질문: {user_query}")
        print(f"{'='*54}")

        # 클러스터 예측 모듈 호출
        print("\n[1/3] 🔍 사용자님의 여행 스타일을 분석 중입니다...")
        cluster_id, cluster_profile = get_user_cluster(user_query)
        #cluster_id, cluster_profile = 0, ''
        region_keywords = extract_region_from_query(user_query)
        
        if cluster_id is None:
            print("사용자 분석에 실패했습니다. 일반적인 검색을 시도합니다.")
            rag_query = user_query
        else:
            print(f"분석 완료! [클러스터 {cluster_id}: {cluster_profile.split('.')[0]}]")
            # RAG 검색을 위한 '슈퍼 쿼리' 생성
            rag_query = f"{cluster_profile} 특징을 가진 여행객이 '{user_query}'와 같은 여행을 할 때 가기 좋은 곳"
        
        print("\n[2/3] 맞춤 여행지를 검색하고 추천 답변을 생성 중입니다...")
        print(f"   (RAG 검색어: {rag_query})")
        
        # RAG 모듈 호출 (검색과 답변 생성을 한 번에 처리)
        final_answer = get_rag_recommendation(rag_query, region_keywords)

        # 최종 결과 출력
        print("\n[3/3] 여행 추천이 완료되었습니다!")
        print("-" * 50)
        print(final_answer)
        print("-" * 50)