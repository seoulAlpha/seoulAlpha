# main.py

import os
from dotenv import load_dotenv

# 각 모듈에서 대표 함수들을 가져옵니다.
from cluster_predictor_v2 import get_user_cluster
from region_extractor import extract_region_from_query
from rag_retriever import get_rag_recommendation


# --- 초기 설정 ---
load_dotenv()
# --- 메인 실행 로직 ---
if __name__ == "__main__":
    # 1. 테스트 시나리오를 제거하고, 무한 루프(while True)로 변경
    while True:
        # 대화 상태를 저장할 변수 초기화
        conversation_context = {}
        full_conversation = []
        cluster_info = None
        max_turns = 3 # 최대 대화 횟수 제한

        # 2. 첫 질문을 input()으로 직접 입력받기
        print("\n\n" + "="*54)
        current_query = input("👤 사용자: ")
        print("="*54)

        # '종료' 입력 시 프로그램 종료
        if current_query.lower() in ["종료", "exit", "quit"]:
            print("프로그램을 종료합니다.")
            break

        # 3. 클러스터가 확정될 때까지 대화 루프 실행
        for turn in range(max_turns):
            print(f"\n대화 {turn + 1}")
            full_conversation.append(current_query)

            # 클러스터 예측 모듈 호출
            status, data = get_user_cluster(current_query, conversation_context)

            if status == "SUCCESS":
                cluster_info = data # (cluster_id, cluster_profile)
                print(f"클러스터 분석 성공! [클러스터 {cluster_info[0]}]")
                break # 대화 루프 종료

            elif status == "RETRY_WITH_QUESTION":
                question_to_user, updated_context = data
                conversation_context = updated_context # 대화 내용 업데이트
                print(f"AI: {question_to_user}")

                # 4. 추가 답변도 input()으로 직접 입력받도록 수정
                current_query = input("👤 사용자: ")

            elif status == "FAIL":
                print(f"분석 실패: {data}")
                break

        # 5. 클러스터링이 최종 성공한 경우에만 RAG 실행
        if cluster_info:
            cluster_id, cluster_profile = cluster_info
            
            # --- Region Extractor 단계 ---
            final_query_for_rag = " ".join(full_conversation)
            region_keywords = extract_region_from_query(final_query_for_rag)
            
            # --- RAG 단계 ---
            rag_query = f"{cluster_profile} 특징을 가진 여행객이 '{final_query_for_rag}'와 같은 여행을 할 때 가기 좋은 곳"
            print("\n맞춤 여행지를 검색하고 추천 답변을 생성 중입니다...")
            print(f"   (RAG 검색어: {rag_query})")
            
            final_answer = get_rag_recommendation(rag_query, region_keywords)

            # --- 최종 결과 출력 ---
            print("\n여행 추천이 완료되었습니다!")
            print("-" * 50)
            print(f"AI:\n{final_answer}")
            print("-" * 50)
        else:
            print("\n최종 클러스터 분석에 실패하여 추천을 생성할 수 없습니다.")