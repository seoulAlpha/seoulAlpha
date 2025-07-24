import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'

import json
import faiss
import numpy as np

# --- 설정 ---
# 원본 데이터가 있는 루트 폴더 이름
ROOT_DIR = 'faiss_integrated_dbs'
# 결과물을 저장할 폴더 이름
OUTPUT_DIR = 'faiss_merged_output'
# 생성될 통합 파일 이름
MERGED_INDEX_FILE = 'merged.index'
MERGED_METADATA_FILE = 'merged_metadata.jsonl'
# --- 설정 끝 ---


def preprocess_and_merge():
    """
    지정된 폴더 구조를 스캔하여 여러 FAISS index와 메타데이터 파일을
    하나의 index와 메타데이터 파일로 통합합니다.
    """
    # 결과물 저장 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    merged_index = None
    all_metadata = []
    global_vector_id_counter = 0
    is_first_index = True

    print(f"🚀 통합 작업을 시작합니다. 대상 폴더: '{ROOT_DIR}'")

    # 루트 폴더부터 모든 하위 폴더를 순회
    for dirpath, _, filenames in os.walk(ROOT_DIR):
        # 현재 폴더에 .index 파일이 없으면 건너뜀
        if not any(fname.endswith('.index') for fname in filenames):
            continue

        # 폴더 이름을 '지역' 정보로 사용
        region = os.path.basename(dirpath)
        print(f"\n📂 지역 [{region}] 처리 중...")

        # 메타데이터 파일을 기준으로 해당 카테고리의 index 파일을 찾음
        for filename in sorted(filenames):
            if filename.startswith('metadata_') and filename.endswith('.jsonl'):
                
                # 파일명에서 '카테고리' 정보 추출
                category = filename.replace('metadata_', '').replace('.jsonl', '')
                
                metadata_path = os.path.join(dirpath, filename)
                index_path = os.path.join(dirpath, f'faiss_{category}.index')

                if not os.path.exists(index_path):
                    print(f"  - ⚠️ 경고: '{metadata_path}'에 대한 인덱스 파일이 없습니다. 건너뜁니다.")
                    continue

                print(f"  - 카테고리 '{category}' 처리 중...")

                # 개별 FAISS 인덱스 로드
                sub_index = faiss.read_index(index_path)

                # 첫 번째 인덱스 파일을 기준으로 통합 인덱스의 차원을 결정하고 초기화
                if is_first_index:
                    dimension = sub_index.d
                    print(f"  - 벡터 차원({dimension}) 확인. 통합 인덱스를 생성합니다.")
                    # 각 벡터에 고유 ID를 매핑할 수 있는 IndexIDMap 사용
                    index_flat = faiss.IndexFlatL2(dimension)
                    merged_index = faiss.IndexIDMap(index_flat)
                    is_first_index = False
                
                # 차원이 맞지 않는 경우 에러 처리
                if sub_index.d != merged_index.d:
                    print(f"  - ❌ 에러: 벡터 차원이 다릅니다! (기대: {merged_index.d}, 현재: {sub_index.d}). 이 파일은 건너뜁니다.")
                    continue

                # 인덱스에서 벡터들과 총 개수 가져오기
                vectors = sub_index.reconstruct_n(0, sub_index.ntotal)
                num_vectors = sub_index.ntotal

                # 통합 인덱스에 사용할 새로운 고유 ID 배열 생성
                new_global_ids = np.arange(global_vector_id_counter, global_vector_id_counter + num_vectors)

                # 통합 인덱스에 벡터와 새로운 ID 추가
                merged_index.add_with_ids(vectors, new_global_ids)

                # 메타데이터 처리: 한 줄씩 읽어 새로운 정보 추가
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        try:
                            meta_obj = json.loads(line)
                            # 새로운 고유 vector_id, 지역, 카테고리 정보 추가
                            meta_obj['vector_id'] = int(new_global_ids[i])
                            meta_obj['region'] = region
                            meta_obj['category'] = category
                            all_metadata.append(meta_obj)
                        except (json.JSONDecodeError, IndexError) as e:
                            print(f"    - ⚠️ 경고: '{metadata_path}' 파일의 {i+1}번째 줄 처리 중 오류 발생. 건너뜁니다. ({e})")
                
                print(f"    - {num_vectors}개 벡터 추가 완료. (총 {merged_index.ntotal}개)")
                # 다음 파일을 위해 ID 카운터 업데이트
                global_vector_id_counter += num_vectors

    # 모든 파일 처리가 끝난 후, 최종 결과물 저장
    if merged_index and len(all_metadata) > 0:
        output_index_path = os.path.join(OUTPUT_DIR, MERGED_INDEX_FILE)
        output_metadata_path = os.path.join(OUTPUT_DIR, MERGED_METADATA_FILE)

        print("\n💾 최종 파일을 저장합니다...")
        faiss.write_index(merged_index, output_index_path)
        print(f"  - 통합 인덱스 저장 완료: {output_index_path}")

        with open(output_metadata_path, 'w', encoding='utf-8') as f:
            for meta_obj in all_metadata:
                # ensure_ascii=False 옵션으로 한글이 깨지지 않게 저장
                f.write(json.dumps(meta_obj, ensure_ascii=False) + '\n')
        print(f"  - 통합 메타데이터 저장 완료: {output_metadata_path}")
        
        print("\n🎉 모든 작업이 성공적으로 완료되었습니다!")
        print(f"  - 총 {merged_index.ntotal}개의 벡터가 통합되었습니다.")
    else:
        print("\n처리할 데이터가 없습니다. 'ROOT_DIR' 설정과 파일 구조를 확인해주세요.")


if __name__ == '__main__':
    preprocess_and_merge()