import os
import json

def remove_key_from_data(data, key_to_remove):
    """
    재귀적으로 딕셔너리 또는 딕셔너리 리스트에서 특정 키를 삭제합니다.
    """
    if isinstance(data, dict):
        # 딕셔너리에서 키를 찾으면 삭제
        if key_to_remove in data:
            del data[key_to_remove]
        # 딕셔너리의 값들을 순회하며 재귀 호출
        for key, value in data.items():
            remove_key_from_data(value, key_to_remove)
    elif isinstance(data, list):
        # 리스트의 각 항목에 대해 재귀 호출
        for item in data:
            remove_key_from_data(item, key_to_remove)

def process_all_json_files(directory_path, key_to_remove):
    """
    지정된 디렉토리 안의 모든 JSON 파일을 찾아 특정 키를 삭제하고 저장합니다.
    """
    try:
        filenames = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"❌ 오류: '{directory_path}' 폴더를 찾을 수 없습니다. 폴더 경로를 확인해주세요.")
        return

    for filename in filenames:
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            print(f"--- 작업 시작: {filename} ---")

            try:
                # JSON 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)

                # 키 삭제 함수 호출
                remove_key_from_data(loaded_data, key_to_remove)
                
                # 수정된 내용을 원본 파일에 덮어쓰기
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(loaded_data, f, ensure_ascii=False, indent=4)
                
                print(f"-> ✅ 완료: '{key_to_remove}' 키를 삭제하고 저장했습니다.")

            except Exception as e:
                print(f"-> ❌ 오류 발생: {e}")

def main():
    """메인 실행 함수"""
    json_folder_path = 'data/json'
    key_to_delete = "sentiment_results"  # 삭제할 키 이름
    
    print(f"'{json_folder_path}' 폴더의 모든 JSON 파일에서 '{key_to_delete}' 키를 삭제합니다.")
    process_all_json_files(json_folder_path, key_to_delete)
    print("\n모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()