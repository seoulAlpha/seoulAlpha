import os
import json
import re
from kiwipiepy import Kiwi

def clean_text(text):
    """리뷰 텍스트를 전처리하는 함수입니다."""
    text = text.replace('\n', ' ')
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_all_json_files(directory_path):
    """
    지정된 디렉토리 안의 모든 JSONL 파일을 찾아 전처리하고 저장합니다.
    각 줄이 JSON 객체인 JSONL 형식을 유지합니다.
    """
    kiwi = Kiwi()

    try:
        filenames = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"❌ 오류: '{directory_path}' 폴더를 찾을 수 없습니다.")
        return

    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(".jsonl"):
                file_path = os.path.join(dirpath, filename)
                print(f"--- 작업 시작: {file_path} ---")

                try:
                    loaded_data = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            loaded_data.append(json.loads(line))

                    if not isinstance(loaded_data, list):
                        print("-> ❌ 오류: JSONL 파일이 리스트 형태가 아닙니다.")
                        continue

                    total_sentences = 0
                    for place_data in loaded_data:
                        if 'comments' not in place_data or not place_data.get('comments'):
                            continue

                        final_sentences = []
                        for comment in place_data['comments']:
                            if not isinstance(comment, str):
                                continue
                            result = kiwi.split_into_sents(comment)
                            for sentence in result:
                                cleaned_sentence = clean_text(sentence.text)
                                if cleaned_sentence:
                                    final_sentences.append(cleaned_sentence)

                        place_data['processed_sentences'] = final_sentences
                        total_sentences += len(final_sentences)

                    # ✅ JSONL 형식으로 다시 저장 (한 줄에 하나의 JSON 객체)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for item in loaded_data:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')

                    print(f"-> ✅ 완료: 총 {total_sentences}개의 문장을 추출하여 저장했습니다.")

                except json.JSONDecodeError:
                    print(f"-> ❌ 오류: '{filename}'은 유효한 JSON 파일이 아닙니다.")
                except Exception as e:
                    print(f"-> ❌ 오류 발생: {e}")

def main():
    """메인 실행 함수"""
    json_folder_path = 'data/json'
    process_all_json_files(json_folder_path)

if __name__ == "__main__":
    main()
