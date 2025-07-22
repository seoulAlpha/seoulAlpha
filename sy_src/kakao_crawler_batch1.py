import time
import json
import os
from tqdm import tqdm  # [추가] tqdm 라이브러리
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options

# ===================================================================
# 1. 스크래핑할 데이터 목록 정의
# ===================================================================

# 검색할 카테고리
CATEGORIES = ['카페', '식당', '관광명소', '문화재', '레저', '쇼핑', '숙박']

# 행정구역 정보 (전체 목록은 너무 길어 일부만 예시로 포함)
# 실제 사용 시에는 이전에 제공된 전체 목록을 복사하여 사용하세요.
ADMIN_AREAS = {
    # Batch1
    "서울특별시": ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"],
    "부산광역시": ["강서구", "금정구", "기장군", "남구", "동구", "동래구", "부산진구", "북구", "사상구", "사하구", "서구", "수영구", "연제구", "영도구", "중구", "해운대구"],
    "대구광역시": ["군위군", "남구", "달서구", "달성군", "동구", "북구", "서구", "수성구", "중구"],
    "인천광역시": ["강화군", "계양구", "남동구", "동구", "미추홀구", "부평구", "서구", "연수구", "옹진군", "중구"]
    

    # Batch2
    #"광주광역시": ["광산구", "남구", "동구", "북구", "서구"],
    #"대전광역시": ["대덕구", "동구", "서구", "유성구", "중구"],
    # "울산광역시": ["남구", "동구", "북구", "울주군", "중구"],
    #"세종특별자치시": ["세종시"],
    # "경기도": ["수원시", "용인시", "고양시", "성남시", "화성시", "부천시", "남양주시", "안산시", "평택시", "안양시", "시흥시", "파주시", "김포시", "의정부시", "광주시", "하남시", "오산시", "이천시", "안성시", "의왕시", "양주시", "구리시", "포천시", "동두천시", "과천시", "여주시", "양평군", "가평군", "연천군"],
    # "강원특별자치도": ["춘천시", "원주시", "강릉시", "동해시", "태백시", "속초시", "삼척시", "홍천군", "횡성군", "영월군", "평창군", "정선군", "철원군", "화천군", "양구군", "인제군", "고성군", "양양군"],

    # Batch3
    # "충청북도": ["청주시", "충주시", "제천시", "보은군", "옥천군", "영동군", "증평군", "진천군", "괴산군", "음성군", "단양군"],
    # "충청남도": ["천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시", "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군"],
    # "전북특별자치도": ["전주시", "익산시", "군산시", "정읍시", "남원시", "김제시", "완주군", "진안군", "무주군", "장수군", "임실군", "순창군", "고창군", "부안군"],
    # "전라남도": ["목포시", "여수시", "순천시", "나주시", "광양시", "담양군", "곡성군", "구례군", "고흥군", "보성군", "화순군", "장흥군", "강진군", "해남군", "영암군", "무안군", "함평군", "영광군", "장성군", "완도군", "진도군", "신안군"],
    # "경상북도": ["포항시", "경주시", "김천시", "안동시", "구미시", "영주시", "영천시", "상주시", "문경시", "경산시", "의성군", "청송군", "영양군", "영덕군", "청도군", "고령군", "성주군", "칠곡군", "예천군", "봉화군", "울진군", "울릉군"],
    # "경상남도": ["창원시", "진주시", "통영시", "사천시", "김해시", "밀양시", "거제시", "양산시", "의령군", "함안군", "창녕군", "고성군", "남해군", "하동군", "산청군", "함양군", "거창군", "합천군"],
    # "제주특별자치도": ["제주시", "서귀포시"]
}

# ===================================================================
# 2. 핵심 스크래핑 함수 정의
# ===================================================================
def scrape_query(query):
    """
    주어진 검색어(query)에 대해 카카오맵을 스크래핑하고 결과를 JSON 파일로 저장합니다.
    """
    
    # [수정] 데이터 저장 폴더 경로 변경
    province = query.split("_")[0]
    province, file_name = query.split(" ", 1)
    file_name = file_name.replace(" ", "_")
    output_dir = f'data/json/{province}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_filename = os.path.join(output_dir, f"{file_name}.jsonl")


    # 이미 파일이 존재하면 건너뛰기
    if os.path.exists(output_filename):
        return # tqdm 진행률을 위해 print문은 제거

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920x1080')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
    
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get("https://map.kakao.com/")
        wait = WebDriverWait(driver, 10)
        
        search_box = wait.until(EC.presence_of_element_located((By.ID, "search.keyword.query")))
        search_box.send_keys(query)
        search_box.send_keys(Keys.ENTER)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "placelist")))
        time.sleep(2)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        place_items = soup.select('ul.placelist > li.PlaceItem')
        
        places_to_scrape = []
        for item in place_items:
            try:
                place_name = item.select_one('a.link_name').get_text(strip=True)
                comment_link_tag = item.select_one('a[href*="#comment"]')
                if comment_link_tag:
                    places_to_scrape.append({
                        "name": place_name,
                        "address": item.select_one('div.addr > p:nth-of-type(1)').get_text(strip=True),
                        "url": comment_link_tag['href'].split('#')[0]
                    })
            except AttributeError:
                continue
        
        # 검색 결과가 없으면 함수 종료
        if not places_to_scrape:
            return

        all_places_details = []
        for place in places_to_scrape:
            driver.get(place['url'] + '#comment')
            time.sleep(2)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            raw_comments = [p.get_text(strip=True) for p in soup.select('p.desc_review')]
            comments = []
            for text in raw_comments:
                if '...더보기' in text:
                    end_marker_pos = -1
                    for marker in ['.', '!', '?', '요']:
                        pos = text.rfind(marker, 0, text.find('...더보기'))
                        if pos > end_marker_pos: end_marker_pos = pos
                    comments.append(text[:end_marker_pos + 1] if end_marker_pos != -1 else text.split('...더보기')[0])
                else:
                    comments.append(text)
            
            driver.get(place['url'] + '#review')
            time.sleep(2)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            ai_summary_div = soup.select_one('div.ai_summary')
            ai_summary_text = ", ".join([span.get_text(strip=True) for span in ai_summary_div.select('span.option_review')]) if ai_summary_div else ""
            
            all_places_details.append({
                "name": place['name'], "address": place['address'], 
                "comments": comments, "ai_summary": ai_summary_text
            })

        #with open(output_filename, 'w', encoding='utf-8') as f:
        #    json.dump(all_places_details, f, ensure_ascii=False, indent=4)

        with open(output_filename, 'w', encoding='utf-8') as f:
            # all_places_details 리스트의 각 항목(딕셔너리)을 순회합니다.
            for place_detail in all_places_details:
                # 각 딕셔너리를 JSON 문자열로 변환합니다.
                json_line = json.dumps(place_detail, ensure_ascii=False)
                # 변환된 문자열 한 줄과 줄바꿈 문자(\n)를 파일에 씁니다.
                f.write(json_line + '\n')

    except Exception as e:
        # tqdm 진행률 표시줄과 겹치지 않게 에러 메시지 형식 변경
        tqdm.write(f"*** 에러 발생: [{query}] | 사유: {e} ***")
    
    finally:
        driver.quit()

# ===================================================================
# 3. 메인 실행 블록
# ===================================================================
if __name__ == "__main__":
    # [추가] tqdm을 사용하기 위해 전체 작업 목록을 미리 생성
    all_queries = []
    for province, cities in ADMIN_AREAS.items():
        for city in cities:
            for category in CATEGORIES:
                province_short = province.replace("특별시", "").replace("광역시", "").replace("특별자치도", "").replace("특별자치시", "")
                search_query = f"{province_short} {city} {category}"
                all_queries.append(search_query)

    print(f"총 {len(all_queries)}개의 검색어에 대한 스크래핑을 시작합니다.")

    # [수정] tqdm을 적용하여 반복 실행
    for query in tqdm(all_queries, desc="전국 스크래핑 진행률"):
        scrape_query(query)
        # 카카오맵의 과도한 요청 방지를 위한 휴식 시간
        time.sleep(5) # 테스트를 위해 이전보다 시간을 약간 줄임

    print("\n\n🎉🎉🎉 모든 스크래핑 작업이 완료되었습니다. 🎉🎉🎉")