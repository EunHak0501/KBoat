import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
from tqdm import tqdm

def get_tms_and_dayOrd(year):
    회차=4
    일차=1

    # URL 설정
    url = f'https://www.kboat.or.kr/contents/information/fixedChuljuPage.do?stndYear={year}&tms={회차}&dayOrd={일차}'

    # HTTP GET 요청
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 회차(tms)와 일차(dayOrd) 값 추출
    tms_options = soup.select('select[name="tms"] option')
    dayOrd_options = soup.select('select[name="dayOrd"] option')

    tms_list = [int(option['value']) for option in tms_options]
    dayOrd_list = [int(option['value']) for option in dayOrd_options]

    return tms_list, dayOrd_list

def crawl_race_info(year, tms, day_ord):
    # URL 설정
    url = f'https://www.kboat.or.kr/contents/information/fixedChuljuPage.do?stndYear={year}&tms={tms}&dayOrd={day_ord}'

    # HTTP GET 요청
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 경주 수 확인
    race_headers = soup.select('h4.titPlayChart')
    total_races = len(race_headers) // 2

    # 경주 정보 가져오기
    race_info_list = []

    # 각 경주에 대한 정보가 있는 부분 추출
    race_blocks = soup.select('.boxr-head.clearfix')
    for race_no in range(1, total_races + 1):
        race_block = race_blocks[race_no - 1]
        # 제 1경주와 같은 경주 번호 정보
        race_name = race_block.select_one('h4.titPlayChart').text.strip()

        # "일반 플라잉 2주회"와 같은 경주 종류 정보
        race_type_element = race_block.select_one('p.bg')
        race_type = race_type_element.text.strip() if race_type_element else "정보 없음"

        if "플라잉" in race_type:
            race_type = "플라잉"
        elif "온라인" in race_type:
            race_type = "온라인"

        # 경주 정보를 리스트에 저장
        race_info_list.append([
            year, tms, day_ord, race_no, race_type
        ])

    columns = [
        '연도', '회차', '일차', '경주번호', '경기종류'
    ]

    df = pd.DataFrame(race_info_list, columns=columns)
    return df

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python crawlling_entry.py <year>")
        sys.exit(1)

    year = int(sys.argv[1])
    tms_list, dayOrd_list = get_tms_and_dayOrd(year)
    tms_list = sorted(tms_list)
    dayOrd_list = sorted(dayOrd_list)
    # print(f'({tms_list[0]}회 {dayOrd_list[0]}일차 부터 {tms_list[-1]}회 {dayOrd_list[-1]}일차까지, 총 {len(tms_list)*len(dayOrd_list)}개 수집 시작')

    all_entries_data = []

    for tms in tqdm(tms_list, leave=False):
        for day_ord in range(1,4):
            try:
                df = crawl_race_info(year, tms, day_ord)
                all_entries_data.append(df)
            except Exception as e:
                print(f'{tms}회, {day_ord}일차 존재하지 않음')
                continue

    # 전체 데이터를 하나의 DataFrame으로 합치기
    final_df = pd.concat(all_entries_data, ignore_index=True)

    # CSV 파일로 저장

    final_df.to_csv(f'./crawlled_data/kboat_race_info_{year}.csv', index=False, encoding='utf-8-sig')
    print(f'{year}년 모든 출주표 정보 크롤링 완료')

### ex) python crawlling_race_info.py 2016