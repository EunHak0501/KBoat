import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# URL 정의
url = "https://www.kboat.or.kr/contents/information/raceResultList.do?stndYear=2016&tms=1&dayOrd=1"

# HTTP GET 요청
response = requests.get(url)

# BeautifulSoup을 이용하여 HTML 파싱
soup = BeautifulSoup(response.content, 'html.parser')

# 결과 데이터를 담을 리스트
data = []

# 테이블 찾기
table = soup.find('table', {'class': 'tblType1'})

# 테이블의 모든 행(row) 추출
rows = table.find_all('tr')

# 각 행에서 데이터 추출
for row in rows[1:]:  # 첫 행은 헤더이므로 제외
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]

    # 연도, 회차, 일차는 URL이나 수작업으로 입력 (여기서는 2016년, 1회차, 1일차로 고정)
    year = "2016"
    round_num = "1"
    day_num = "1"

    # 경주 데이터 추출
    race = cols[0]  # 경주 번호
    first_place = cols[1]  # 1위
    second_place = cols[2]  # 2위
    third_place = cols[3]  # 3위
    dansung = cols[4]  # 단승식
    yeonsung = cols[5]  # 연승식
    ssangsung = cols[6]  # 쌍승식
    boksung = cols[7]  # 복승식
    samboksung = cols[8]  # 삼복승식

    # 데이터 리스트에 추가
    data.append([year, round_num, day_num, race, first_place, second_place, third_place, dansung, yeonsung, ssangsung, boksung, samboksung])

# Pandas DataFrame으로 변환
columns = ['연도', '회차', '일차', '경주', '1위', '2위', '3위', '단승식', '연승식', '쌍승식', '복승식', '삼복승식']
df = pd.DataFrame(data, columns=columns)

# 결과 출력
print(df)




# 데이터프레임을 CSV 파일로 저장 (선택 사항)
# df.to_csv('race_results.csv', index=False, encoding='utf-8-sig')
