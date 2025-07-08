# 시계열 카테고리 데이터 변화점 탐지 라이브러리 (CPD Analysis Library)

이 라이브러리는 학술 논문 데이터와 같은 시계열 카테고리 데이터에서 의미 있는 변화가 발생한 시점(Change Point)을 탐지하고 시각화하기 위해 개발되었습니다. PELT 알고리즘과 디리클레-다항 분포 모델을 기반으로, 다양한 관점의 분석을 유연하게 수행할 수 있습니다.

---

## ## 주요 기능

- **유연한 데이터 전처리**: 협업 관계, 저자(1저자/교신저자), 연구 분야 등 다양한 기준의 분석 데이터 생성
- **다양한 시간 단위 집계**: 월(M), 분기(Q), 반기(2Q), 연도(Y) 단위 분석 지원
- **자동 카테고리 그룹화**: 상위 N개 카테고리와 나머지를 'Etc'로 묶어 분석을 단순화하는 기능
- **고급 변화점 탐지**: PELT 알고리즘과 디리클레 분포 모델을 사용한 통계적 변화점 탐지
- **유연한 패널티 설정**: 분석 민감도를 BIC, AIC 또는 가중치를 통해 자유롭게 조절
- **결과 시각화 및 저장**: 분석 결과를 그래프(PNG)와 데이터(CSV)로 손쉽게 저장

---

## ## 설치 방법

1.  **저장소 복제 (Git 사용 시)**
    ```bash
    git clone [저장소 URL]
    cd cpd_analysis_lib
    ```

2.  **라이브러리 설치**
    프로젝트 폴더의 루트 디렉토리에서 아래 명령어를 실행하여 라이브러리를 개발 모드로 설치합니다.
    ```bash
    pip install -e .
    ```

---

## ## 빠른 시작 (Quick Start)

라이브러리를 설치한 후, 아래와 같이 몇 줄의 코드만으로 분석을 실행할 수 있습니다.

```python
# 예제: 국가간 협업 관계를 반기별로 분석하고, 상위 5개 외에는 'Etc'로 그룹화

import pandas as pd
from cpdlib import create_analysis_data, run_pelt_analysis_and_plot, preprocessing

# 1. 원본 데이터 로드

filepath = 'AI_Semi.csv'
df = pd.read_csv(filepath, encoding='cp949')

# 2. 날짜 컬럼 기본 전처리
df['Month'] = df['Publication Date'].apply(preprocessing.extract_month)
rows_before = len(df)
df.dropna(subset=['Month'], inplace=True)
rows_after = len(df)
print(f"월 추출 성공 후 데이터 행 수: {rows_after}")
print(f"제거된 행 수: {rows_before - rows_after}")

# 'Publication Year'와 추출된 'Month'를 조합하여 'YearMonth' 생성
df['Month'] = df['Month'].astype(int)
df['YearMonth'] = pd.to_datetime(df['Publication Year'].astype(str) + '-' + df['Month'].astype(str)).dt.to_period('M')

# 3. 분석 데이터 생성
proportions_df = create_analysis_data(
    df=df,
    analysis_type='collaboration', 
    time_col='YearMonth', 
    category_col='Addresses', 
    time_unit='2Q', # 반기
    top_n=5         # 상위 5개 + Etc
)

# 3-2. 가공 데이터가 있는 경우(예제)
filepath_example = '/cpd-analysis-library/examples/proportions_example_M.csv' # 월간 예시 데이터
#filepath_example = '/cpd-analysis-library/examples/proportions_example_2Q.csv' # 반기 예시 데이터
proportions_df = pd.read_csv(filepath_example, encoding='cp949')

# 4. 변화점 분석 및 시각화 실행
run_pelt_analysis_and_plot(
    proportions_df=proportions_df,
    time_unit_name="Collaboration (Semi-annual, Top 5+Etc)",
    date_format_str="%Y-H",
    penalty_method='AIC', # AIC 기준 사용
    penalty_weight=1.2,   # 패널티 20% 강화
    output_plot_path='collaboration_analysis.png',
    output_csv_path='collaboration_changepoints.csv'
)
