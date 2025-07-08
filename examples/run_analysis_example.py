import pandas as pd
from cpdlib import create_analysis_data, run_pelt_analysis_and_plot, preprocessing

# 분석 데이터 읽기
filepath_example = '/content/cpd-analysis-library/examples/proportions_example_M.csv' # 월간 예시 데이터 - 경로는 직접 지정
proportions_df = pd.read_csv(filepath_example, encoding='cp949')

# 변화점 분석 및 시각화 실행
run_pelt_analysis_and_plot(
    proportions_df=proportions_df,
    time_unit_name="Collaboration (Semi-annual, Top 5+Etc)",
    date_format_str="%Y-H",
    penalty_method='AIC', # AIC 기준 사용
    penalty_weight=1.2,   # 패널티 20% 강화
    output_plot_path='collaboration_analysis.png',
    output_csv_path='collaboration_changepoints.csv'
)
