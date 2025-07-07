import re 
import pandas as pd
import numpy as np

# --- 1. 국가 추출 함수 개선 ---
def find_countries_improved(affiliation_string):
    """
    주요 국가 목록을 기반으로 소속 국가를 모두 추출하는 함수
    """
    if not isinstance(affiliation_string, str):
        return set()

    # 탐지할 국가와 해당 국가를 찾기 위한 정규표현식 패턴 딕셔너리
    country_patterns = {
        'USA': r'\b(USA|U\.S\.A\.|United States)\b',
        'China': r'\b(China|People\'s R\. China|Peoples R China)\b',
        'Japan': r'\b(Japan)\b',
        'Germany': r'\b(Germany)\b',
        'U.K.': r'\b(U\.K\.|United Kingdom|UK)\b',
        'France': r'\b(France)\b',
        'Canada': r'\b(Canada)\b',
        'South Korea': r'\b(South Korea|Korea)\b'
        # 필요에 따라 다른 국가들을 추가할 수 있습니다.
    }

    countries_found = set()
    for country, pattern in country_patterns.items():
        if re.search(pattern, affiliation_string, re.IGNORECASE):
            countries_found.add(country)

    return countries_found

# --- 2. 카테고리 분류 함수 개선 ---
def categorize_paper_final(countries_set):
    """
    국가 정보를 바탕으로 논문을 6개의 세분화된 카테고리로 분류합니다.
    - US-China Collaboration: 미국과 중국이 모두 포함된 협력
    - US-Non China: 미국은 포함, 중국은 미포함, 제3국이 포함된 협력
    - China-Non US: 중국은 포함, 미국은 미포함, 제3국이 포함된 협력
    - US Only: 미국 단독 연구
    - China Only: 중국 단독 연구
    - Other: 그 외 모든 경우 (미국, 중국 모두 미포함)
    """
    has_us = 'USA' in countries_set
    has_cn = 'China' in countries_set
    num_countries = len(countries_set)

    if has_us and has_cn:
        return 'US-China Collaboration'
    elif has_us and not has_cn and num_countries > 1:
        return 'US-Non China'
    elif has_cn and not has_us and num_countries > 1:
        return 'China-Non US'
    elif has_us and num_countries == 1:
        return 'US Only'
    elif has_cn and num_countries == 1:
        return 'China Only'
    else:
        return 'Other'

#--- 지능형 월(Month) 추출  ---#
def extract_month(date_str):
    """
    비정형 날짜 문자열에서 월(Month)을 추출하는 함수
    """
    if not isinstance(date_str, str):
        return np.nan

    # 대문자로 변환하고 공백 제거하여 처리 용이하게 만듦
    s = date_str.upper().strip()

    # 1. 월(Month) 이름과 숫자 매핑
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }

    # 2. 계절(Season) 매핑 (북반구 기준 대표 월)
    season_map = {'SPR': 3, 'SUM': 6, 'FAL': 9, 'WIN': 12}

    # 3. 규칙 기반 월 추출
    # 3-1. 월 범위 (예: 'JAN-MAR', 'JUL-DEC') -> 시작 월을 대표로 사용
    range_match = re.search(r'([A-Z]{3})-[A-Z]{3}', s)
    if range_match and range_match.group(1) in month_map:
        return month_map[range_match.group(1)]

    # 3-2. 계절 (예: 'SPR', 'SUM')
    for season, month_num in season_map.items():
        if season in s:
            return month_num

    # 3-3. 일반적인 월 이름 포함 여부 (가장 확실한 방법)
    # 예: '12-OCT', 'SEPT', 'JULY15,' 등
    for month_name, month_num in month_map.items():
        if month_name in s:
            return month_num

    # 3-4. 모든 규칙에 해당하지 않으면 NaN 반환
    return np.nan

def create_analysis_data(df, analysis_type, time_col, category_col, 
                         time_unit='M', top_n=None, author_type=None):
    """
    범용 데이터 전처리 함수

    Args:
        df (pd.DataFrame): 원본 데이터프레임
        analysis_type (str): 분석 종류 ('collaboration', 'author', 'field')
        time_col (str): 시간 정보로 사용할 컬럼 이름 (예: 'YearMonth')
        category_col (str): 분류 기준으로 사용할 컬럼 이름 (예: 'Addresses', 'Research Areas')
        time_unit (str, optional): 집계할 시간 단위 ('M', 'Q', '2Q', 'Y'). 기본값 'M'.
        top_n (int, optional): 상위 N개 카테고리와 'Etc'로 그룹화. None이면 모두 사용.
        author_type (str, optional): 저자 분석 시 타입 지정 ('first', 'corresponding').
    
    Returns:
        pd.DataFrame: 분석에 사용할 수 있는 비율(proportion) 데이터프레임
    """
    print(f"\n>>> '{analysis_type}' 유형 분석을 '{time_unit}' 단위로 시작합니다...")
    
    # 1. 분석 유형에 따른 카테고리 생성
    if analysis_type == 'collaboration':
        df['countries'] = df[category_col].apply(find_countries_improved)
        df['analysis_category'] = df['countries'].apply(categorize_paper_final)
    
    elif analysis_type == 'author':
        if author_type == 'first':
            target_col = df[category_col].str.split(';', n=1, expand=True)[0]
        elif author_type == 'corresponding':
            # 'Reprint Addresses' 컬럼이 있다는 가정 하에 진행
            target_col = df['Reprint Addresses'] 
        else:
            raise ValueError("'author_type'은 'first' 또는 'corresponding'이어야 합니다.")
            
        countries = target_col.apply(find_countries_improved)
        df['analysis_category'] = countries.apply(lambda s: next(iter(s)) if s else None)
        
    elif analysis_type == 'field':
        df_exploded = df[[time_col, category_col]].dropna().copy()
        df_exploded['analysis_category'] = df_exploded[category_col].str.strip().str.split(';')
        df = df_exploded.explode('analysis_category')
        df['analysis_category'] = df['analysis_category'].str.strip()
        
    else:
        raise ValueError("'analysis_type'은 'collaboration', 'author', 'field' 중 하나여야 합니다.")
        
    df.dropna(subset=['analysis_category'], inplace=True)
    
    # 2. 시간 단위별 집계
    counts = df.groupby(time_col)['analysis_category'].value_counts().unstack(fill_value=0)
    
    if time_unit != 'M':
        # 월별(M)이 아닌 다른 시간 단위일 경우 재집계
        grouper = {
            'Q': lambda p: p.to_timestamp().to_period('Q'),
            '2Q': lambda p: f"{p.year}-H{(p.month-1)//6 + 1}",
            'Y': lambda p: p.year
        }[time_unit]
        counts = counts.groupby(grouper).sum()

    # 3. 상위 N개 + Etc 그룹화 (top_n이 지정된 경우)
    if top_n:
        total_counts = counts.sum().sort_values(ascending=False)
        top_categories = total_counts.nlargest(top_n).index.tolist()
        print(f"상위 {top_n}개 카테고리: {top_categories}")
        
        grouped_counts = counts[top_categories].copy()
        other_categories = [col for col in counts.columns if col not in top_categories]
        if other_categories:
            grouped_counts['Etc'] = counts[other_categories].sum(axis=1)
        counts = grouped_counts

    # 4. 최종 비율 데이터프레임 반환
    epsilon = 1e-9
    proportions = counts.apply(lambda x: (x + epsilon) / (x.sum() + epsilon * len(x)), axis=1)
    
    print("분석 데이터 생성이 완료되었습니다.")
    return proportions
