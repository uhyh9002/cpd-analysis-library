# ==============================================================================
# 섹션 3: 분석 및 시각화 실행 함수
# ==============================================================================
from .models import pelt_dirichlet

def run_pelt_analysis_and_plot(proportions_df, time_unit_name, date_format_str, 
                               penalty_method='BIC', penalty_weight=1.0, 
                               output_plot_path=None, output_csv_path=None):
    """
    [최종 버전] 패널티 방법(BIC/AIC)과 가중치를 선택하여 분석 유연성을 높인 함수

    Args:
        proportions_df (pd.DataFrame): 비율 데이터
        time_unit_name (str): 그래프 제목에 사용할 시간 단위
        date_format_str (str): 결과 출력에 사용할 날짜 형식
        penalty_method (str, optional): 사용할 패널티 기준 ('BIC' 또는 'AIC'). 기본값은 'BIC'.
        penalty_weight (float, optional): 계산된 패널티에 적용할 가중치. 기본값은 1.0.
        output_plot_path (str, optional): 그래프를 저장할 PNG 파일 경로.
        output_csv_path (str, optional): 변화점 데이터를 저장할 CSV 파일 경로.
    """
    if proportions_df is None or proportions_df.empty:
        print(f"{time_unit_name} 데이터가 비어있어 분석을 건너뜁니다.")
        return

    print(f"\n--- {time_unit_name} 데이터 분석 시작 ---")
    data_matrix = proportions_df.values
    n, K = data_matrix.shape
    
    # --- [수정] 패널티 계산 로직 ---
    base_penalty = 0
    if penalty_method.upper() == 'BIC':
        base_penalty = K * np.log(n)
        print(f"패널티 기준: BIC (기본값: {base_penalty:.2f})")
    elif penalty_method.upper() == 'AIC':
        base_penalty = K * 2
        print(f"패널티 기준: AIC (기본값: {base_penalty:.2f})")
    else:
        print(f"경고: 알 수 없는 패널티 기준 '{penalty_method}'. BIC를 기본값으로 사용합니다.")
        base_penalty = K * np.log(n)
        penalty_method = 'BIC'

    # 가중치를 적용한 최종 패널티 계산
    penalty = base_penalty * penalty_weight
    print(f"적용된 가중치: {penalty_weight}")
    print(f"최종 패널티 값: {penalty:.2f}")
    # --------------------------------

    print(f"데이터 포인트 수(n): {n}, 카테고리 수(K): {K}")
    detected_cps_indices = pelt_dirichlet(data_matrix, penalty_val=penalty)

    # --- 이후 로직은 이전과 동일 ---
    print("\n--- 최종 분석 결과 ---")
    is_datetime_like = isinstance(proportions_df.index, (pd.PeriodIndex, pd.DatetimeIndex))
    valid_indices = [i for i in detected_cps_indices if i < n] if detected_cps_indices else []
    
    if not valid_indices:
        print("탐지된 변화점이 없습니다.")
    else:
        print("탐지된 변화점:")
        if is_datetime_like:
            detected_dates = proportions_df.index[valid_indices].to_timestamp()
            for date in detected_dates:
                print(date.strftime(date_format_str))
        else:
            detected_labels = proportions_df.index[valid_indices]
            for label in detected_labels:
                print(label)
    
    if output_csv_path and valid_indices:
        cps_df = pd.DataFrame({'ChangePointIndex': valid_indices, 'ChangePointLabel': proportions_df.index[valid_indices]})
        cps_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 변화점 데이터가 '{output_csv_path}' 경로에 저장되었습니다.")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))
    proportions_df.plot(ax=ax, colormap='viridis', linewidth=2)

    if valid_indices:
        if is_datetime_like:
            detected_dates_for_plot = proportions_df.index[valid_indices].to_timestamp()
            for date in detected_dates_for_plot:
                ax.axvline(x=date, color='crimson', linestyle='--', linewidth=2.5, label=f'Change Point ({date.strftime(date_format_str)})')
        else:
            for cp_index in valid_indices:
                label_text = proportions_df.index[cp_index]
                ax.axvline(x=cp_index, color='crimson', linestyle='--', linewidth=2.5, label=f'Change Point ({label_text})')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12)
    ax.set_title(f'{time_unit_name} Proportions with Change Points', fontsize=18)
    ax.set_xlabel(f'Time ({time_unit_name})', fontsize=14)
    ax.set_ylabel('Proportion', fontsize=14)
    plt.tight_layout()

    if output_plot_path:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ 그래프가 '{output_plot_path}' 경로에 저장되었습니다.")
    
    plt.show()
