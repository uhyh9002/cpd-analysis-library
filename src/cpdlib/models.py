import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
import matplotlib.cm as cmx
from scipy.stats import multinomial
from scipy.special import gammaln
from scipy.special import digamma

# ==============================================================================
# 섹션 2: 변화점 탐지(PELT) 모델링 함수
# ==============================================================================

def dirichlet_loglikelihood(x, alpha):
    """디리클레 분포의 로그 우도(log-likelihood)를 계산합니다."""
    return np.sum(gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum((alpha - 1) * np.log(x), axis=1))

def gradient_ascent_dirichlet(alpha_init, x, learning_rate=0.01, max_iter=1000, tol=1e-7):
    """경사 상승법으로 디리클레 분포의 파라미터(alpha)를 최적화합니다."""
    alpha = alpha_init.copy()
    N = x.shape[0]
    log_pk_sum = np.sum(np.log(x), axis=0)

    for i in range(max_iter):
        alpha_prev = alpha.copy()
        grad = N * (digamma(np.sum(alpha)) - digamma(alpha)) + log_pk_sum

        # 파라미터가 양수이고 우도가 개선될 때까지 학습률을 조정하며 업데이트
        lr = learning_rate
        while True:
            alpha_new = alpha + lr * grad
            if np.all(alpha_new > 0) and dirichlet_loglikelihood(x, alpha_new) > dirichlet_loglikelihood(x, alpha):
                alpha = alpha_new
                break
            lr /= 2
            if lr < 1e-9: # 학습률이 너무 작아지면 업데이트 중단
                return alpha_prev

        # 수렴 확인
        if np.sum(np.abs(alpha - alpha_prev)) < tol:
            break

    return alpha

def pelt_dirichlet(data, penalty_val):
    """디리클레 모델을 비용 함수로 사용하는 PELT 알고리즘 (수정된 버전)"""
    n = len(data)
    F = np.full(n + 1, np.inf)
    F[0] = -penalty_val
    change_points_store = {0: []} # 각 t시점의 최적 분할점 리스트 저장
    R = [0]
    ini_alpha = np.ones(data.shape[1])

    print("PELT 알고리즘으로 변화점 탐지를 시작합니다...")
    for t_star in range(1, n + 1):
        if t_star % (n // 10 or 1) == 0:
            print(f"진행률: {int((t_star / n) * 100)}%")

        costs_from_R = []
        for t in R:
            segment_data = data[t:t_star]
            segment_cost = -2 * dirichlet_loglikelihood(segment_data, gradient_ascent_dirichlet(ini_alpha, segment_data)) if len(segment_data) > 0 else 0
            total_cost = F[t] + segment_cost + penalty_val
            costs_from_R.append((total_cost, t))

        min_cost, t_hat = min(costs_from_R, key=lambda x: x[0])
        F[t_star] = min_cost
        change_points_store[t_star] = change_points_store[t_hat] + [t_hat]

        # Pruning 단계
        R = [t for total_cost, t in costs_from_R if total_cost <= F[t_star]] + [t_star]

    print("변화점 탐지가 완료되었습니다.")
    final_cps = sorted(list(set(change_points_store[n])))
    return final_cps[1:] # 맨 앞의 0 제외
