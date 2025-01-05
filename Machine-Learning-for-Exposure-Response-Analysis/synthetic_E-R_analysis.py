# 필요한 라이브러리 불러오기
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import warnings
import os

# 경고 메시지 무시
warnings.filterwarnings(action="ignore", category=UserWarning, module="xgboost")


# Weibull 함수 정의
def weibull_function(cI, scale=3):
    return 10 * (cI / scale) ** 2 * np.exp(-((cI / scale) ** 3))


# Hill 함수 정의
def hill_function(tI, Emax=3, EC50=10):
    return (Emax * tI) / (EC50 + tI)


# Nearest Positive Semi-Definite Matrix 계산 함수
def nearest_positive_semi_definite(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


# 데이터 생성
mean_vector = np.array(
    [
        -2,
        np.log(5),
        np.log(5),
        np.log(5),
        np.log(5),
        np.log(2),
        np.log(2),
        np.log(5),
        np.log(5),
    ]
)
cov_matrix = np.array(
    [
        [0.1, -0.1, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3, 0.3],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.2, 0.0, 0.1],
    ]
)

# Positive semi-definite covariance matrix 수정
cov_matrix = nearest_positive_semi_definite(cov_matrix)

# 유도 단계 데이터 생성

# 변수 정의
# HI: 초기 건강 상태 (health status)
# XI1, XI2, XI3, XI4: 공변량 (예: 나이, BMI, 성별, 혈압)
# CI: 유도 단계 교란 변수
# CM: 유지 단계 교란 변수
# TI: 유도 단계 약물 노출
# TM: 유지 단계 약물 노출
# hI: 유도 단계 초기 건강 상태
# xI1, xI2, xI3, xI4: 유도 단계 공변량
# cI: 유도 단계 교란 변수
# rI: 유도 단계 치료 무작위화 변수 (0=위약군, 1=치료군)
# tI: 유도 단계 약물 노출
# hM: 유도 단계 종료 시 건강 상태
# y1: 유도 단계 이진 결과 변수 (0 or 1)

samples = np.random.multivariate_normal(mean_vector, cov_matrix, size=1000)
data_induction = pd.DataFrame(
    samples, columns=["HI", "XI1", "XI2", "XI3", "XI4", "CI", "CM", "TI", "TM"]
)
data_induction["hI"] = data_induction["HI"]
data_induction["xI1"] = data_induction["XI1"]
data_induction["xI2"] = np.log(data_induction["XI2"].clip(lower=1e-5))
data_induction["xI3"] = np.log(data_induction["XI3"].clip(lower=1e-5))
data_induction["xI4"] = np.log(data_induction["XI4"].clip(lower=1e-5))
data_induction["cI"] = np.log(data_induction["CI"].clip(lower=1e-5))
data_induction["rI"] = (np.random.rand(1000) > 0.5).astype(int)
data_induction["tI"] = np.where(
    data_induction["rI"] == 1, np.log(data_induction["TI"].clip(lower=1e-5)), 0
)
data_induction["hM"] = (
    data_induction["hI"]
    + weibull_function(data_induction["cI"])
    + hill_function(data_induction["tI"])
    - 1
)
data_induction["y1"] = np.random.binomial(1, 1 / (1 + np.exp(-data_induction["hM"])))

# 유지 단계 데이터 생성
# 변수 정의
# xM1, xM2, xM3: 유지 단계 공변량 (유도 단계 공변량에 약간의 노이즈 추가)
# cM: 유지 단계 교란 변수
# rM: 유지 단계 치료 무작위화 변수 (0=위약군, 1=치료군)
# tM: 유지 단계 약물 노출
# hF: 유지 단계 종료 시 건강 상태
# y2: 유지 단계 이진 결과 변수 (0 or 1)

data_maintenance = pd.DataFrame(
    {
        "xM1": data_induction["xI1"] + np.random.normal(0, 0.01, 1000),
        "xM2": data_induction["xI2"] + np.random.normal(0, 0.01, 1000),
        "xM3": data_induction["xI3"] + np.random.normal(0, 0.01, 1000),
        "cM": np.log(data_induction["CM"].clip(lower=1e-5)),
        "rM": (np.random.rand(1000) > 0.5).astype(int),
        "tM": np.where(
            (np.random.rand(1000) > 0.5).astype(int) == 1,
            np.log(data_induction["TM"].clip(lower=1e-5)),
            0,
        ),
    }
)
data_maintenance["hF"] = (
    data_induction["hM"]
    + weibull_function(data_maintenance["cM"])
    + hill_function(data_maintenance["tM"], Emax=2)
    - 1
)
data_maintenance["y2"] = np.random.binomial(
    1, 1 / (1 + np.exp(-data_maintenance["hF"]))
)

# 모델 학습 및 검증
kf = KFold(n_splits=5, shuffle=True)
model_induction = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
X_induction = data_induction[["hI", "xI1", "xI2", "xI3", "xI4", "rI", "cI", "tI"]]
y_induction = data_induction["y1"]
scores_I = cross_val_score(
    model_induction, X_induction, y_induction, cv=kf, scoring="accuracy"
)
print(f"유도 단계 교차 검증 정확도: {scores_I}")
print(f"유도 단계 평균 정확도: {scores_I.mean():.2f}")

# 유지 단계 모델 정의 및 데이터 준비
model_maintenance = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
X_maintenance = data_maintenance[["xM1", "xM2", "xM3", "cM", "rM", "tM"]]
y_maintenance = data_maintenance["y2"]
scores_M = cross_val_score(
    model_maintenance, X_maintenance, y_maintenance, cv=kf, scoring="accuracy"
)
print(f"유지 단계 교차 검증 정확도: {scores_M}")
print(f"유지 단계 평균 정확도: {scores_M.mean():.2f}")

# SHAP 분석 (유도 단계)
model_induction.fit(X_induction, y_induction)
explainer_I = shap.Explainer(model_induction, X_induction)
shap_values_I = explainer_I(X_induction)
shap.summary_plot(shap_values_I, X_induction, show=False)
plt.title("Induction Stage SHAP Summary Plot")
plt.savefig(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "induction_stage_shap_summary.png"
    )
)
plt.show()

# SHAP 분석 (유지 단계)
model_maintenance.fit(X_maintenance, y_maintenance)
explainer_M = shap.Explainer(model_maintenance, X_maintenance)
shap_values_M = explainer_M(X_maintenance)
shap.summary_plot(shap_values_M, X_maintenance, show=False)
plt.title("Maintenance Stage SHAP Summary Plot")
plt.savefig(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "maintenance_stage_shap_summary.png"
    )
)
plt.show()
