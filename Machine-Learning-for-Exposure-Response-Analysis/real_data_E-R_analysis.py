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
def weibull_function(cI, scale=10):
    return 10 * (cI / scale) ** 2 * np.exp(-((cI / scale) ** 3))


# Hill 함수 정의
def hill_function(tI, Emax=3, EC50=10):
    return (Emax * tI) / (EC50 + tI)


# Biased_Data.csv 파일 읽기
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Biased_Data.csv")
data = pd.read_csv(file_path)

# 유도 단계 데이터 생성
# 변수 정의
# hI: 유도 단계 초기 건강 상태
# xI1, xI2, xI3, xI4: 유도 단계 공변량
# cI: 유도 단계 교란 변수
# rI: 유도 단계 치료 무작위화 변수 (0=위약군, 1=치료군)
# tI: 유도 단계 약물 노출
# hM: 유도 단계 종료 시 건강 상태
# y1: 유도 단계 이진 결과 변수 (0 or 1)
data_induction = data.copy()
data_induction["hI"] = data_induction["HI"]
data_induction["xI1"] = data_induction["XI1"]
data_induction["xI2"] = np.log(data_induction["XI2"].clip(lower=1e-5))
data_induction["xI3"] = np.log(data_induction["XI3"].clip(lower=1e-5))
data_induction["xI4"] = np.log(data_induction["XI4"].clip(lower=1e-5))
data_induction["cI"] = np.log(data_induction["CI"].clip(lower=1e-5))
data_induction["rI"] = (np.random.rand(len(data_induction)) > 0.5).astype(int)
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
# xM1, xM2, xM3: 유지 단계 공변량 (Biased_Data.csv에서 읽어오기)
# cM: 유지 단계 교란 변수
# rM: 유지 단계 치료 무작위화 변수 (0=위약군, 1=치료군)
# tM: 유지 단계 약물 노출
# hF: 유지 단계 종료 시 건강 상태
# y2: 유지 단계 이진 결과 변수 (0 or 1)
data_maintenance = pd.DataFrame(
    {
        "xM1": data["xM1"],
        "xM2": data["xM2"],
        "xM3": data["xM3"],
        "cM": np.log(data["CM"].clip(lower=1e-5)),
        "rM": (np.random.rand(len(data)) > 0.5).astype(int),
        "tM": np.where(
            (np.random.rand(len(data)) > 0.5).astype(int) == 1,
            np.log(data["TM"].clip(lower=1e-5)),
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
model_induction = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    max_depth=6,  # 트리의 최대 깊이
    learning_rate=0.01,  # 학습률
    colsample_bytree=0.8,  # 트리당 샘플링 피처 비율
    subsample=0.8,  # 샘플링 비율
    n_estimators=100,  # 트리 개수
    gamma=1,  # 분할 기준 강화
    scale_pos_weight=1.5,  # 클래스 불균형 보정
)
X_induction = data_induction[["hI", "xI1", "xI2", "xI3", "xI4", "rI", "cI", "tI"]]
y_induction = data_induction["y1"]
scores_I = cross_val_score(
    model_induction, X_induction, y_induction, cv=kf, scoring="accuracy"
)
print(f"유도 단계 교차 검증 정확도: {scores_I}")
print(f"유도 단계 평균 정확도: {scores_I.mean():.2f}")

# 유지 단계 모델 정의 및 데이터 준비
model_maintenance = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    max_depth=6,
    learning_rate=0.01,
    colsample_bytree=0.8,
    subsample=0.8,
    n_estimators=100,
    gamma=1,
    scale_pos_weight=1.5,
)
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
