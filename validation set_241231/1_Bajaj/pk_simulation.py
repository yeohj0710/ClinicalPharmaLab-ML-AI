import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp

# ----------------------------
# 1) 파라미터 설정
# ----------------------------
TVCL = 0.0094
TVV1 = 3.63
TVV2 = 2.78
TVQ = 0.0321
TVEMAX = -0.295

# 무작위효과 가정: ECL=0, EV1=0, EV2=0, EEMAX=0
# 따라서 CL(t) = (위 식) * exp(0) * (exp((TVEMAX * t^3.15)/(...)) + 0)

# 임상공변량
BW = 68.28
EGFR = 66.37
BPS = 1
SEX = 1
RAAS = 1

# 투여량 및 주입시간
amt = 240.0  # mg
tinf = 1.0  # hr

# 시뮬레이션 시간 범위
end_time = 336  # 336시간까지
delta = 1  # 1시간 간격 출력(원하면 더 잘게 해도 됨)


# ----------------------------
# 2) time-varying CL(t) 함수
# ----------------------------
EEMAX = 0.0


def CL_t(t):
    time_part = np.exp((TVEMAX * (t**3.15)) / ((1410.0**3.15) + (t**3.15)))

    base = (
        TVCL
        * (BW / 80.0) ** 0.566
        * (EGFR / 90.0) ** 0.186
        * np.exp(0.172 * BPS)
        * np.exp(0.165 * SEX)
        * np.exp(-0.125 * RAAS)
    )

    return base * (time_part + EEMAX)


# ----------------------------
# 3) 주입률(rate in) 함수
# ----------------------------
def rate_in(t):
    if 0 <= t <= tinf:
        return amt / tinf
    else:
        return 0.0


# ----------------------------
# 4) ODE 정의: dCENT/dt, dPERIPH/dt
# ----------------------------
def two_comp_ode(t, y):
    # y = [CENT, PERIPH]
    cent, peri = y
    # 계산 편의를 위해 미리 파라미터 호출
    cl = CL_t(t)
    v1 = TVV1 * (BW / 80.0) ** 0.597 * np.exp(0.152 * SEX)  # (EV1=0 가정)
    v2 = TVV2 * 1.0  # (EV2=0 가정)
    q = TVQ

    # dCENT/dt
    dcent_dt = rate_in(t) - (cl * cent / v1) - q * (cent / v1 - peri / v2)
    # dPERIPH/dt
    dperi_dt = q * (cent / v1 - peri / v2)

    return [dcent_dt, dperi_dt]


# ----------------------------
# 5) ODE 풀기
# ----------------------------
t_eval = np.arange(0, end_time + delta, delta)  # 0~336시간까지 1시간 간격
# 초기값: CENT=0(주입 전), PERIPH=0
y0 = [0.0, 0.0]

sol = solve_ivp(
    fun=two_comp_ode,
    t_span=[0, end_time],
    y0=y0,
    t_eval=np.arange(0, end_time + delta, delta),
    method="LSODA",
    rtol=1e-9,
    atol=1e-9,
)
# sol.y는 [CENT, PERIPH]가 행으로, 시간축이 열로 구성되어 있음
cent_vals = sol.y[0]
peri_vals = sol.y[1]

# CP = CENT / V1, 여기서 V1은 "시간마다" 달라질 수도 있지만,
#     보통 NONMEM에서 post-hoc은 TIME마다 CL만 달라지고, V1은 일정(=초기 계산값)하게 씁니다.
#     만약 SEX, BW 등이 바뀌면 그때 재계산하시면 됩니다.
v1_const = TVV1 * (BW / 80.0) ** 0.597 * np.exp(0.152 * SEX)
cp_vals = cent_vals / v1_const  # residual error(1+EPS(1))는 0 가정

# ----------------------------
# 6) 결과를 DataFrame으로
# ----------------------------
# 기존 DataFrame 생성 후 (CENT, PERIPH 등 불필요한 컬럼 없이)
df = pd.DataFrame({"time": t_eval, "CP": cp_vals})
df["BW"] = BW
df["EGFR"] = EGFR
df["SEX"] = SEX
df["RAAS"] = RAAS
df["BPS"] = BPS
df["amt"] = amt

df.insert(0, "ID", 1)
df.insert(0, "", np.arange(1, len(df) + 1))

# ----------------------------
# 7) 저장 및 확인
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(current_dir, "result")
os.makedirs(result_dir, exist_ok=True)

df.to_csv(os.path.join(result_dir, "result_demo.csv"), index=False)
print(df.head(10))
