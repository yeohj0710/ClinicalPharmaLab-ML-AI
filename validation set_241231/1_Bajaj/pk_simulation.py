import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp

# ----------------------------
# 1) 파라미터 설정
# ----------------------------
# 기본 파라미터 (모든 단위 포함)
TVCL = 0.0094  # CL의 기본값 (L/h)
TVV1 = 3.63  # 중심구획의 분포용적 V1 (L)
TVV2 = 2.78  # 말초구획의 분포용적 V2 (L)
TVQ = 0.0321  # 구획 간 약물 이동속도 Q (L/h)
TVEMAX = -0.295  # 시간에 따른 CL 변화 조정 (무단위)

# 환자 공변량
BW = 68.28  # 체중 (kg)
EGFR = 66.37  # 신사구체 여과율 (eGFR, mL/min/1.73m²)
BPS = 1  # BPS 상태 (0 = 없음, 1 = 있음)
SEX = 1  # 성별 (0 = 여성, 1 = 남성)
RAAS = 1  # RAAS 활성 여부 (0 = 없음, 1 = 있음)

# 투여량 및 주입시간
amt = 240.0  # mg
tinf = 1.0  # hr

# 시뮬레이션 시간 범위
end_time = 336  # 시뮬레이션 종료 시간 (hr)
delta = 1  # 출력 시간 간격 (hr)

# ----------------------------
# 2) 난류항(개체간 변동) 샘플링
# ----------------------------
# NONMEM의 OMEGA 블록으로부터 추정한 공분산 행렬
cov_matrix = np.array([[0.123, 0.0432, 0.0], [0.0432, 0.123, 0.0], [0.0, 0.0, 0.258]])

# 난류항 샘플링 (평균=0, 공분산=cov_matrix)
ECL_rand, EV1_rand, EV2_rand = np.random.multivariate_normal([0, 0, 0], cov_matrix)

# EEMAX는 단일 정규분포로 샘플링
EEMAX_rand = np.random.normal(0, np.sqrt(0.0719))

# ----------------------------
# 3) 추가 투여 스케줄 설정
# ----------------------------
# 투여 시작 시간 리스트 (hr)
dose_times = [0, 336, 672]  # 0시간과 336시간에 투여


def rate_in(t):
    """
    특정 시간 t에 주입률(rate in)을 반환.
    투여 시작 시간 dose_times 리스트에 있는 시간 동안만 amt/tinf 만큼 주입.
    """
    for dstart in dose_times:
        if dstart <= t < (dstart + tinf):
            return amt / tinf  # mg/hr
    return 0.0


# ----------------------------
# 4) CL(t) 계산 함수 정의
# ----------------------------
def CL_t(t):
    """
    시간 t에 따른 CL을 계산.
    개체간 변동 반영: EEMAX_rand, ECL_rand
    """
    # 시간에 따른 CL 변화 부분
    time_part = np.exp((TVEMAX * (t**3.15)) / ((1410.0**3.15) + (t**3.15)))

    # 환자 공변량에 따른 기본 CL
    base = (
        TVCL
        * (BW / 80.0) ** 0.566
        * (EGFR / 90.0) ** 0.186
        * np.exp(0.172 * BPS)
        * np.exp(0.165 * SEX)
        * np.exp(-0.125 * RAAS)
    )

    # CL(t) 계산: base * (time_part + EEMAX_rand) * exp(ECL_rand)
    return base * (time_part + EEMAX_rand) * np.exp(ECL_rand)


# ----------------------------
# 5) ODE 정의: dCENT/dt, dPERIPH/dt
# ----------------------------
def two_comp_ode(t, y):
    """
    두 구획 모델의 ODE 정의.
    y = [CENT, PERIPH]
    """
    cent, peri = y

    # 시간에 따른 CL 계산
    cl = CL_t(t)  # L/h

    # V1과 V2에 대한 개체간 변동 반영
    v1 = TVV1 * (BW / 80.0) ** 0.597 * np.exp(0.152 * SEX) * np.exp(EV1_rand)  # L
    v2 = TVV2 * np.exp(EV2_rand)  # L

    q = TVQ  # L/h

    # 중심구획 농도의 변화율 (dCENT/dt)
    dcent_dt = rate_in(t) - (cl * cent / v1) - q * (cent / v1 - peri / v2)

    # 말초구획 농도의 변화율 (dPERIPH/dt)
    dperi_dt = q * (cent / v1 - peri / v2)

    return [dcent_dt, dperi_dt]


# ----------------------------
# 6) ODE 풀기 및 시뮬레이션
# ----------------------------
# 시뮬레이션 시간 포인트 설정
t_eval = np.arange(0, end_time + tinf + delta, delta)  # 0~337시간까지 (tinf=1 포함)

# 초기값: CENT=0, PERIPH=0
y0 = [0.0, 0.0]

# ODE 풀기
sol = solve_ivp(
    fun=two_comp_ode,
    t_span=[0, end_time + tinf],  # 0부터 337시간까지
    y0=y0,
    t_eval=t_eval,
    method="LSODA",
    rtol=1e-9,
    atol=1e-9,
)

# ODE 결과
cent_vals = sol.y[0]  # CENT 농도
peri_vals = sol.y[1]  # PERIPH 농도

# ----------------------------
# 7) CP 계산
# ----------------------------
# V1 계산 (동일한 V1 사용, 개체간 변동 반영됨)
v1_const = TVV1 * (BW / 80.0) ** 0.597 * np.exp(0.152 * SEX) * np.exp(EV1_rand)  # L

# CP = CENT / V1 (잔차변동 없음, EPS=0)
cp_vals = cent_vals / v1_const  # mg/L

# ----------------------------
# 8) 결과를 DataFrame으로 정리
# ----------------------------
df = pd.DataFrame(
    {
        "ID": 1,  # 단일 개체 ID
        "Time": sol.t,  # 시간 (hr)
        "CP": cp_vals,  # 혈중 농도 (mg/L)
        "BW": BW,
        "EGFR": EGFR,
        "SEX": SEX,
        "RAAS": RAAS,
        "BPS": BPS,
        "amt": amt,
    }
)

# ----------------------------
# 9) 결과 저장 및 확인
# ----------------------------
# 결과를 저장할 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리
result_dir = os.path.join(current_dir, "result")
os.makedirs(result_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# CSV 파일로 저장
output_path = os.path.join(result_dir, "result_demo.csv")
df.to_csv(output_path, index=False)

# 결과 일부 출력
print(df.head(10))
print(f"\n전체 결과가 {output_path}에 저장되었습니다.")
