import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# 1. 더미 데이터셋 정의 (PK: 지수 감쇠, PD: PK 합산 효과에 노이즈 추가)
class DummyPKPDDataset(Dataset):
    def __init__(self, n_samples=200, t_max=10.0, num_timesteps=50):
        super().__init__()
        self.n_samples = n_samples
        self.t_max = t_max
        self.num_timesteps = num_timesteps
        self.times = torch.linspace(0, t_max, num_timesteps)
        # 정적 특성: 예를 들어, [나이, 체중, 투여량]
        self.ages = torch.randint(20, 80, (n_samples,))  # 20~80세
        self.weights = torch.randint(50, 100, (n_samples,))  # 50~100kg
        self.doses = torch.randint(50, 500, (n_samples,))  # 50~500mg
        self.features = torch.stack(
            [self.ages, self.weights, self.doses], dim=1
        ).float()
        doses = self.features[:, 2].unsqueeze(1)  # 투여량으로 사용
        # PK: 투여량에 비례한 지수 감쇠 + 약간의 노이즈
        self.pk_targets = doses * torch.exp(
            -self.times.unsqueeze(0) / 3.0
        ) + 0.1 * torch.randn(n_samples, num_timesteps)
        # PD: PK 시계열의 누적합(간단한 가정) + 노이즈
        self.pd_targets = torch.sum(
            self.pk_targets, dim=1, keepdim=True
        ) + 0.1 * torch.randn(n_samples, 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 각 샘플은 (정적 피처, 시간 벡터, PK 타깃, PD 타깃)로 구성됨
        return (
            self.features[idx],
            self.times,
            self.pk_targets[idx],
            self.pd_targets[idx],
        )


# 2. ODE 함수 정의: 은닉 상태의 연속적인 변화율을 학습
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t, h):
        return self.net(h)


# 3. Neural ODE를 활용한 PK/PD 예측 모델 정의
class NeuralODEPKPD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralODEPKPD, self).__init__()
        # 정적 입력을 은닉 상태로 매핑
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        # PK 예측 헤드: 각 시간 점에서의 혈중 농도 예측 (출력 차원 1)
        self.fc_pk = nn.Linear(hidden_dim, output_dim)
        # PD 예측 헤드: 최종 은닉 상태로부터 하나의 스칼라값 예측
        self.fc_pd = nn.Linear(hidden_dim, 1)

    def forward(self, static_input, times):
        # 초기 은닉 상태 h0 계산 (배치 크기: [B, hidden_dim])
        h0 = self.fc_in(static_input)
        # ODE 솔버를 사용하여 연속 시간상에서 은닉 상태를 계산
        # 결과 h_ts: [T, B, hidden_dim] (T: 시간 스텝 수)
        h_ts = odeint(self.odefunc, h0, times)
        # 차원 변경: [B, T, hidden_dim]
        h_ts = h_ts.transpose(0, 1)
        # PK 예측: 각 시간 스텝마다 fc_pk를 적용 → [B, T, 1]
        pk_pred = self.fc_pk(h_ts)
        # PD 예측: 마지막 시간 스텝의 은닉 상태를 이용 → [B, 1]
        pd_pred = self.fc_pd(h_ts[:, -1, :])
        return pk_pred, pd_pred


# 4. 학습 루프 (에포크별 손실 기록 추가)
def train(model, dataset, n_epochs=500, batch_size=16, lr=1e-3, device="cpu"):
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    epoch_losses = []  # 에포크별 손실 기록

    for epoch in range(n_epochs):
        total_loss = 0.0
        for static_input, times, pk_target, pd_target in dataloader:
            static_input = static_input.to(device)
            # times가 [B, T]일 경우, 첫 번째 배치의 값만 사용 (모든 샘플에 동일한 시간 벡터라고 가정)
            if times.ndim > 1:
                times = times[0]
            times = times.to(device)  # times: [T]
            pk_target = pk_target.to(device)
            pd_target = pd_target.to(device)

            optimizer.zero_grad()
            pk_pred, pd_pred = model(static_input, times)
            # pk_pred: [B, T, 1] → pk_target: [B, T] 이므로 차원 맞춤
            loss_pk = criterion(pk_pred.squeeze(-1), pk_target)
            loss_pd = criterion(pd_pred, pd_target)
            loss = loss_pk + loss_pd
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    return epoch_losses


# 5. 데이터셋 및 모델 생성 후 학습 실행
dataset = DummyPKPDDataset(n_samples=200)
model = NeuralODEPKPD(input_dim=3, hidden_dim=16, output_dim=1)
epoch_losses = train(model, dataset, n_epochs=500, batch_size=16, lr=1e-3)


##########################################
# 6. 시각화
##########################################
import random

# 6-1. 에포크별 학습 손실 변화 그래프
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Training Loss per Epoch")
plt.grid(True)
plt.show()


# 6-2. 학습 데이터 중 일부 샘플의 정적 특성과 PD 타깃 (표 형태)
# 예시: 데이터셋에서 랜덤 5개 샘플 추출
num_samples_to_show = 5
sample_indices = random.sample(range(len(dataset)), num_samples_to_show)
sample_data = [dataset[i] for i in sample_indices]

# 정적 특성과 PD 타깃을 리스트로 정리
table_data = []
for idx, (static_input, _, _, pd_target) in zip(sample_indices, sample_data):
    # 정적 특성: [나이, 체중, 투여량]으로 가정
    table_data.append(
        [
            f"{idx}",
            f"{static_input[0].item():.2f}",
            f"{static_input[1].item():.2f}",
            f"{static_input[2].item():.2f}",
            f"{pd_target.item():.2f}",
        ]
    )

# 컬럼명 설정
column_labels = ["Index", "Age", "Weight", "Dose", "PD Target"]

plt.figure(figsize=(6, 2))
plt.axis("tight")
plt.axis("off")
the_table = plt.table(cellText=table_data, colLabels=column_labels, loc="center")
plt.title("Sample Training Data (Static Features and PD Target)")
plt.show()


# 6-3. 테스트할 사람의 데이터와 모델 예측 결과 비교 (PK 시계열 및 PD)
# 예시: 데이터셋에서 첫 번째 샘플 사용
static_input, times, pk_target, pd_target = dataset[0]
static_input = static_input.unsqueeze(0)  # 배치 차원 추가

with torch.no_grad():
    pk_pred, pd_pred = model(static_input, times)

# PK 시계열 비교 (실제 vs 예측)
times_np = times.cpu().numpy()
pk_target_np = pk_target.cpu().numpy()  # [T]
pk_pred_np = pk_pred.squeeze(-1).cpu().numpy()[0]  # [T]

plt.figure(figsize=(8, 4))
plt.plot(times_np, pk_target_np, label="PK Target", marker="o")
plt.plot(times_np, pk_pred_np, label="PK Prediction", marker="x")
plt.xlabel("Time")
plt.ylabel("PK Value")
plt.title("PK Time Series: Target vs Prediction")
plt.legend()
plt.grid(True)
plt.show()

# PD 값 비교 (바 차트)
pd_target_val = pd_target.item()
pd_pred_val = pd_pred.item()

plt.figure(figsize=(4, 4))
plt.bar(
    ["PD Target", "PD Prediction"],
    [pd_target_val, pd_pred_val],
    color=["blue", "orange"],
)
plt.title("PD Comparison")
plt.ylabel("PD Value")
plt.show()
