"""
System 2: 공간-시간 역학 추정 계층 (Spatial-Temporal Dynamic Modeling Layer - Neural ODE)
=========================================================================================
구성요소:
1. `SpatialBackbone`: EfficientNet-B4 기반 고해상도 안저 공간 특징 추출기
2. `LatentSpaceProjector`: Linear & LayerNorm (d=64) 잠재 공간 투영기
3. `RetinalODEFunc` + `TemporalODEEngine`: 불규칙 내원 간격(Delta t)을 연속 시간 상미분방정식
   (dz/dt = f(z, t))으로 적분하는 Neural ODE 엔진 (`torchdiffeq` 및 내장 RK4/Dopri5 솔버 호환)
4. `AgeRegressorHead`: 잠재 궤적(z(t))으로부터 생물학적 안저 나이(Retinal Age), 노화 속도(Aging Velocity),
   노화 등급(5-Class Ordinal Grade)을 추정하는 회귀 헤드
5. `NeuroSymbolicRetinalAgingModel`: 공간 백본 + 잠재 투영 + Neural ODE + 생리학적 단조 증가 제약
   (Neuro-Symbolic Monotonicity Guardrail)을 통합한 엔드투엔드 모델
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    torchdiffeq_odeint = None
    TORCHDIFFEQ_AVAILABLE = False

try:
    import torchvision.models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:
    tv_models = None
    TORCHVISION_AVAILABLE = False


def _rk4_odeint(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t: torch.Tensor,
    step_size: float = 0.25,
) -> torch.Tensor:
    """
    내장 미분가능 4차 룬게-쿠타(Runge-Kutta 4, RK4) 상미분방정식 솔버.
    `torchdiffeq` 미설치 환경에서도 동일한 인터페이스(`func(t, y)`)로 동작합니다.
    """
    ys = [y0]
    y_curr = y0
    for i in range(len(t) - 1):
        t0 = t[i]
        t1 = t[i + 1]
        dt_total = float((t1 - t0).item())
        if abs(dt_total) < 1e-7:
            ys.append(y_curr)
            continue

        n_steps = max(1, int(abs(dt_total) / max(step_size, 0.05)) + 1)
        h = (t1 - t0) / float(n_steps)
        t_step = t0
        for _ in range(n_steps):
            k1 = func(t_step, y_curr)
            k2 = func(t_step + 0.5 * h, y_curr + 0.5 * h * k1)
            k3 = func(t_step + 0.5 * h, y_curr + 0.5 * h * k2)
            k4 = func(t_step + h, y_curr + h * k3)
            y_curr = y_curr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t_step = t_step + h
        ys.append(y_curr)

    return torch.stack(ys, dim=0)  # (T, B, D)


class SpatialBackbone(nn.Module):
    """
    Spatial Feature Extractor (`EfficientNet-B4` Feature Extractor).
    고해상도 안저 이미지에서 혈관 리모델링(Vascular Remodeling) 및 미세 병변의 공간적 특징을 추출합니다.
    오프라인/샌드박스 환경을 고려하여 사전학습 가중치 미존재 시에도 즉시 초기화되도록 설계되었습니다.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = False,
        out_channels: int = 1792,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.out_channels = out_channels

        # 경량 다중 스케일 혈관/색조 바이오마커 보조 브랜치 (빠른 수렴 및 공간 통계 보존)
        self.biomarker_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.use_tv_effnet = False
        if TORCHVISION_AVAILABLE and backbone_name == "efficientnet_b4":
            try:
                weights = "DEFAULT" if pretrained else None
                effnet = tv_models.efficientnet_b4(weights=weights)
                self.features = effnet.features
                self.pool = effnet.avgpool
                self.out_channels = 1792
                self.use_tv_effnet = True
            except Exception:
                self.use_tv_effnet = False

        if not self.use_tv_effnet:
            # EfficientNet-B4 MBConv 스타일 계층형 특징 추출기 (1792-d 출력 호환)
            self.features = nn.Sequential(
                nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.SiLU(),
                nn.Conv2d(48, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, 384, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(384),
                nn.SiLU(),
                nn.Conv2d(384, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.SiLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 전역 색조/명암비 요약 통계(채널별 평균 및 표준편차 6차원) + 바이오마커(64차원) 결합 게이트
        self.stat_proj = nn.Linear(64 + 6, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, H, W) 안저 텐서
        Returns:
            feat: (N, 1792) 공간 특징 벡터
        """
        deep_feat = self.pool(self.features(x)).flatten(1)
        bio_feat = self.biomarker_branch(x)
        ch_mean = x.mean(dim=(2, 3))
        ch_std = x.std(dim=(2, 3))
        stat_feat = self.stat_proj(torch.cat([bio_feat, ch_mean, ch_std], dim=-1))
        return deep_feat * 0.25 + stat_feat


class LatentSpaceProjector(nn.Module):
    """
    Latent Space Projector (`Linear & LayerNorm, d=64`).
    고차원 공간 특징(1792-d)을 연속 시간 미분방정식 적분에 적합한 64차원 잠재 다양체(Latent Manifold)로 투영합니다.
    """

    def __init__(self, in_features: int = 1792, latent_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.projector = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        return self.projector(spatial_features)


class RetinalODEFunc(nn.Module):
    """
    Neural ODE 벡터장 함수 (`dz/dt = f_theta(z, t)`).
    안저 미세혈관 노화 및 신경망막층 퇴행의 연속 시간 동역학을 모델링합니다.
    Lipschitz 안정성을 위해 LayerNorm과 Tanh/SiLU 게이팅을 결합합니다.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of Function Evaluations 추적

        # 시간 t를 주파수 임베딩으로 변환하여 비자율(Non-autonomous) 역학 표현
        self.time_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, latent_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),  # dz/dt 폭주 방지 (Bounded Vector Field)
        )
        self.scale = nn.Parameter(torch.tensor(0.35))

    def reset_nfe(self) -> None:
        self.nfe = 0

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 스칼라 텐서 (현재 시간, 단위: 년)
            z: (B, latent_dim) 잠재 상태 벡터
        Returns:
            dz_dt: (B, latent_dim) 시간 미분값
        """
        self.nfe += 1
        t_vec = t.view(1, 1).expand(z.size(0), 1).to(z.dtype)
        t_emb = self.time_net(t_vec)
        dz_dt = self.net(torch.cat([z, t_emb], dim=-1)) * F.softplus(self.scale)
        return dz_dt


class TemporalODEEngine(nn.Module):
    """
    Temporal ODE Engine (`RetinalODEFunc` + `torchdiffeq`).
    환자마다 불규칙한 내원 간격(Delta t)을 고정 크기 RNN이 아닌 연속 미분방정식으로 적분하며,
    관측 시점마다 공간 특징 관측치(z_obs)와 ODE 예측 상태(z_ode)를 융합합니다.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        solver_method: str = "rk4",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        step_size: float = 0.25,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.ode_func = RetinalODEFunc(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.solver_method = solver_method
        self.rtol = rtol
        self.atol = atol
        self.step_size = step_size

        # 관측 시점 t_k에서 ODE 예측 궤적과 실제 관측 잠재 벡터를 결합하는 GRU-ODE 업데이트 셀
        self.update_gate = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def integrate_interval(self, z0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """단일 시간 구간 [t_a, t_b]에 대해 dz/dt = f(z, t)를 적분."""
        if TORCHDIFFEQ_AVAILABLE and torchdiffeq_odeint is not None:
            options = {"step_size": self.step_size} if self.solver_method in ("rk4", "euler") else {}
            traj = torchdiffeq_odeint(
                self.ode_func,
                z0,
                t_span,
                method=self.solver_method,
                rtol=self.rtol,
                atol=self.atol,
                options=options,
            )
            return traj[-1]
        return _rk4_odeint(self.ode_func, z0, t_span, step_size=self.step_size)[-1]

    def forward(
        self, z_obs_seq: torch.Tensor, timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_obs_seq: (B, T, D) 각 방문 시점 안저 영상의 잠재 특징 시퀀스
            timestamps: (B, T) 환자별 불규칙 내원 시점 (단위: 년)
        Returns:
            z_traj: (B, T, D) 연속 시간 적분 및 관측 보정된 잠재 궤적
            dz_dt_seq: (B, T, D) 각 시점에서의 순간 변화율 벡터 (dz/dt)
        """
        self.ode_func.reset_nfe()
        bsz, seq_len, _ = z_obs_seq.shape
        z_curr = z_obs_seq[:, 0, :]

        traj_list = [z_curr]
        dz_list = [self.ode_func(timestamps[:, 0].mean(), z_curr)]

        for k in range(1, seq_len):
            # 배치 내 환자별 시간 간격 dt_k = t_k - t_{k-1}
            dt_batch = (timestamps[:, k] - timestamps[:, k - 1]).clamp(min=1e-3)
            mean_dt = float(dt_batch.mean().item())
            t_span = torch.tensor([0.0, mean_dt], device=z_obs_seq.device, dtype=z_obs_seq.dtype)

            # 1) ODE 연속 적분 예측
            z_pred = self.integrate_interval(z_curr, t_span)
            # 환자별 개별 dt 비율 미세 보정
            dz_k = self.ode_func(timestamps[:, k].mean(), z_pred)
            dt_ratio = (dt_batch / max(mean_dt, 1e-4)).unsqueeze(-1)
            z_ode = z_curr + (z_pred - z_curr) * dt_ratio

            # 2) 관측된 안저 특징(z_obs_seq[:, k])과 ODE 예측 상태 융합
            z_curr = self.norm(self.update_gate(z_obs_seq[:, k, :], z_ode))
            traj_list.append(z_curr)
            dz_list.append(dz_k)

        z_traj = torch.stack(traj_list, dim=1)  # (B, T, D)
        dz_dt_seq = torch.stack(dz_list, dim=1)  # (B, T, D)
        return z_traj, dz_dt_seq


class AgeRegressorHead(nn.Module):
    """
    Age Regressor Head (생물학적 나이 및 노화 속도 예측).
    잠재 공간(Latent Space, d=64)의 궤적 변화를 입력받아:
    1) 연속적인 생물학적 안저 나이(Retinal Age)
    2) 순간 노화 속도(Aging Velocity, 세/년)
    3) 5단계 노화 등급(Ordinal Aging Stage 0~4, QWK 평가용)
    을 산출합니다.
    """

    def __init__(self, latent_dim: int = 64, num_grades: int = 5) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_grades = num_grades

        # 연속 생물학적 나이 회귀기
        self.age_mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # 순간 노화 속도(dAge/dt) 추정기 (양수 보장 Softplus)
        self.velocity_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        # 5단계 노화 등급 분류/순서형 헤드 (예측 연령과 잠재 특징을 결합하여 일관된 등급 산출)
        self.grade_head = nn.Linear(latent_dim + 1, num_grades)
        # 기준 연령 임계값 버퍼 (42, 52, 62, 72세)
        self.register_buffer("grade_centers", torch.tensor([36.0, 47.0, 57.0, 67.0, 78.0]))

    def forward(
        self, z_traj: torch.Tensor, dz_dt_seq: torch.Tensor, timestamps: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_traj: (B, T, D)
            dz_dt_seq: (B, T, D)
            timestamps: (B, T)
        """
        # 기준 연령 오프셋(55세 중심) + 잠재 회귀 출력
        raw_age_offset = self.age_mlp(z_traj).squeeze(-1) * 18.0 + 55.0  # (B, T)
        vel = self.velocity_mlp(torch.cat([z_traj, dz_dt_seq], dim=-1)).squeeze(-1) + 0.5  # (B, T)

        # 초기 시점 연령 + 시간 적분 노화량을 결합한 궤적 연령
        dt_from_start = timestamps - timestamps[:, :1]
        pred_ages = 0.65 * raw_age_offset + 0.35 * (raw_age_offset[:, :1] + vel * dt_from_start)
        pred_final_age = pred_ages[:, -1]  # (B,)

        # 5단계 노화 등급 로짓: 예측 나이 기반 거리 로짓 + 잠재 공간 보정 로짓
        z_last = z_traj[:, -1, :]
        dist_logits = -0.18 * (pred_final_age.unsqueeze(-1) - self.grade_centers.unsqueeze(0)) ** 2
        res_logits = self.grade_head(torch.cat([z_last, pred_final_age.unsqueeze(-1) / 100.0], dim=-1))
        grade_logits = dist_logits + 0.25 * res_logits

        # Neuro-Symbolic 생리학적 단조 증가 제약 (시간 경과에 따른 생물학적 역노화 페널티)
        age_diffs = pred_ages[:, 1:] - pred_ages[:, :-1]
        monotonicity_violation = F.relu(-age_diffs + 0.1).mean()

        return {
            "pred_ages": pred_ages,  # (B, T)
            "pred_final_age": pred_final_age,  # (B,)
            "aging_velocity": vel[:, -1],  # (B,)
            "grade_logits": grade_logits,  # (B, 5)
            "monotonicity_loss": monotonicity_violation,
        }


class NeuroSymbolicRetinalAgingModel(nn.Module):
    """
    System 2 & 3 통합 모델 (`NeuroSymbolicRetinalAgingModel`).
    - Spatial Backbone (`EfficientNet-B4`)
    - Latent Space Projector (`Linear & LayerNorm, d=64`)
    - Temporal ODE Engine (`RetinalODEFunc + torchdiffeq`)
    - Age Regressor Head (`AgeRegressorHead`)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        ode_hidden_dim: int = 128,
        solver_method: str = "rk4",
        pretrained_backbone: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_backbone = SpatialBackbone(
            backbone_name="efficientnet_b4",
            pretrained=pretrained_backbone,
        )
        self.latent_projector = LatentSpaceProjector(
            in_features=self.spatial_backbone.out_channels,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.temporal_ode = TemporalODEEngine(
            latent_dim=latent_dim,
            hidden_dim=ode_hidden_dim,
            solver_method=solver_method,
        )
        self.regressor_head = AgeRegressorHead(latent_dim=latent_dim, num_grades=5)

    def forward(self, images_seq: torch.Tensor, timestamps: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            images_seq: (B, T, C, H, W) 종단 전처리 안저 이미지 시퀀스
            timestamps: (B, T) 불규칙 방문 시점 텐서 (단위: 년)
        Returns:
            Dict containing predicted biological ages, latent trajectories, aging velocity,
            ordinal grade logits, and neuro-symbolic monotonicity loss.
        """
        bsz, seq_len, c, h, w = images_seq.shape
        flat_imgs = images_seq.reshape(bsz * seq_len, c, h, w)

        # 1. Spatial Backbone (EfficientNet-B4 Feature Extraction)
        spatial_feats = self.spatial_backbone(flat_imgs)  # (B*T, 1792)

        # 2. Latent Space Projector (Linear & LayerNorm, d=64)
        z_obs = self.latent_projector(spatial_feats).reshape(bsz, seq_len, self.latent_dim)

        # 3. Temporal ODE Engine (Continuous-time trajectory integration)
        z_traj, dz_dt_seq = self.temporal_ode(z_obs, timestamps)

        # 4. Age Regressor Head
        outputs = self.regressor_head(z_traj, dz_dt_seq, timestamps)
        outputs["z_obs"] = z_obs
        outputs["z_traj"] = z_traj
        outputs["nfe"] = self.temporal_ode.ode_func.nfe
        return outputs

