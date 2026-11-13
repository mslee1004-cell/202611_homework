"""
System 3: 예측 및 폐루프 제어 계층 (Control & Regression Layer)
====================================================================
구성요소:
- `compute_quadratic_weighted_kappa`: 5단계 생물학적 노화 등급에 대한 Quadratic Weighted Kappa (QWK) 산출
- `AutonomousAgentController`: NS-Copilot 스타일의 자율 폐루프 자가 수정 제어기 (Closed-Loop Controller)
  - 모델의 수행 결과나 성능 지표가 목표 임계값(예: QWK >= 0.85)에 미치지 못할 경우,
    원인을 스스로 진단하여 **MODIFY(국소 패치)** 또는 **REPLAN(전략 전면 재수립)** 경로로 자율 분기합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class ControlAction(str, Enum):
    """폐루프 제어기의 자율 분기 결정 상태."""
    ACCEPT = "ACCEPT"   # 목표 임계값(QWK >= 0.85) 달성 -> 학습 종료 및 모델 확정
    MODIFY = "MODIFY"   # 국소 코드/하이퍼파라미터 패치 (손실 가중치, 학습률, ODE 스텝 정밀도 조정)
    REPLAN = "REPLAN"   # 전략 전면 재수립 (솔버 교체, 옵티마이저 재구축, 백본 학습률 스케줄 재설계)


@dataclass
class ControllerConfig:
    """폐루프 제어기 임계값 및 하이퍼파라미터 설정."""
    target_qwk: float = 0.85
    modify_floor_qwk: float = 0.65
    target_mae: float = 3.50
    max_monotonicity_violation: float = 0.05
    max_closed_loop_rounds: int = 4


@dataclass
class DiagnosticReport:
    """매 라운드 검증 후 제어기가 생성하는 진단 및 처방 리포트."""
    round_idx: int
    qwk: float
    mae: float
    rmse: float
    r2: float
    monotonicity_violation_rate: float
    action: ControlAction
    root_cause: str
    prescription: Dict[str, Any] = field(default_factory=dict)


def compute_quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5
) -> float:
    """
    Quadratic Weighted Kappa (QWK) 계산 함수.
    안저 질환 및 생물학적 노화 등급 평가의 국제 표준 지표입니다.
    """
    y_true_int = np.clip(np.asarray(y_true, dtype=np.int64), 0, num_classes - 1)
    y_pred_int = np.clip(np.asarray(y_pred, dtype=np.int64), 0, num_classes - 1)

    if len(y_true_int) == 0:
        return 0.0

    # 관측 혼동 행렬 O (Observed Matrix)
    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true_int, y_pred_int):
        O[t, p] += 1.0

    # 가중치 행렬 W (Quadratic Penalty)
    i_idx, j_idx = np.meshgrid(np.arange(num_classes), np.arange(num_classes), indexing="ij")
    W = ((i_idx - j_idx) ** 2) / float(max(1, (num_classes - 1) ** 2))

    # 기댓값 행렬 E (Expected Matrix under independence)
    hist_true = np.bincount(y_true_int, minlength=num_classes).astype(np.float64)
    hist_pred = np.bincount(y_pred_int, minlength=num_classes).astype(np.float64)
    total = float(len(y_true_int))
    E = np.outer(hist_true, hist_pred) / max(total, 1.0)

    denom = float(np.sum(W * E))
    if denom < 1e-12:
        return 1.0 if np.array_equal(y_true_int, y_pred_int) else 0.0

    num = float(np.sum(W * O))
    return float(1.0 - (num / denom))


def age_to_ordinal_grade(ages: np.ndarray) -> np.ndarray:
    """연속 생물학적 나이를 5단계 임상 노화 등급(0~4)으로 변환."""
    return np.digitize(ages, bins=[42.0, 52.0, 62.0, 72.0]).astype(np.int64)


class AutonomousAgentController:
    """
    NS-Copilot 스타일 자율 폐루프 제어기 (`AutonomousAgentController`).

    동작 원리:
    1. 검증 데이터셋에서 산출된 QWK, MAE, R², 단조성 위반율(Monotonicity Violation)을 감사(Audit)합니다.
    2. 목표 임계값(`QWK >= 0.85`) 달성 여부를 판정합니다.
    3. 미달 시 오차 원인을 분석하여:
       - `MODIFY` (국소 패치): 등급 경계 보정 가중치 상향, ODE 적분 스텝 세분화, 학습률 미세 조정 수행
       - `REPLAN` (전면 재수립): 심각한 과소적합/발산 시 솔버/옵티마이저 전략 재구성 및 집중 학습 수행
    """

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()
        self.history: List[DiagnosticReport] = []

        # 현재 활성 하이퍼파라미터 상태 (제어기가 동적으로 수정)
        self.state: Dict[str, Any] = {
            "lr": 2.5e-3,
            "weight_decay": 1e-4,
            "age_loss_weight": 1.0,
            "ordinal_loss_weight": 0.6,
            "monotonicity_weight": 2.0,
            "ode_step_size": 0.25,
            "ode_solver": "rk4",
            "sigma_scale": 30.0,
            "epochs_per_round": 6,
        }

    def evaluate_and_decide(
        self,
        round_idx: int,
        true_ages: np.ndarray,
        pred_ages: np.ndarray,
        pred_grades: np.ndarray,
        monotonicity_violation_rate: float = 0.0,
    ) -> DiagnosticReport:
        """
        검증 결과를 바탕으로 성능을 평가하고 ACCEPT / MODIFY / REPLAN 경로를 자율 결정합니다.
        """
        true_grades = age_to_ordinal_grade(true_ages)
        grade_from_age = age_to_ordinal_grade(pred_ages)
        # 회귀 기반 등급과 분류 헤드 등급의 앙상블 판정
        blended_grades = np.rint(0.7 * grade_from_age + 0.3 * pred_grades).astype(np.int64)

        qwk = compute_quadratic_weighted_kappa(true_grades, blended_grades, num_classes=5)
        errors = np.abs(true_ages - pred_ages)
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean((true_ages - pred_ages) ** 2)))

        ss_res = float(np.sum((true_ages - pred_ages) ** 2))
        ss_tot = float(np.sum((true_ages - np.mean(true_ages)) ** 2))
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))

        # 1. 목표 임계값 달성 확인 (QWK >= target_qwk)
        if qwk >= self.config.target_qwk and monotonicity_violation_rate <= self.config.max_monotonicity_violation:
            report = DiagnosticReport(
                round_idx=round_idx,
                qwk=qwk,
                mae=mae,
                rmse=rmse,
                r2=r2,
                monotonicity_violation_rate=monotonicity_violation_rate,
                action=ControlAction.ACCEPT,
                root_cause=(
                    f"Target achieved: QWK={qwk:.4f} >= {self.config.target_qwk:.2f}, "
                    f"MAE={mae:.2f} yrs, Monotonicity violation={monotonicity_violation_rate:.4f}."
                ),
                prescription={"status": "LOCKED_OPTIMAL_CHECKPOINT"},
            )
            self.history.append(report)
            return report

        # 2. 미달 시 원인 진단: REPLAN vs MODIFY 분기
        prev_qwk = self.history[-1].qwk if self.history else -1.0
        stagnated = (len(self.history) >= 1) and ((qwk - prev_qwk) < 0.02) and (qwk < self.config.modify_floor_qwk)

        if qwk < self.config.modify_floor_qwk or stagnated:
            # REPLAN: 전략 전면 재수립
            self.state["lr"] = max(8e-4, self.state["lr"] * 0.85)
            self.state["ode_step_size"] = 0.125
            self.state["ordinal_loss_weight"] = min(2.0, self.state["ordinal_loss_weight"] * 1.5)
            self.state["monotonicity_weight"] = min(5.0, self.state["monotonicity_weight"] * 1.4)
            self.state["epochs_per_round"] = min(12, self.state["epochs_per_round"] + 3)
            self.state["reinit_scheduler"] = True

            report = DiagnosticReport(
                round_idx=round_idx,
                qwk=qwk,
                mae=mae,
                rmse=rmse,
                r2=r2,
                monotonicity_violation_rate=monotonicity_violation_rate,
                action=ControlAction.REPLAN,
                root_cause=(
                    f"QWK ({qwk:.4f}) below modify floor ({self.config.modify_floor_qwk:.2f}) "
                    f"or trajectory stagnation detected (MAE={mae:.2f} yrs). Replanning optimization & ODE resolution."
                ),
                prescription=dict(self.state),
            )
        else:
            # MODIFY: 국소 파라미터 패치
            self.state["lr"] = max(5e-4, self.state["lr"] * 0.75)
            self.state["ode_step_size"] = max(0.10, self.state["ode_step_size"] * 0.75)
            self.state["ordinal_loss_weight"] = min(1.8, self.state["ordinal_loss_weight"] + 0.25)
            if monotonicity_violation_rate > self.config.max_monotonicity_violation:
                self.state["monotonicity_weight"] *= 1.5
            self.state["reinit_scheduler"] = False

            report = DiagnosticReport(
                round_idx=round_idx,
                qwk=qwk,
                mae=mae,
                rmse=rmse,
                r2=r2,
                monotonicity_violation_rate=monotonicity_violation_rate,
                action=ControlAction.MODIFY,
                root_cause=(
                    f"QWK ({qwk:.4f}) approaching target ({self.config.target_qwk:.2f}) "
                    f"but needs fine boundary calibration (MAE={mae:.2f} yrs). Applying local patch."
                ),
                prescription=dict(self.state),
            )

        self.history.append(report)
        return report

    def apply_prescription(
        self,
        report: DiagnosticReport,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        preprocess_transform: Optional[Any] = None,
    ) -> None:
        """
        진단 리포트의 처방(MODIFY 또는 REPLAN)을 실제 모델, 옵티마이저, 전처리기에 반영합니다.
        """
        if report.action == ControlAction.ACCEPT:
            return

        rx = report.prescription
        # 1. 옵티마이저 학습률 패치
        for param_group in optimizer.param_groups:
            param_group["lr"] = rx.get("lr", param_group["lr"])

        # 2. Temporal ODE Engine 적분 정밀도 패치
        if hasattr(model, "temporal_ode"):
            model.temporal_ode.step_size = float(rx.get("ode_step_size", model.temporal_ode.step_size))

        # 3. 전처리기(System 1) 국소 명암비 파라미터 패치
        if preprocess_transform is not None and hasattr(preprocess_transform, "update_hyperparameters"):
            preprocess_transform.update_hyperparameters(sigma_scale=rx.get("sigma_scale", 30.0))

