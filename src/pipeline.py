"""
End-to-End Pipeline: 안저 데이터 기반 노화 추정 및 폐루프 제어 실행 엔진
=======================================================================
System 1 (`BenGrahamFundusPreprocessd`) -> System 2 (`NeuroSymbolicRetinalAgingModel`)
-> System 3 (`AgeRegressorHead` + `AutonomousAgentController`)의 전 과정을 통합 실행하고
시각화 아티팩트 및 정량 평가 리포트를 생성합니다.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .controller import (
    AutonomousAgentController,
    ControlAction,
    ControllerConfig,
    DiagnosticReport,
    age_to_ordinal_grade,
    compute_quadratic_weighted_kappa,
)
from .models import NeuroSymbolicRetinalAgingModel
from .preprocessing import BenGrahamFundusPreprocessd, LongitudinalFundusDataset


def run_system1_quality_and_visuals(
    image_dir: Path, output_dir: Path, img_size: int = 256, num_samples: int = 3
) -> Dict[str, Any]:
    """
    System 1 (`BenGrahamFundusPreprocessd`) 4단계 전처리 검증 및 시각화 이미지 생성.
    실제 `./train` 안저 이미지를 로드하여 검은 여백 비율(25~53%), 선명도, 각 단계 변환 결과를 기록합니다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = BenGrahamFundusPreprocessd(
        keys=["image"],
        img_size=img_size,
        return_stages=True,
    )

    sample_paths: List[Path] = []
    if image_dir.exists():
        with os.scandir(image_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith((".jpeg", ".jpg", ".png")):
                    sample_paths.append(Path(entry.path))
                    if len(sample_paths) >= num_samples:
                        break

    if not sample_paths:
        raise FileNotFoundError(f"No fundus images found in {image_dir}")

    metrics_list: List[Dict[str, Any]] = []
    fig, axes = plt.subplots(len(sample_paths), 5, figsize=(16, 3.3 * len(sample_paths)))
    if len(sample_paths) == 1:
        axes = np.expand_dims(axes, axis=0)

    stage_titles = [
        "0. Raw Fundus\n(Multi-center Input)",
        "1. Dynamic Black Margin Crop\n(25~53% Margin Removed)",
        "2. Lanczos4 Resize\n(Square Standardized)",
        "3. Local Contrast Norm\n(Gaussian Blur Subtracted)",
        "4. Circular Masking\n(Edge Halo Removed)",
    ]

    for row_idx, img_path in enumerate(sample_paths):
        out = preprocessor({"image": img_path})
        meta = out["image_meta"]
        stages = out["image_stages"]

        metrics_list.append(
            {
                "filename": img_path.name,
                "black_margin_ratio_pct": round(meta["black_margin_ratio"] * 100.0, 2),
                "sharpness_score": round(meta["sharpness_score"], 2),
                "is_valid": meta["is_valid"],
            }
        )

        stage_keys = ["0_raw", "1_cropped", "2_lanczos4", "3_local_contrast", "4_circular_masked"]
        for col_idx, sk in enumerate(stage_keys):
            ax = axes[row_idx, col_idx]
            ax.imshow(stages[sk])
            if row_idx == 0:
                ax.set_title(stage_titles[col_idx], fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{img_path.name}\nMargin: {meta['black_margin_ratio']*100:.1f}%",
                    fontsize=9,
                    fontweight="bold",
                )
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(
        "System 1: MONAI BenGrahamFundusPreprocessd 4-Stage Pipeline Verification",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    vis_path = output_dir / "system1_preprocessing_stages.png"
    fig.savefig(vis_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    avg_margin = float(np.mean([m["black_margin_ratio_pct"] for m in metrics_list]))
    return {
        "samples_analyzed": len(metrics_list),
        "average_black_margin_removed_pct": round(avg_margin, 2),
        "sample_metrics": metrics_list,
        "visualization_path": str(vis_path),
    }


def evaluate_model(
    model: NeuroSymbolicRetinalAgingModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """검증 또는 테스트 데이터셋에서 모델의 정량 지표 및 궤적 예측치를 추출합니다."""
    model.eval()
    all_true_ages: List[np.ndarray] = []
    all_pred_ages: List[np.ndarray] = []
    all_pred_grades: List[np.ndarray] = []
    all_chron_ages: List[np.ndarray] = []
    all_traj_true: List[np.ndarray] = []
    all_traj_pred: List[np.ndarray] = []
    all_timestamps: List[np.ndarray] = []
    mono_violations: List[float] = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            ts = batch["timestamps"].to(device)
            ages_seq = batch["ages"].to(device)
            target_age = batch["target_age"].to(device)
            chron_age = batch["chronological_age"].to(device)

            out = model(imgs, ts)
            pred_seq = out["pred_ages"]  # (B, T)
            pred_final = out["pred_final_age"]  # (B,)
            grade_preds = out["grade_logits"].argmax(dim=-1)  # (B,)

            # 시간 경과에 따른 단조 증가 위반 비율 계산
            diffs = pred_seq[:, 1:] - pred_seq[:, :-1]
            viol_rate = float((diffs < -0.05).float().mean().item())
            mono_violations.append(viol_rate)

            all_true_ages.append(target_age.cpu().numpy())
            all_pred_ages.append(pred_final.cpu().numpy())
            all_pred_grades.append(grade_preds.cpu().numpy())
            all_chron_ages.append(chron_age.cpu().numpy())
            all_traj_true.append(ages_seq.cpu().numpy())
            all_traj_pred.append(pred_seq.cpu().numpy())
            all_timestamps.append(ts.cpu().numpy())

    true_ages_np = np.concatenate(all_true_ages, axis=0)
    pred_ages_np = np.concatenate(all_pred_ages, axis=0)
    pred_grades_np = np.concatenate(all_pred_grades, axis=0)
    chron_ages_np = np.concatenate(all_chron_ages, axis=0)

    return {
        "true_ages": true_ages_np,
        "pred_ages": pred_ages_np,
        "pred_grades": pred_grades_np,
        "chron_ages": chron_ages_np,
        "traj_true": np.concatenate(all_traj_true, axis=0),
        "traj_pred": np.concatenate(all_traj_pred, axis=0),
        "timestamps": np.concatenate(all_timestamps, axis=0),
        "monotonicity_violation_rate": float(np.mean(mono_violations)) if mono_violations else 0.0,
    }


def train_and_run_closed_loop(
    image_dir: Path,
    output_dir: Path,
    max_patients: int = 120,
    img_size: int = 96,
    batch_size: int = 16,
    target_qwk: float = 0.85,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    System 2 (Neural ODE) 학습 및 System 3 (`AutonomousAgentController`) 폐루프 자가 보정 파이프라인 실행.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 환자 단위 분할 데이터셋 로드 (Group Leakage 원천 차단)
    shared_transform = BenGrahamFundusPreprocessd(keys=["images"], img_size=img_size)
    train_ds = LongitudinalFundusDataset(
        image_dir=image_dir,
        split="train",
        max_patients=max_patients,
        seq_len=3,
        img_size=img_size,
        seed=seed,
        preprocess_transform=shared_transform,
    )
    val_ds = LongitudinalFundusDataset(
        image_dir=image_dir,
        split="val",
        max_patients=max_patients,
        seq_len=3,
        img_size=img_size,
        seed=seed,
        preprocess_transform=shared_transform,
    )
    test_ds = LongitudinalFundusDataset(
        image_dir=image_dir,
        split="test",
        max_patients=max_patients,
        seq_len=3,
        img_size=img_size,
        seed=seed,
        preprocess_transform=shared_transform,
    )

    # 환자 ID 중복 누수 검증 (Strict Featurization & Split Integrity)
    train_pids = {r["patient_id"] for r in train_ds.records}
    val_pids = {r["patient_id"] for r in val_ds.records}
    test_pids = {r["patient_id"] for r in test_ds.records}
    assert len(train_pids & val_pids) == 0, "Patient group leakage detected between train and val!"
    assert len(train_pids & test_pids) == 0, "Patient group leakage detected between train and test!"

    # 캐싱을 통한 빠른 반복 에포크 수행
    train_cache = [train_ds[i] for i in range(len(train_ds))]
    val_cache = [val_ds[i] for i in range(len(val_ds))]
    test_cache = [test_ds[i] for i in range(len(test_ds))]

    train_loader = DataLoader(train_cache, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_cache, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_cache, batch_size=batch_size, shuffle=False)

    # 2. 모델 및 폐루프 제어기 초기화
    model = NeuroSymbolicRetinalAgingModel(
        latent_dim=64,
        ode_hidden_dim=128,
        solver_method="rk4",
        pretrained_backbone=False,
    ).to(device)

    ctrl_cfg = ControllerConfig(
        target_qwk=target_qwk,
        modify_floor_qwk=0.65,
        target_mae=3.50,
        max_closed_loop_rounds=4,
    )
    controller = AutonomousAgentController(config=ctrl_cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=controller.state["lr"],
        weight_decay=controller.state["weight_decay"],
    )

    best_state_dict = copy.deepcopy(model.state_dict())
    best_qwk = -1.0

    # 3. 자율 폐루프 제어 루프 (Closed-Loop Execution)
    for round_idx in range(1, ctrl_cfg.max_closed_loop_rounds + 1):
        epochs = int(controller.state["epochs_per_round"])
        age_w = float(controller.state["age_loss_weight"])
        ord_w = float(controller.state["ordinal_loss_weight"])
        mono_w = float(controller.state["monotonicity_weight"])

        model.train()
        for _ in range(epochs):
            for batch in train_loader:
                imgs = batch["images"].to(device)
                ts = batch["timestamps"].to(device)
                ages_seq = batch["ages"].to(device)
                target_age = batch["target_age"].to(device)
                age_grade = batch["age_grade"].to(device)

                optimizer.zero_grad()
                out = model(imgs, ts)

                # 종단 전체 시퀀스 Smooth L1 손실 + 최종 시점 연령 손실
                loss_seq_age = F.smooth_l1_loss(out["pred_ages"], ages_seq, beta=1.5)
                loss_final_age = F.smooth_l1_loss(out["pred_final_age"], target_age, beta=1.5)
                loss_grade = F.cross_entropy(out["grade_logits"], age_grade)
                loss_mono = out["monotonicity_loss"]

                total_loss = (
                    age_w * (0.5 * loss_seq_age + 0.5 * loss_final_age)
                    + ord_w * loss_grade
                    + mono_w * loss_mono
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

        # 검증 세트 평가 및 폐루프 제어기 진단
        val_eval = evaluate_model(model, val_loader, device)
        report = controller.evaluate_and_decide(
            round_idx=round_idx,
            true_ages=val_eval["true_ages"],
            pred_ages=val_eval["pred_ages"],
            pred_grades=val_eval["pred_grades"],
            monotonicity_violation_rate=val_eval["monotonicity_violation_rate"],
        )

        if report.qwk >= best_qwk:
            best_qwk = report.qwk
            best_state_dict = copy.deepcopy(model.state_dict())

        if report.action == ControlAction.ACCEPT:
            break

        # MODIFY 또는 REPLAN 처방 적용
        controller.apply_prescription(report, model, optimizer, shared_transform)

    # 최적 체크포인트 복원 및 최종 Held-out Test 평가
    model.load_state_dict(best_state_dict)
    test_eval = evaluate_model(model, test_loader, device)
    test_true_grades = age_to_ordinal_grade(test_eval["true_ages"])
    test_pred_grades = np.rint(
        0.7 * age_to_ordinal_grade(test_eval["pred_ages"]) + 0.3 * test_eval["pred_grades"]
    ).astype(np.int64)

    test_qwk = compute_quadratic_weighted_kappa(test_true_grades, test_pred_grades, num_classes=5)
    test_mae = float(np.mean(np.abs(test_eval["true_ages"] - test_eval["pred_ages"])))
    test_rmse = float(np.sqrt(np.mean((test_eval["true_ages"] - test_eval["pred_ages"]) ** 2)))
    ss_res = float(np.sum((test_eval["true_ages"] - test_eval["pred_ages"]) ** 2))
    ss_tot = float(np.sum((test_eval["true_ages"] - np.mean(test_eval["true_ages"])) ** 2))
    test_r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))

    # Baseline 비교 (Naive Mean Baseline vs Neural ODE System)
    train_mean_age = float(np.mean([b["target_age"].item() for b in train_cache]))
    naive_preds = np.full_like(test_eval["true_ages"], train_mean_age)
    naive_mae = float(np.mean(np.abs(test_eval["true_ages"] - naive_preds)))
    naive_qwk = compute_quadratic_weighted_kappa(
        test_true_grades, age_to_ordinal_grade(naive_preds), num_classes=5
    )

    # 모델 체크포인트 저장
    ckpt_path = output_dir / "neuro_symbolic_retinal_ode.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "controller_state": controller.state,
            "test_metrics": {
                "qwk": test_qwk,
                "mae": test_mae,
                "rmse": test_rmse,
                "r2": test_r2,
            },
        },
        ckpt_path,
    )

    # 4. System 2 & System 3 시각화 리포트 생성
    vis_path = _plot_system2_3_results(
        test_eval=test_eval,
        test_true_grades=test_true_grades,
        test_pred_grades=test_pred_grades,
        test_qwk=test_qwk,
        test_mae=test_mae,
        test_r2=test_r2,
        controller_history=controller.history,
        target_qwk=target_qwk,
        output_dir=output_dir,
    )

    return {
        "dataset_splits": {
            "train_patients": len(train_ds),
            "val_patients": len(val_ds),
            "test_patients": len(test_ds),
            "patient_leakage_overlap": 0,
        },
        "closed_loop_history": [
            {
                "round": r.round_idx,
                "action": r.action.value,
                "val_qwk": round(r.qwk, 4),
                "val_mae": round(r.mae, 2),
                "val_rmse": round(r.rmse, 2),
                "val_r2": round(r.r2, 4),
                "monotonicity_violation_rate": round(r.monotonicity_violation_rate, 4),
                "root_cause": r.root_cause,
            }
            for r in controller.history
        ],
        "test_performance": {
            "neural_ode_qwk": round(test_qwk, 4),
            "neural_ode_mae_years": round(test_mae, 2),
            "neural_ode_rmse_years": round(test_rmse, 2),
            "neural_ode_r2": round(test_r2, 4),
            "monotonicity_violation_rate": round(test_eval["monotonicity_violation_rate"], 4),
            "naive_baseline_mae_years": round(naive_mae, 2),
            "naive_baseline_qwk": round(naive_qwk, 4),
        },
        "checkpoint_path": str(ckpt_path),
        "visualization_path": str(vis_path),
    }


def _plot_system2_3_results(
    test_eval: Dict[str, Any],
    test_true_grades: np.ndarray,
    test_pred_grades: np.ndarray,
    test_qwk: float,
    test_mae: float,
    test_r2: float,
    controller_history: List[DiagnosticReport],
    target_qwk: float,
    output_dir: Path,
) -> Path:
    """System 2(Neural ODE 궤적) 및 System 3(회귀 & 폐루프 제어) 4-패널 분석 대시보드 생성."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Neural ODE Longitudinal Trajectories
    ax_traj = axes[0, 0]
    n_plot = min(6, len(test_eval["timestamps"]))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_plot))
    for i in range(n_plot):
        t_seq = test_eval["timestamps"][i]
        y_true_seq = test_eval["traj_true"][i]
        y_pred_seq = test_eval["traj_pred"][i]
        ax_traj.plot(t_seq, y_true_seq, "o--", color=colors[i], alpha=0.65, label="True" if i == 0 else "")
        ax_traj.plot(t_seq, y_pred_seq, "s-", color=colors[i], linewidth=2.0, label="ODE Pred" if i == 0 else "")
    ax_traj.set_title("A. System 2: Neural ODE Longitudinal Aging Trajectories (dz/dt)", fontweight="bold")
    ax_traj.set_xlabel("Follow-up Time from Baseline Visit (Years, Irregular Δt)")
    ax_traj.set_ylabel("Biological Retinal Age (Years)")
    ax_traj.grid(True, linestyle=":", alpha=0.6)
    ax_traj.legend(loc="upper left")

    # Panel B: Predicted vs Ground-Truth Biological Age Scatter
    ax_scat = axes[0, 1]
    t_ages = test_eval["true_ages"]
    p_ages = test_eval["pred_ages"]
    ax_scat.scatter(t_ages, p_ages, c=t_ages, cmap="coolwarm", edgecolors="k", s=65, alpha=0.85)
    min_a, max_a = float(min(t_ages.min(), p_ages.min()) - 2), float(max(t_ages.max(), p_ages.max()) + 2)
    ax_scat.plot([min_a, max_a], [min_a, max_a], "r--", lw=2, label="Ideal Identity (y = x)")
    ax_scat.set_title(
        f"B. System 3: Biological Age Regression (MAE={test_mae:.2f}y, R²={test_r2:.3f})",
        fontweight="bold",
    )
    ax_scat.set_xlabel("Ground-Truth Biological Retinal Age (Years)")
    ax_scat.set_ylabel("Predicted Retinal Age (Years)")
    ax_scat.legend(loc="upper left")
    ax_scat.grid(True, linestyle=":", alpha=0.6)

    # Panel C: 5-Stage Aging Grade Confusion Matrix (QWK)
    ax_cm = axes[1, 0]
    cm = np.zeros((5, 5), dtype=np.int64)
    for yt, yp in zip(test_true_grades, test_pred_grades):
        cm[int(np.clip(yt, 0, 4)), int(np.clip(yp, 0, 4))] += 1
    im = ax_cm.imshow(cm, cmap="Blues")
    for r in range(5):
        for c in range(5):
            ax_cm.text(
                c,
                r,
                str(cm[r, c]),
                ha="center",
                va="center",
                color="white" if cm[r, c] > cm.max() / 2 else "black",
                fontweight="bold",
            )
    grade_labels = ["G0 (<42)", "G1 (42-52)", "G2 (52-62)", "G3 (62-72)", "G4 (72+)"]
    ax_cm.set_xticks(range(5))
    ax_cm.set_yticks(range(5))
    ax_cm.set_xticklabels(grade_labels, rotation=20)
    ax_cm.set_yticklabels(grade_labels)
    ax_cm.set_xlabel("Predicted Aging Stage")
    ax_cm.set_ylabel("True Aging Stage")
    ax_cm.set_title(f"C. Ordinal Aging Stage Confusion Matrix (QWK = {test_qwk:.4f})", fontweight="bold")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # Panel D: Closed-Loop Controller Progression
    ax_ctrl = axes[1, 1]
    rounds = [r.round_idx for r in controller_history]
    qwks = [r.qwk for r in controller_history]
    maes = [r.mae for r in controller_history]
    actions = [r.action.value for r in controller_history]

    ax_ctrl.plot(rounds, qwks, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Validation QWK")
    ax_ctrl.axhline(target_qwk, color="green", linestyle="--", linewidth=1.8, label=f"Target QWK ({target_qwk:.2f})")
    for x_r, y_q, act in zip(rounds, qwks, actions):
        badge_color = "#2ca02c" if act == "ACCEPT" else ("#ff7f0e" if act == "MODIFY" else "#d62728")
        ax_ctrl.annotate(
            f"{act}\n({y_q:.3f})",
            (x_r, y_q),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=badge_color,
        )
    ax_ctrl.set_ylim(max(0.0, min(qwks) - 0.15), 1.05)
    ax_ctrl.set_xticks(rounds)
    ax_ctrl.set_xlabel("Autonomous Closed-Loop Controller Round")
    ax_ctrl.set_ylabel("Quadratic Weighted Kappa (QWK)")
    ax_ctrl.set_title("D. System 3: Autonomous Closed-Loop Self-Correction (Modify vs Replan)", fontweight="bold")
    ax_ctrl.legend(loc="lower right")
    ax_ctrl.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(
        "Neuro-Symbolic Retinal Aging System (System 2 Neural ODE & System 3 Closed-Loop Control)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    vis_path = output_dir / "system2_3_aging_trajectories_and_control.png"
    fig.savefig(vis_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return vis_path

