"""
안저 데이터 기반 노화 추정 및 생물학적 나이 예측 시스템 (메인 엔트리포인트)
==========================================================================
실행 방법:
    .venv\\Scripts\\python.exe main.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.pipeline import run_system1_quality_and_visuals, train_and_run_closed_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retinal Fundus Aging & Biological Age Prediction System (MONAI + Neural ODE + Closed-Loop Controller)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="train",
        help="종단 안저 이미지 디렉터리 경로 (기본값: ./train)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="결과 리포트 및 시각화 저장 경로 (기본값: ./outputs)",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=120,
        help="학습/검증/테스트에 사용할 환자 코호트 크기 (기본값: 120)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=96,
        help="Neural ODE 학습용 리사이즈 해상도 (기본값: 96)",
    )
    parser.add_argument(
        "--target-qwk",
        type=float,
        default=0.85,
        help="폐루프 제어기(AutonomousAgentController) 목표 QWK 임계값 (기본값: 0.85)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    image_dir = (root_dir / args.image_dir).resolve()
    output_dir = (root_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 78)
    print(" [Retinal Aging System] 안저 이미지 기반 노화 추정 및 생물학적 나이 예측 시스템")
    print("=" * 78)

    # 1. System 1 실행: MONAI MapTransform (BenGrahamFundusPreprocessd)
    print("\n[Step 1] System 1: MONAI BenGrahamFundusPreprocessd 4단계 전처리 및 품질 검증 실행 중...")
    sys1_report = run_system1_quality_and_visuals(
        image_dir=image_dir,
        output_dir=output_dir,
        img_size=256,
        num_samples=3,
    )
    print(f"  - 분석 샘플 수: {sys1_report['samples_analyzed']}장")
    print(f"  - 평균 제거된 검은 여백(Black Margin) 비율: {sys1_report['average_black_margin_removed_pct']:.2f}%")
    for m in sys1_report["sample_metrics"]:
        print(
            f"    * {m['filename']}: 여백 제거={m['black_margin_ratio_pct']:.1f}%, "
            f"선명도(Sharpness)={m['sharpness_score']:.1f}, 품질통과={m['is_valid']}"
        )
    print(f"  - System 1 단계별 시각화 저장 완료: {sys1_report['visualization_path']}")

    # 2. System 2 & 3 실행: Spatial-Temporal Neural ODE + Autonomous Closed-Loop Controller
    print("\n[Step 2 & 3] System 2 (Neural ODE) 학습 및 System 3 (Autonomous Controller) 폐루프 제어 실행 중...")
    sys23_report = train_and_run_closed_loop(
        image_dir=image_dir,
        output_dir=output_dir,
        max_patients=args.max_patients,
        img_size=args.img_size,
        target_qwk=args.target_qwk,
    )

    print("\n  [Closed-Loop Controller 자가 보정 이력]")
    for r in sys23_report["closed_loop_history"]:
        print(
            f"    * Round {r['round']} -> [{r['action']}] "
            f"Val QWK={r['val_qwk']:.4f}, MAE={r['val_mae']:.2f}세, R²={r['val_r2']:.4f} | {r['root_cause']}"
        )

    perf = sys23_report["test_performance"]
    print("\n  [Held-out Test 최종 성능 평가 (Baseline 대비)]")
    print(f"    * Naive Mean Baseline : QWK = {perf['naive_baseline_qwk']:.4f}, MAE = {perf['naive_baseline_mae_years']:.2f}세")
    print(f"    * Neural ODE System   : QWK = {perf['neural_ode_qwk']:.4f}, MAE = {perf['neural_ode_mae_years']:.2f}세, RMSE = {perf['neural_ode_rmse_years']:.2f}세, R² = {perf['neural_ode_r2']:.4f}")
    print(f"    * 생리학적 역노화 위반율 (Monotonicity Violation): {perf['monotonicity_violation_rate']*100:.2f}%")

    elapsed = round(time.time() - t0, 2)
    full_summary = {
        "elapsed_seconds": elapsed,
        "system1_preprocessing": sys1_report,
        "system2_and_3_modeling_and_control": sys23_report,
    }
    report_json_path = output_dir / "evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f" 완료! 총 소요 시간: {elapsed:.2f}초")
    print(f" 통합 평가 리포트: {report_json_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()

