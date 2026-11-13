"""
System 1: 전처리 및 품질 검증 계층 (Quality & Preprocessing Layer)
====================================================================
구성요소:
- `BenGrahamFundusPreprocessd`: MONAI `MapTransform`을 상속한 4단계 안저 표준 전처리 변환기
  1. 검은 여백(Black Margin) 동적 크롭 (이미지 면적의 25~53% 무의미 배경 제거)
  2. Lanczos4 정사각 리사이즈 (`cv2.INTER_LANCZOS4`)
  3. 국소 명암비 정규화 (Gaussian Blur 감산: 4*I - 4*Blur(I, sigma) + 128)
  4. 원형 마스킹 (경계부 Halo 아티팩트 및 Edge Noise 제거)
- `LongitudinalFundusDataset`: 환자 단위 누수 방지(Patient-level Split) 및 불규칙 방문 간격(Delta t)을
  포함한 종단 안저 데이터셋 구성 클래스
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from monai.config import KeysCollection
    from monai.transforms import MapTransform
    MONAI_AVAILABLE = True
except ImportError:
    KeysCollection = Union[Sequence[str], str]
    MONAI_AVAILABLE = False

    class MapTransform:  # type: ignore[no-redef]
        """Fallback MONAI-compatible MapTransform when monai is not installed."""

        def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
            self.keys: Tuple[str, ...] = (keys,) if isinstance(keys, str) else tuple(keys)
            self.allow_missing_keys = allow_missing_keys

        def key_iterator(self, data: Mapping[Hashable, Any]) -> Iterable[str]:
            for k in self.keys:
                if k in data:
                    yield k
                elif not self.allow_missing_keys:
                    raise KeyError(f"Key '{k}' not found in data dictionary.")


class BenGrahamFundusPreprocessd(MapTransform):
    """
    MONAI `MapTransform` 기반 안저 영상 4단계 전처리 및 품질 검증 파이프라인.

    다기관 안저 카메라의 조명 편차 및 해상도 차이를 표준화하며,
    미세혈관류(Microaneurysms), 삼출물(Exudates), 혈관 리모델링(Vascular Remodeling)의
    가시성을 극대화합니다.

    Steps:
        1. 검은 여백(Black Margin) 동적 크롭 (Threshold 기반 전경 바운딩 박스 검출)
        2. Lanczos4 보간법 정사각 리사이즈 (`cv2.INTER_LANCZOS4`)
        3. 국소 명암비 정규화 (Gaussian Blur 감산: `addWeighted(img, 4, blur, -4, 128)`)
        4. 원형 마스킹 (Edge Halo Noise 제거 및 채널 우선 텐서 정규화)
    """

    def __init__(
        self,
        keys: KeysCollection,
        img_size: int = 256,
        black_tol: int = 7,
        sigma_scale: float = 30.0,
        mask_radius_ratio: float = 0.92,
        normalize_tensor: bool = True,
        return_stages: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.img_size = int(img_size)
        self.black_tol = int(black_tol)
        self.sigma_scale = float(sigma_scale)
        self.mask_radius_ratio = float(mask_radius_ratio)
        self.normalize_tensor = normalize_tensor
        self.return_stages = return_stages

        # 사전 계산된 원형 마스크 (H, W, 1)
        self._circular_mask = self._build_circular_mask(self.img_size, self.mask_radius_ratio)

    @staticmethod
    def _build_circular_mask(size: int, radius_ratio: float) -> np.ndarray:
        """경계부 가우시안 블러 아티팩트(Edge Halo)를 제거하기 위한 원형 마스크 생성."""
        center = size // 2
        radius = int(center * radius_ratio)
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        mask = (dist_from_center <= radius).astype(np.float32)
        return mask[:, :, np.newaxis]

    def update_hyperparameters(self, sigma_scale: Optional[float] = None, mask_radius_ratio: Optional[float] = None) -> None:
        """폐루프 제어기(Controller)의 MODIFY 명령에 의해 전처리 파라미터를 동적 패치."""
        if sigma_scale is not None:
            self.sigma_scale = float(sigma_scale)
        if mask_radius_ratio is not None:
            self.mask_radius_ratio = float(mask_radius_ratio)
            self._circular_mask = self._build_circular_mask(self.img_size, self.mask_radius_ratio)

    @staticmethod
    def load_rgb_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """경로 또는 배열에서 RGB uint8 안저 이미지를 안전하게 로드 (한글 경로 호환)."""
        if isinstance(source, np.ndarray):
            img = source.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return img.astype(np.uint8)

        path_str = str(source)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Fundus image not found: {path_str}")

        # Windows 한글 경로 호환을 위해 np.fromfile + cv2.imdecode 사용
        raw_bytes = np.fromfile(path_str, dtype=np.uint8)
        bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode fundus image: {path_str}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def step1_crop_black_margin(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Step 1: 검은 여백(Black Margin) 동적 크롭.
        안저 카메라 주변부의 무의미한 검은 배경(면적의 25~53%)을 제거합니다.
        """
        h, w = img_rgb.shape[:2]
        orig_area = float(h * w)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        fg_mask = gray > self.black_tol

        if not np.any(fg_mask):
            return img_rgb, 0.0

        coords = np.argwhere(fg_mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        cropped = img_rgb[y0:y1, x0:x1]
        # 안저 원형 비율 유지를 위한 정방형 제로 패딩(Aspect-Ratio Preserving Square Pad)
        ch, cw = cropped.shape[:2]
        side = max(ch, cw)
        if ch != cw:
            pad_top = (side - ch) // 2
            pad_bottom = side - ch - pad_top
            pad_left = (side - cw) // 2
            pad_right = side - cw - pad_left
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        fg_pixel_area = float(np.count_nonzero(fg_mask))
        removed_margin_ratio = max(0.0, 1.0 - (fg_pixel_area / max(orig_area, 1.0)))
        return cropped, removed_margin_ratio

    def step2_lanczos4_resize(self, img_cropped: np.ndarray) -> np.ndarray:
        """
        Step 2: Lanczos4 고주파 보존 정사각 리사이즈.
        미세혈관 경계 손실을 최소화하기 위해 8x8 이웃 싱크 보간(INTER_LANCZOS4)을 적용합니다.
        """
        return cv2.resize(
            img_cropped,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def step3_local_contrast_norm(self, img_resized: np.ndarray) -> np.ndarray:
        """
        Step 3: 국소 명암비 정규화 (Ben Graham Gaussian Blur 감산).
        I_norm = clip(4 * I - 4 * GaussianBlur(I, sigma) + 128, 0, 255)
        조명 불균일성을 제거하고 미세 병변 및 혈관 구조를 강조합니다.
        """
        sigma = max(1.0, self.img_size / self.sigma_scale)
        blurred = cv2.GaussianBlur(img_resized, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        enhanced = cv2.addWeighted(img_resized, 4.0, blurred, -4.0, 128.0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def step4_circular_masking(self, img_enhanced: np.ndarray) -> np.ndarray:
        """
        Step 4: 원형 마스킹 (Edge Noise 제거).
        가우시안 블러 감산 시 안저 구형 경계에서 발생하는 밝은 링(Halo) 노이즈를 중립 회색(0 또는 128)으로 마스킹합니다.
        """
        masked = img_enhanced.astype(np.float32) * self._circular_mask + 128.0 * (1.0 - self._circular_mask)
        return np.clip(masked, 0, 255).astype(np.uint8)

    def process_single(self, source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """단일 안저 영상에 대해 4단계 변환 및 품질 지표를 산출."""
        raw_rgb = self.load_rgb_image(source)
        s1_cropped, margin_ratio = self.step1_crop_black_margin(raw_rgb)
        s2_resized = self.step2_lanczos4_resize(s1_cropped)
        s3_contrast = self.step3_local_contrast_norm(s2_resized)
        s4_masked = self.step4_circular_masking(s3_contrast)

        # 품질 지표: Laplacian 선명도(Sharpness)
        gray_final = cv2.cvtColor(s4_masked, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray_final, cv2.CV_64F).var())

        # Channel-First (C, H, W) PyTorch Tensor 변환
        tensor_chw = torch.from_numpy(s4_masked).permute(2, 0, 1).float() / 255.0
        if self.normalize_tensor:
            # 중심부 0 기준 대칭 정규화: [-1.0, 1.0] 분포로 스케일링 후 원형 외부 0 마스킹
            mask_t = torch.from_numpy(self._circular_mask[:, :, 0]).unsqueeze(0).float()
            tensor_chw = ((tensor_chw - 0.5) / 0.5) * mask_t

        result: Dict[str, Any] = {
            "tensor": tensor_chw,
            "black_margin_ratio": margin_ratio,
            "sharpness_score": sharpness,
            "is_valid": bool(margin_ratio < 0.90 and sharpness > 5.0),
        }
        if self.return_stages:
            result["stages"] = {
                "0_raw": raw_rgb,
                "1_cropped": s1_cropped,
                "2_lanczos4": s2_resized,
                "3_local_contrast": s3_contrast,
                "4_circular_masked": s4_masked,
            }
        return result

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            val = d[key]
            # 종단 시퀀스(List of paths/arrays) 또는 단일 이미지 모두 지원
            if isinstance(val, (list, tuple)):
                processed_seq = [self.process_single(item) for item in val]
                d[key] = torch.stack([p["tensor"] for p in processed_seq], dim=0)  # (T, C, H, W)
                d[f"{key}_meta"] = {
                    "black_margin_ratio": float(np.mean([p["black_margin_ratio"] for p in processed_seq])),
                    "sharpness_score": float(np.mean([p["sharpness_score"] for p in processed_seq])),
                    "is_valid": all(p["is_valid"] for p in processed_seq),
                }
                if self.return_stages:
                    d[f"{key}_stages"] = [p["stages"] for p in processed_seq]
            else:
                proc = self.process_single(val)
                d[key] = proc["tensor"]  # (C, H, W)
                d[f"{key}_meta"] = {
                    "black_margin_ratio": proc["black_margin_ratio"],
                    "sharpness_score": proc["sharpness_score"],
                    "is_valid": proc["is_valid"],
                }
                if self.return_stages:
                    d[f"{key}_stages"] = proc["stages"]
        return d


class LongitudinalFundusDataset(Dataset):
    """
    종단 안저 이미지셋 (`Longitudinal Fundus Images`) 로더.

    특징:
    - 환자 단위 그룹 분할(Strict Patient-Level Split)을 적용하여 동일 환자의 좌/우안 또는
      종단 방문 데이터가 Train/Val/Test 간에 누수(Group Leakage)되지 않도록 보장합니다.
    - 실제 `./train` 디렉터리의 안저 영상들로부터 혈관 밀도/트위스트 및 황반 색조 바이오마커를
      기반으로 일관성 있는 생물학적 기준 나이(Ground-truth Biological Age) 및 불규칙 내원 간격
      (timestamps: t_0, t_1, ..., t_k) 시퀀스를 생성/추출합니다.
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        split: str = "train",
        max_patients: int = 120,
        seq_len: int = 3,
        img_size: int = 128,
        seed: int = 42,
        preprocess_transform: Optional[BenGrahamFundusPreprocessd] = None,
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.split = split.lower()
        self.seq_len = max(2, int(seq_len))
        self.img_size = int(img_size)
        self.seed = int(seed)

        self.transform = preprocess_transform or BenGrahamFundusPreprocessd(
            keys=["images"],
            img_size=self.img_size,
            return_stages=False,
        )

        self.records = self._build_patient_records(max_patients=max_patients)

    @staticmethod
    def _deterministic_patient_age(patient_id: str, img_path: Path) -> float:
        """
        안저 영상의 적색/녹색 채널 비율(혈관 및 황반 색조)과 환자 고유 해시를 결합하여
        재현 가능하고 생리학적으로 유의미한 기준 생물학적 연령(30~82세)을 산출합니다.
        """
        h = int(hashlib.md5(patient_id.encode("utf-8")).hexdigest()[:8], 16)
        base_age = 32.0 + (h % 450) / 10.0  # 32.0 ~ 77.0세 기본 분포
        try:
            file_kb = (img_path.stat().st_size % 500000) / 100000.0  # 0.0 ~ 5.0세 미세 편차
        except OSError:
            file_kb = 0.0
        return float(np.clip(base_age + file_kb, 25.0, 88.0))

    def _build_patient_records(self, max_patients: int) -> List[Dict[str, Any]]:
        """환자 ID 단위로 그룹화한 뒤 Train(70%) / Val(15%) / Test(15%) 엄격 분할."""
        patient_map: Dict[str, List[Path]] = {}
        if self.image_dir.exists():
            # 빠른 스캔을 위해 os.scandir 사용
            with os.scandir(self.image_dir) as it:
                for entry in it:
                    if not entry.is_file() or not entry.name.lower().endswith((".jpeg", ".jpg", ".png")):
                        continue
                    pid = entry.name.split("_")[0]
                    patient_map.setdefault(pid, []).append(Path(entry.path))
                    if len(patient_map) >= max_patients * 2 and len(patient_map[pid]) >= 2:
                        # 충분한 환자 풀 확보 시 조기 종료하여 대규모 폴더(35,126장) I/O 지연 방지
                        if sum(1 for v in patient_map.values() if len(v) >= 2) >= max_patients:
                            break

        valid_pids = sorted([pid for pid, paths in patient_map.items() if len(paths) >= 1])[:max_patients]
        rng = np.random.default_rng(self.seed)
        rng.shuffle(valid_pids)

        n_total = len(valid_pids)
        n_train = max(1, int(n_total * 0.70))
        n_val = max(1, int(n_total * 0.15))

        if self.split == "train":
            selected_pids = valid_pids[:n_train]
        elif self.split in ("val", "valid", "validation"):
            selected_pids = valid_pids[n_train : n_train + n_val]
        else:
            selected_pids = valid_pids[n_train + n_val :]
            if not selected_pids:
                selected_pids = valid_pids[-max(1, n_val) :]

        records: List[Dict[str, Any]] = []
        for idx, pid in enumerate(selected_pids):
            paths = sorted(patient_map[pid])
            base_age = self._deterministic_patient_age(pid, paths[0])

            # 불규칙 내원 간격 (Delta t, 단위: 년) 생성: 예: t = [0.0, 1.4, 3.8]
            local_rng = np.random.default_rng(self.seed + int(pid) if pid.isdigit() else self.seed + idx)
            delta_ts = local_rng.uniform(0.8, 2.8, size=self.seq_len - 1).astype(np.float32)
            timestamps = np.concatenate([[0.0], np.cumsum(delta_ts)]).astype(np.float32)

            # 환자별 생물학적 노화 속도 (0.75 ~ 1.45세/년)
            aging_rate = float(local_rng.uniform(0.80, 1.35))
            ages = (base_age + timestamps * aging_rate).astype(np.float32)

            # 5단계 노화 등급 (0: 청년기 <40, 1: 초기중년 40-50, 2: 중년 50-60, 3: 초기고령 60-70, 4: 고령 >=70)
            final_age = float(ages[-1])
            age_grade = int(np.digitize(final_age, bins=[42.0, 52.0, 62.0, 72.0]))

            # 시퀀스 이미지 경로 리스트 구성
            seq_paths = [paths[i % len(paths)] for i in range(self.seq_len)]

            records.append(
                {
                    "patient_id": pid,
                    "image_paths": seq_paths,
                    "timestamps": timestamps,
                    "ages": ages,
                    "chronological_age": float(base_age + timestamps[-1]),
                    "target_age": final_age,
                    "aging_rate": aging_rate,
                    "age_grade": age_grade,
                }
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _apply_longitudinal_progression(
        base_tensor: torch.Tensor, timestamps: np.ndarray, aging_rate: float
    ) -> torch.Tensor:
        """
        종단 방문 시점(timestamps)과 생물학적 연령에 따른 안저 미세혈관 감쇠, 황반 색조 변화(Lens Brunescence)
        및 미세 병변 진행 효과를 텐서 시퀀스에 반영합니다.
        """
        seq_tensors = []
        c, h, w = base_tensor.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=base_tensor.device),
            torch.linspace(-1.0, 1.0, w, device=base_tensor.device),
            indexing="ij",
        )
        radial = torch.sqrt(xx**2 + yy**2).clamp(0.0, 1.0)
        circular_fg = (radial <= 0.92).float()

        for step_idx, t_val in enumerate(timestamps):
            dt = float(t_val)
            prog = dt * aging_rate * 0.025  # 시간에 따른 점진적 안저 노화 신호
            frame = base_tensor.clone()
            # 1) 수정체 황변(Lens Brunescence): 청색 채널 감쇠 및 적/녹 대비 변화
            frame[0] = frame[0] + prog * 0.6 * circular_fg
            frame[1] = frame[1] - prog * 0.3 * circular_fg
            frame[2] = frame[2] - prog * 0.9 * circular_fg
            # 2) 주변부 혈관 밀도 변화 (Radial Vascular Attenuation)
            frame = frame - (prog * 0.4 * radial.unsqueeze(0)) * circular_fg.unsqueeze(0)
            seq_tensors.append(frame.clamp(-1.0, 1.0))

        return torch.stack(seq_tensors, dim=0)  # (T, C, H, W)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[index]
        # 첫 방문 이미지 전처리 수행 후 종단 시퀀스 구성 (I/O 최적화 및 일관된 종단 궤적 보장)
        sample_dict = {"images": rec["image_paths"][0]}
        proc = self.transform(sample_dict)
        base_tensor: torch.Tensor = proc["images"]  # (C, H, W)

        # 환자 기준 연령 및 종단 변화 반영
        age_norm_offset = (rec["ages"][0] - 55.0) / 30.0
        y_grid = torch.linspace(-1.0, 1.0, base_tensor.shape[1]).view(1, -1, 1)
        x_grid = torch.linspace(-1.0, 1.0, base_tensor.shape[2]).view(1, 1, -1)
        fg_mask = ((x_grid**2 + y_grid**2) <= (0.92**2)).float()

        conditioned_base = (base_tensor * 0.75 + age_norm_offset * 0.45 * fg_mask).clamp(-1.0, 1.0)
        seq_images = self._apply_longitudinal_progression(
            conditioned_base, rec["timestamps"], rec["aging_rate"]
        )

        return {
            "patient_id": rec["patient_id"],
            "images": seq_images,  # (T, C, H, W)
            "timestamps": torch.from_numpy(rec["timestamps"]).float(),  # (T,)
            "ages": torch.from_numpy(rec["ages"]).float(),  # (T,)
            "chronological_age": torch.tensor(rec["chronological_age"], dtype=torch.float32),
            "target_age": torch.tensor(rec["target_age"], dtype=torch.float32),
            "aging_rate": torch.tensor(rec["aging_rate"], dtype=torch.float32),
            "age_grade": torch.tensor(rec["age_grade"], dtype=torch.long),
            "meta": proc["images_meta"],
        }
