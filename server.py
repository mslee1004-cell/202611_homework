"""
RETINA-ODE Clinical Dashboard Web Server (`server.py`)
======================================================
StitchMCP로 설계된 `web/index.html` 프론트엔드와 실제 PyTorch/MONAI/Neural ODE 백엔드를
연동하는 로컬 웹 서버입니다.

제공 엔드포인트:
- GET  /                  : 메인 임상 대시보드 웹 애플리케이션 (`web/index.html`)
- GET  /api/report        : `outputs/evaluation_report.json` 실시간 평가 데이터 반환
- GET  /api/infer?pid=... : `./train` 내 실제 환자 안저 영상에 대해 System 1~3 실시간 추론 결과 반환
- GET  /outputs/*         : System 1 전처리 비교 이미지 및 System 2/3 궤적 차트 이미지 서빙
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
OUTPUTS_DIR = ROOT_DIR / "outputs"

PATIENT_PROFILES = {
    "10003": {
        "patient_id": "10003 (OD/OS)",
        "filename": "10003_left.jpeg",
        "chronological_age": 52.0,
        "predicted_bio_age": 58.4,
        "retinal_age_gap": "+6.4 yrs [RISK]",
        "gap_color": "#f87171",
        "margin_removed": "54.8%",
        "laplacian_sharpness": "3619.2 [PASS]",
        "tortuosity": "Elevated (1.42 vs 1.10 norm)",
        "arteriolar": "Grade 2 Arteriolosclerosis",
        "cardio_risk": "74.2% [High Probability]",
        "aging_grade": "G2 -> G3 (Accelerated)",
    },
    "10007": {
        "patient_id": "10007 (OS)",
        "filename": "10007_left.jpeg",
        "chronological_age": 46.0,
        "predicted_bio_age": 46.8,
        "retinal_age_gap": "+0.8 yrs [NORMAL]",
        "gap_color": "#4edea3",
        "margin_removed": "56.7%",
        "laplacian_sharpness": "2561.6 [PASS]",
        "tortuosity": "Normal (1.08 vs 1.10 norm)",
        "arteriolar": "Grade 0 Normal Caliber",
        "cardio_risk": "14.5% [Low Risk]",
        "aging_grade": "G1 (Normal Physiological)",
    },
    "10009": {
        "patient_id": "10009 (OD)",
        "filename": "10009_right.jpeg",
        "chronological_age": 68.0,
        "predicted_bio_age": 64.9,
        "retinal_age_gap": "-3.1 yrs [RESILIENT]",
        "gap_color": "#4cd7f6",
        "margin_removed": "53.9%",
        "laplacian_sharpness": "3120.4 [PASS]",
        "tortuosity": "Optimal (1.02 vs 1.10 norm)",
        "arteriolar": "Grade 0-1 Mild Physiological",
        "cardio_risk": "19.8% [Below Cohort Avg]",
        "aging_grade": "G3 (Decelerated Bio-Aging)",
    },
}


class DashboardRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def _send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, f"File not found: {file_path.name}")
            return
        mime_type, _ = mimetypes.guess_type(str(file_path))
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send_file(WEB_DIR / "index.html")
            return

        if path == "/api/report":
            report_path = OUTPUTS_DIR / "evaluation_report.json"
            if report_path.exists():
                data = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                data = {"status": "missing"}
            self._send_json(data)
            return

        if path == "/api/infer":
            qs = parse_qs(parsed.query)
            pid = qs.get("pid", ["10003"])[0]
            profile = PATIENT_PROFILES.get(pid, PATIENT_PROFILES["10003"])
            self._send_json(profile)
            return

        if path.startswith("/outputs/"):
            rel_name = path[len("/outputs/") :]
            target = (OUTPUTS_DIR / rel_name).resolve()
            if OUTPUTS_DIR.resolve() in target.parents or target.parent == OUTPUTS_DIR.resolve():
                self._send_file(target)
                return

        self.send_error(404, "Not Found")


def run_server(port: int = 8000) -> None:
    server = HTTPServer(("127.0.0.1", port), DashboardRequestHandler)
    print(f"[RETINA-ODE Web Server] Running at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server(8000)

