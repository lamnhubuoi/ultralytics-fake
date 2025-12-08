from __future__ import annotations

import base64
import io
from pathlib import Path

from ultralytics.solutions.dashboard_server import DashboardServer


class DummyResult:
    def __init__(self, save_dir: Path, speed: dict[str, float]) -> None:
        self.save_dir = Path(save_dir)
        self.speed = speed


class DummyModel:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)

    def predict(self, source: str, *, project: str, name: str, save: bool, exist_ok: bool, verbose: bool):
        dest_dir = Path(project) / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        src_path = Path(source)
        dest_path = dest_dir / src_path.name
        dest_path.write_bytes(src_path.read_bytes())
        return [DummyResult(dest_dir, {"inference": 12.34})]


_PNG_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAAJ/U4kAAAAASUVORK5CYII=")


def test_health_endpoint_reports_ok(tmp_path: Path):
    server = DashboardServer(model=DummyModel(tmp_path / "results"), root=tmp_path / "runs")
    client = server.app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"


def test_predict_endpoint_returns_saved_result(tmp_path: Path):
    server = DashboardServer(model=DummyModel(tmp_path / "results"), root=tmp_path / "runs")
    client = server.app.test_client()

    response = client.post(
        "/predict",
        data={"file": (io.BytesIO(_PNG_BYTES), "test.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()

    output_path = server.results_dir / payload["result_path"]
    assert output_path.exists()
    assert payload["result_url"].startswith("/results/")
    assert payload["timings_ms"].get("inference") == 12.34
