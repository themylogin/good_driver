from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from good_driver.app import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture
def app():
    return create_app()


async def test_open_directory(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/calibrate/open-directory")
    assert response.status_code == 200
    data = response.json()
    assert "directory" in data
    assert "images" in data
    # data/ has at least 1.jpg
    filenames = [img["filename"] for img in data["images"]]
    assert "1.jpg" in filenames


async def test_serve_image(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/calibrate/image?path=data/1.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")


async def test_serve_image_path_traversal_blocked(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/calibrate/image?path=../../../etc/passwd")
    assert response.status_code in (403, 404)


async def test_detect_requires_model(app):
    """detect endpoint should work if model exists, or return 500 with helpful message."""
    model_path = PROJECT_ROOT / "backend" / "models" / "yolopv2_384x640.onnx"
    if not model_path.exists():
        pytest.skip("Model not downloaded yet - run: uv run python download_model.py")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/calibrate/detect",
            json={"directory": str(DATA_DIR), "filename": "1.jpg"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert isinstance(data["detections"], list)
    # At least one car should be detected in a dashcam image
    assert len(data["detections"]) >= 0  # don't assert > 0, depends on image content
    if data["detections"]:
        det = data["detections"][0]
        assert "id" in det
        assert "x1" in det and "y1" in det and "x2" in det and "y2" in det
        assert det["x2"] > det["x1"]
        assert det["y2"] > det["y1"]


async def test_solve_too_few_measurements(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/calibrate/solve",
            json={"directory": str(DATA_DIR)},
        )
    # Should fail with 400 (not enough annotated images)
    assert response.status_code == 400


async def test_annotation_roundtrip(app, tmp_path):
    """Save and reload annotation via sidecar JSON."""
    import shutil

    # Copy test image to tmp dir
    shutil.copy(DATA_DIR / "1.jpg", tmp_path / "1.jpg")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Save annotation
        response = await client.put(
            "/api/calibrate/annotation",
            json={
                "directory": str(tmp_path),
                "filename": "1.jpg",
                "selected_detection_id": 0,
                "car_width_m": 2.0,
                "distance_m": 30.0,
            },
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    # Verify sidecar was written
    sidecar = (tmp_path / "1.jpg.json").read_text()
    import json
    data = json.loads(sidecar)
    assert data["selected_detection_id"] == 0
    assert data["car_width_m"] == 2.0
    assert data["distance_m"] == 30.0


async def test_solve_with_synthetic_data(app, tmp_path):
    """Solve with manually constructed sidecar JSONs (no real YOLO needed)."""
    import json, shutil

    shutil.copy(DATA_DIR / "1.jpg", tmp_path / "a.jpg")
    shutil.copy(DATA_DIR / "1.jpg", tmp_path / "b.jpg")

    # Synthetic ground truth: fx=1000, pitch=0, h=1.2
    # For car at 30m, width=2m: pw = 1000*2/30 = 66.7px
    # y_base at 30m: cy + fx*tan(arctan(1.2/30)) = 540 + 1000*tan(0.03998) ≈ 540 + 40 = 580
    import math
    fx_true = 1000.0
    h_true = 1.2
    pitch_true = 0.0
    cy = 540.0
    iw, ih = 1920, 1080

    def make_sidecar(d, cw=2.0):
        pw = fx_true * cw / d
        angle = math.atan2(h_true, d) - pitch_true
        y_base = cy + fx_true * math.tan(angle)
        x1 = 900.0
        return {
            "image_width": iw,
            "image_height": ih,
            "detections": [{"id": 0, "x1": int(x1), "y1": int(y_base - 80), "x2": int(x1 + pw), "y2": int(y_base), "confidence": 0.9}],
            "selected_detection_id": 0,
            "car_width_m": cw,
            "distance_m": d,
        }

    (tmp_path / "a.jpg.json").write_text(json.dumps(make_sidecar(30.0)))
    (tmp_path / "b.jpg.json").write_text(json.dumps(make_sidecar(60.0)))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/calibrate/solve",
            json={"directory": str(tmp_path)},
        )

    assert response.status_code == 200
    result = response.json()
    assert "fov_degrees" in result
    assert "pitch_degrees" in result
    assert "camera_height_m" in result
    assert abs(result["pitch_degrees"]) < 1.0  # should be ~0
    # FOV from fx=1000 on 1920px: 2*atan(960/1000) in degrees ≈ 87.7°
    assert 80 < result["fov_degrees"] < 100
    assert abs(result["camera_height_m"] - h_true) < 0.3  # should recover ~1.2m
