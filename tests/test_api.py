from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_success_prediction_and_single_load(checkpoint, image_bytes):
    application = create_app(Settings(model_path=checkpoint))
    with TestClient(application) as client:
        predictor = application.state.predictor
        assert client.get("/health").status_code == 200
        for _ in range(2):
            response = client.post("/predict", files={"file": ("fixture.png", image_bytes, "image/png")})
            assert response.status_code == 200
            result = response.json()
            assert result["predicted_class"] == "normal"
            assert abs(result["confidence"] - 0.7310586) < 1e-5
            assert result["decision_threshold"] == 0.4
            assert result["inference_time_ms"] > 0
        assert application.state.predictor is predictor


def test_bad_requests(checkpoint):
    with TestClient(create_app(Settings(model_path=checkpoint, max_upload_bytes=1000))) as client:
        assert client.post("/predict").status_code == 422
        assert client.post("/predict", files={"file": ("bad.txt", b"x", "text/plain")}).status_code == 415
        assert client.post("/predict", files={"file": ("bad.png", b"x", "image/png")}).status_code == 422
        assert client.post("/predict", files={"file": ("big.png", b"x" * 1001, "image/png")}).status_code == 413
        assert client.post("/predict", content=b"x" * 70000, headers={"content-type": "image/png"}).status_code == 413
        # Chunked/no-content-length requests are bounded before the multipart parser.
        assert client.post("/predict", content=iter([b"x" * 35000, b"x" * 35000])).status_code == 413


def test_missing_model_is_unavailable(tmp_path, image_bytes):
    with TestClient(create_app(Settings(model_path=tmp_path / "missing.pt"))) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", files={"file": ("valid.png", image_bytes, "image/png")}).status_code == 503


def test_pixel_limit_and_spoofed_content(checkpoint, image_bytes, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with TestClient(create_app(Settings(model_path=checkpoint))) as client:
        assert client.post("/predict", files={"file": ("large.png", image_bytes, "image/png")}).status_code == 422
        assert (
            client.post("/predict", files={"file": ("spoof.png", b"<html>bad</html>", "image/png")}).status_code == 422
        )
