import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.inference import Predictor
from app.main import create_app
from src.export_onnx import export


def test_onnx_export_api_and_metadata_integrity(checkpoint, image_bytes):
    report = export(checkpoint, [])
    assert all(row["classes_agree"] for row in report["probability_and_class_parity"])
    path = checkpoint.with_suffix(".onnx")
    with TestClient(create_app(Settings(model_path=path, backend="onnx"))) as client:
        assert client.get("/health").json()["backend"] == "onnx"
        result = client.post("/predict", files={"file": ("image.png", image_bytes, "image/png")})
        assert result.status_code == 200
        assert result.json()["predicted_class"] == "normal"
        assert result.json()["defect_probability"] == pytest.approx(0.2689414, abs=1e-6)
    pytorch = Predictor(Settings(model_path=checkpoint)).predict(Image.new("RGB", (224, 224)))
    assert pytorch.defect_probability == pytest.approx(result.json()["defect_probability"], abs=1e-6)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="differs"):
        Predictor(Settings(model_path=path, backend="onnx"))
