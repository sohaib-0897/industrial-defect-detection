"""Verify an already-running Compose service with real KSDD2 HTTP uploads."""

import json
import time
from pathlib import Path

import httpx

from src.common import write_json


def check() -> None:
    base = "http://127.0.0.1:8011"
    with httpx.Client(timeout=60, trust_env=False) as client:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                health = client.get(base + "/health")
                if health.status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("Container readiness failed; inspect docker compose logs")
        predictions = []
        for filename, expected in (("10000.png", "normal"), ("10021.png", "defective")):
            with (Path("data/KolektorSDD2/train") / filename).open("rb") as image:
                response = client.post(base + "/predict", files={"file": (filename, image, "image/png")})
            response.raise_for_status()
            assert response.json()["predicted_class"] == expected
            predictions.append({"sample": filename, "status": response.status_code, **response.json()})
        invalid = client.post(base + "/predict", files={"file": ("corrupt.png", b"broken", "image/png")})
        assert invalid.status_code == 422
    report = {
        "transport": "real HTTP to running Docker Compose service on port 8011",
        "health_status": health.status_code,
        "health": health.json(),
        "predictions": predictions,
        "corrupted_image_status": invalid.status_code,
    }
    write_json(Path("reports/ksdd2/docker-http.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    check()
