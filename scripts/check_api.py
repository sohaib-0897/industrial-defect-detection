"""Launch an actual uvicorn process, exercise HTTP, and stop that process."""

import argparse
import json
import mimetypes
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from src.common import write_json


def check(checkpoint: Path, sample: Path, backend: str = "pytorch") -> dict:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    model_path = checkpoint if backend == "pytorch" else checkpoint.with_suffix(".onnx")
    environment = {**os.environ, "MODEL_PATH": str(model_path.resolve()), "INFERENCE_BACKEND": backend}
    log_path = checkpoint.parent / f"api-{backend}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
            env=environment,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            with httpx.Client(timeout=10) as client:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"uvicorn exited; inspect {log_path}")
                    try:
                        health = client.get(base + "/health")
                        if health.status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.25)
                else:
                    raise RuntimeError(f"API did not become ready; inspect {log_path}")
                with sample.open("rb") as image:
                    content_type = mimetypes.guess_type(sample.name)[0] or "application/octet-stream"
                    result = client.post(base + "/predict", files={"file": (sample.name, image, content_type)})
                result.raise_for_status()
                invalid = client.post(base + "/predict", files={"file": ("invalid.png", b"broken", "image/png")})
                assert invalid.status_code == 422
                report = {
                    "backend": backend,
                    "health_status": health.status_code,
                    "health": health.json(),
                    "prediction_status": result.status_code,
                    "prediction": result.json(),
                    "malformed_image_status": invalid.status_code,
                    "transport": "real HTTP / uvicorn",
                    "sample": str(sample),
                }
                write_json(checkpoint.parent / f"api-{backend}.json", report)
                return report
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/run-001/best.pt"))
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch")
    args = parser.parse_args()
    print(json.dumps(check(args.checkpoint, args.sample, args.backend), indent=2))
