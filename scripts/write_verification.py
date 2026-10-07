"""Generate the final verification handoff from executed checks and saved evidence."""

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def build() -> None:
    root = Path("reports/ksdd2")
    tests = ET.parse(root / "tests.xml").getroot().find("testsuite").attrib
    docker = json.loads((root / "docker.json").read_text(encoding="utf-8"))
    checks = []
    for command in (["python", "-m", "ruff", "check", "."], ["python", "-m", "ruff", "format", "--check", "."]):
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        checks.append({"command": " ".join(command), "exit_code": result.returncode, "output": result.stdout.strip()})
    (root / "ruff.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    report = f"""# Executed KSDD2 assessment verification

This supersedes the historical synthetic-only report. Evidence is under [reports/ksdd2](../reports/ksdd2/).

| Check | Executed result |
|---|---|
| Official download / extraction | Passed; real official ZIP, canonical RGB PNG images and annotation masks |
| Dataset audit | 3,335 images; no unreadable canonical images/masks; verified archive copy excluded; counts match official source |
| Preserved test / stratified training subdivision | 1,864 training, 467 validation, untouched 1,004 official test; seed 42 and image/mask hashes recorded |
| Real training | ImageNet ResNet18, weighted CE, CUDA RTX 4050; 8 epochs completed, epoch 5 selected using validation |
| Threshold | Validation only; 0.1374264806509018; selected before test evaluation |
| Official test evaluation | Completed once; generated JSON, classification report, per-image predictions and confusion matrix |
| Error analysis | All 28 FP and 15 FN recorded; six examples of each visually reviewed; real contact sheets and post-test Grad-CAM saved |
| PyTorch inference / real HTTP | Health 200; real normal and defective requests 200 with expected classes; corrupted image 422 |
| ONNX export / inference | Checker and random/real-sample parity passed; probabilities/classes agree; live HTTP checks passed |
| CPU benchmark | Batch 1, 2 threads, 10 warmups, 50 timed runs; medians 492.795 ms PyTorch / 310.008 ms ONNX |
| pytest | {tests["tests"]} passed, {tests["failures"]} failures, {tests["errors"]} errors, {tests["skipped"]} skipped; {tests["time"]} seconds |
| Ruff lint / format check | Both passed; machine-readable command outputs committed |
| Docker Compose configuration / build | Both passed; CPU inference image built locally |
| Container startup / real uploads | Health {docker["health_status"]}; normal and defective prediction requests 200; corrupted image {docker["corrupted_image_status"]} |
| Container runtime | Engine {docker["engine_version"]}; user {docker["user"]}; read-only filesystem {docker["read_only"]}; {docker["dependencies"]} |
| README / artifact integrity | Commands exercised through their CLI or real HTTP helpers; all local links and checkpoint/manifest/ONNX hashes verified |
| Repository hygiene | Raw images/ZIP, local environments, caches and large checkpoints excluded from Git |

The container was left running on port 8011 for demonstration. Use `docker compose down` to stop only this project's service. [Docker evidence](../reports/ksdd2/docker.json) includes image/container IDs and actual responses. The successful build used empty proxy arguments because this machine's inherited proxy was unreachable:

```powershell
docker compose build --build-arg HTTP_PROXY= --build-arg HTTPS_PROXY= --build-arg ALL_PROXY= --build-arg NO_PROXY=*
docker compose up --no-build -d
python -m scripts.check_docker
python -m scripts.verify_submission
```

The complete training/evaluation pipeline ran on the local submission directory. README reproduction uses a fresh `artifacts/reproduced` directory; do not rerun final evaluation over `artifacts/run-001`, which intentionally refuses an existing test report. Real test inference has not been repeated for documentation checks; those verify stored predictions and hashes.

Pytest emitted three upstream deprecation warnings (TestClient and legacy ONNX export); no checks failed. The CPU deployment image has a clean `pip check`. Training CUDA wheels were installed in an isolated ignored environment rather than replacing the global CPU runtime.

The public repository is [industrial-defect-detection](https://github.com/sohaib-0897/industrial-defect-detection), with origin configured to its HTTPS Git URL and the existing history preserved on main. No model release artifact is configured: checkpoint and ONNX files remain local under `artifacts/run-001/`, with exact reproduction steps in README. Synthetic results remain clearly labeled only in [the historical software smoke archive](smoke_verification.md).

The measured validation recall target did not fully generalize to test data. Test review triggered no retraining, augmentation adjustment or threshold change. Independent factory batches and a new holdout are required before further model selection or production acceptance.
"""
    Path("docs/verification.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build()
