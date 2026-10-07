# Executed KSDD2 assessment verification

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
| pytest | 19 passed, 0 failures, 0 errors, 0 skipped; 92.111 seconds |
| Ruff lint / format check | Both passed; machine-readable command outputs committed |
| Docker Compose configuration / build | Both passed; CPU inference image built locally |
| Container startup / real uploads | Health 200; normal and defective prediction requests 200; corrupted image 422 |
| Container runtime | Engine 29.7.2; user detector; read-only filesystem True; No broken requirements found. |
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

No Git remote or artifact release destination is configured, so there is no authorized destination to push commits or model weights. The full checkpoint and ONNX model remain available locally under `artifacts/run-001/`; README gives exact reproduction steps. Synthetic results remain clearly labeled only in [the historical software smoke archive](smoke_verification.md).

The measured validation recall target did not fully generalize to test data. Test review triggered no retraining, augmentation adjustment or threshold change. Independent factory batches and a new holdout are required before further model selection or production acceptance.
