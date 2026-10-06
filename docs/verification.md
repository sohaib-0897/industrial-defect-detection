# Real assessment verification status

Rechecked 2026-10-04. The real dataset was not located in the repository or accessible local locations searched. The repository contains only 160 SYNTHETIC fixture images and four generated plots. No real-data training, threshold selection or test evaluation was run, so the real test set has not been consumed.

| Check | Current result |
|---|---|
| Dataset discovery | Blocked: no identifiable manufacturing assessment dataset found |
| Real split and class weights | Not determined |
| Real model training / threshold | Not run |
| Real test metrics / confusion matrix / FP / FN | Not measured |
| Real image error analysis | Not performed |
| Real ONNX export / parity | Not run |
| Real PyTorch / ONNX CPU benchmark | Not measured |
| Real API predictions | Blocked: artifacts/run-001/best.pt does not exist |
| Unit/integration tests | 14 passed, zero failures/errors; use explicitly synthetic test fixtures |
| Ruff lint and format | Passed |
| Docker Compose configuration | Passed |
| Docker build/start | Not run: Docker engine named pipe unavailable |

Discovery covered hidden/ignored repository files, Desktop, Downloads, Documents, OneDrive, source, local data/temp locations and other accessible profile folders. ZIP central directories in Desktop and Downloads were inspected. Unrelated PPE images, UI screenshots and software assets were excluded. No additional mounted drive letters were found. The extended user-profile scan had 22 access errors; dependency/cache/game/system folders were pruned where appropriate. This search cannot establish absence in inaccessible locations.

Historical smoke measurements have been moved to [smoke_verification.md](smoke_verification.md), explicitly labeled synthetic. They are not manufacturing performance. The active machine-readable report is artifacts/verification.json; the old local smoke report is artifacts/smoke_verification.json and remains ignored by Git.

A TestClient/httpx deprecation warning did not fail tests. Previous global pip-check conflict with unrelated aisuite remains documented in the smoke archive; use the clean virtual environment in README.

To unblock: place the labeled assessment dataset under data/ or provide its exact accessible path. Inspect product/session metadata, then follow the audit, training, once-only evaluation, error analysis, export, benchmark and API commands in README. Start Docker Desktop before attempting the container build. Do not record a real-results demonstration until these steps complete.
