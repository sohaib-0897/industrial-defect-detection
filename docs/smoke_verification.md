# Measured verification — synthetic fixtures only

Historical software smoke evidence only. Current real KSDD2 assessment results are in [README](../README.md) and [verification](verification.md). The historical missing-dataset and Docker status below no longer describe the current submission.

Recorded on 2026-10-04, Python 3.12.9, Windows 11, Intel64 Family 6 Model 186 Stepping 2, PyTorch 2.14.0 CPU, torchvision 0.29.0. CUDA was unavailable. The initial workspace had no files or dataset. No manufacturing quality result is claimed.

## What ran

- Audit of the repository: zero provided images; real-data analysis explicitly blocked.
- Synthetic audit: 80 PNG images, 256×256, 40 normal / 40 defective, no corrupt images, no exact/near duplicate pairs. Split: train 56 (28/28), validation 12 (6/6), test 12 (6/6). Loss weights `[1,1]`.
- Two ImageNet-initialized ResNet18 epochs: one frozen backbone, one unfrozen, AdamW head `1e-3` / backbone `1e-4`, batch 8, two CPU threads. Epoch 2 selected by validation F1/loss.
- Validation threshold selection, one held-out evaluation, error sheets, ONNX export and graph/parity checks.
- 14 tests passed; Ruff lint and format checks passed. Tests include an actual untrained fixed-output model fixture separately from the trained synthetic integration model.
- Real uvicorn processes for PyTorch and ONNX: health 200, uploaded PNG prediction 200, malformed PNG 422. The check script stopped both processes afterward.
- Docker engine probe failed: `dockerDesktopLinuxEngine` named pipe unavailable. Docker build/start was not run.
- Pinned requirements and constraints resolve against the installed project dependencies. Global `pip check` reports an unrelated `aisuite 0.1.14` requirement of `httpx<0.28`, whereas this project uses `httpx 0.28.1`. Use the documented clean virtual environment; a fresh install was not performed here. Upstream TestClient and legacy ONNX-export deprecation warnings did not fail checks.

## Synthetic evaluation

| Measure | Measured value |
|---|---:|
| Accuracy | 0.916667 |
| Defect precision | 1.000000 |
| Defect recall | 0.833333 |
| Defect F1 | 0.909091 |
| Threshold from validation | 0.999998927116394 |
| Validation recall at threshold | 1.000000 |
| Test items | 12 |

Confusion matrix (rows true, columns predicted, normal then defective):

```text
                  predicted normal   predicted defective
true normal               6                   0
true defective            1                   5
```

The image is saved locally at `artifacts/smoke/run/confusion_matrix.png`; regenerate it with the verification command after cloning.

There were zero false positives and one false negative, `defective/SYNTHETIC_defective_039.png`. Its defect probability was `0.9999959468841553`, below the selected cutoff, so predicted class was normal with class probability `0.0000040531158447265625`. The contact sheet shows a visible black diagonal synthetic mark near the upper right on a colorful texture. It does not support conclusions about manufacturing lighting, blur or label noise. The numeric decision shows how a threshold chosen from twelve strongly separated validation fixtures can be unstable on twelve other fixtures. The threshold was not changed after inspecting this error. These simple textures are easy to fit and are not independent manufacturing products.

Full machine-readable evaluation, test predictions, trade-off table and error records remain under `artifacts/smoke/run/` (ignored). The historical summary is archived locally as `artifacts/smoke_verification.json` (ignored); `artifacts/verification.json` now records the real-data finalization blocker.

## ONNX parity and latency

ONNX opset 17 passed graph validation and four parity inputs: seeded random batches of size 1 and 2, one normal fixture and one defective fixture. Maximum absolute logit difference: `3.814697265625e-6`; tolerances `atol=1e-5`, `rtol=1e-4`. The two API sample predictions agreed. Numerical parity does not guarantee class agreement for every input arbitrarily close to the threshold.

Measured batch-1 RGB preprocessing + inference + threshold, two CPU threads, ten warmups and fifty sequential repetitions per backend:

| Backend | Mean ms | Median ms | p95 ms |
|---|---:|---:|---:|
| PyTorch CPU | 34.729 | 33.763 | 47.078 |
| ONNX Runtime CPU | 19.852 | 18.729 | 25.881 |

ONNX was faster in this specific run. These timings exclude decode, HTTP, startup and concurrent load, and vary with machine load. They use the synthetic checkpoint; benchmark the real trained artifact separately.

Example actual PyTorch HTTP response for `SYNTHETIC_normal_000.png` (first request, rather than warmed benchmark):

```json
{
  "predicted_class": "normal",
  "confidence": 0.9999999999996123,
  "defect_probability": 3.8773915718784413e-13,
  "model_version": "run",
  "decision_threshold": 0.999998927116394,
  "inference_time_ms": 50.521
}
```

The ONNX first request measured 25.985ms with the same class and probabilities. These are observations, not latency guarantees or calibrated confidence.

## Repeat and inspect

```bash
python -m scripts.verify --output artifacts/smoke-reproduce
```

Use a fresh path on subsequent runs. The same software, seeds and CPU configuration should reproduce the synthetic scores; timings can differ. The script checks live HTTP for both backends and cleans up its processes.

To try curl locally with the already-generated smoke artifact in PowerShell:

```powershell
$env:MODEL_PATH = 'artifacts/smoke/run/best.pt'
$env:INFERENCE_BACKEND = 'pytorch'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
curl.exe http://localhost:8000/health
curl.exe -X POST -F "file=@artifacts/smoke/data/normal/SYNTHETIC_normal_000.png" http://localhost:8000/predict
```

For the real assessment, follow README's `data/` → audit → training → evaluation sequence, inspect FP/FN examples, update its real-results table, and replace all synthetic demo examples. Start Docker Desktop, then build and run the image only after the real checkpoint is present in `artifacts/run-001/`.
