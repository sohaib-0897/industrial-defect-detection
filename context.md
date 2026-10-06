# Project handoff

Last updated: 2026-10-04. Workspace: `C:/Users/Sohaib/Desktop/New folder (2)`.

## Objective

Complete a Computer Vision Engineer assessment: binary manufacturing inspection
(`normal=0`, `defective=1`), transfer learning, defensible evaluation and deployment.
Keep choices explainable and scope realistic for 6-8 hours of engineering.
Never fabricate dataset observations, metrics, benchmarks or verification claims.

## Current status and blockers

- The workspace was empty initially; the full software pipeline has been built.
- User later stated that the real assessment dataset was available and asked to discover it.
- Recursive inspection found only 160 generated SYNTHETIC images and four generated plots.
- Extended search covered common user folders, temporary locations and ZIP contents.
- No identifiable manufacturing dataset was found; unrelated PPE datasets were excluded.
- Search pruned system/dependency folders and encountered 22 filesystem access errors.
- Real dataset counts, distribution, splits, threshold, metrics and errors remain unknown.
- No real model has been trained; `artifacts/run-001/best.pt` does not exist.
- Docker CLI exists, but the Docker engine named pipe is unavailable.
- Obtain the real dataset's accessible path before real training; start Docker for container checks.

## Implemented pipeline

- `src/analyze_data.py`: image audit, corruption detection, label checks and grouped splits.
- Folder aliases infer labels; arbitrary layouts require CSV columns `path,label,group`.
- Exact decoded-pixel duplicates and near-duplicates are grouped before splitting.
- Near-duplicate heuristic: dHash distance <=2, exhaustive only up to 5,000 images.
- Explicit product/session IDs use metadata or `--group-regex`; IDs are not guessed.
- Seed 42; candidate group splits target 70/15/15 per class and require both classes.
- Frozen manifest contains paths, labels, groups, hashes and split membership.
- Shared preprocessing: EXIF-corrected RGB, full-frame 224px letterbox, ImageNet normalization.
- Mild rotation/translation and brightness/contrast augmentation; no flips or random crops.
- ResNet18 starts from ImageNet weights; classifier outputs two logits.
- Head training precedes full fine-tuning; backbone LR is one tenth of head LR.
- AdamW, fixed BatchNorm running statistics, scheduler, early stopping and best checkpoint.
- Weighted cross-entropy only when training majority/minority ratio is at least 1.5.
- Validation-only threshold selection maximizes precision subject to target recall (default 95%).
- Threshold ties use F1 then highest threshold; small validation samples can make it unstable.
- Test evaluation refuses to overwrite its result; preserve the holdout and do not tune on it.
- Error analysis saves FP/FN records and representative contact sheets from saved predictions.
- ONNX export checks graph validity and logit parity; CPU benchmark uses equal thread counts.

## Service and verification

- FastAPI: `POST /predict`, `GET /health`; model loads once during startup.
- Default backend/checkpoint: PyTorch, `artifacts/run-001/best.pt`; missing model returns 503.
- ONNX requires `MODEL_PATH` pointing to `.onnx` plus adjacent verified `.json` metadata.
- Upload validation includes MIME, readable contents, size, pixel limit and single-frame checks.
- JSON returns class, class probability, defect probability, threshold, version and timing.
- Confidence is an uncalibrated class probability; thresholded class need not be argmax.
- Latest checks: 14 tests passed; Ruff lint/format and Docker Compose validation passed.
- Historical synthetic training, ONNX parity and both live HTTP backends were verified.
- These checks do not establish manufacturing quality; real API predictions remain blocked.
- Docker build/start was not run. No real-data evaluation or latency measurement was performed.
- Global Python has an unrelated aisuite/httpx conflict; use a clean Python 3.12 environment.

## Reports and artifacts

- Current truth: `artifacts/verification.json`, `artifacts/dataset_analysis.{json,md}`.
- Main docs: `README.md`, `docs/verification.md`, `docs/demo_script.md`.
- Historical smoke numbers: explicitly labeled `docs/smoke_verification.md`.
- Smoke images/models/results remain locally under `artifacts/smoke*` and are ignored.
- Old smoke JSON is `artifacts/smoke_verification.json`, also ignored.
- Keep datasets, environments, caches, secrets, `.pt` and `.onnx` files out of Git.
- Commit source/tests/scripts/docs/config/Docker files and permitted small audit reports.

## Resume with real data

Run from the workspace root; put real images in `data/` or substitute their actual root.
Audit filename/product/session grouping first; add `--metadata` or `--group-regex` as needed.
Use a fresh training output directory; evaluation is intentionally once-only per run.

```bash
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m src.analyze_data --data data --output artifacts
python -m src.train --data data --manifest artifacts/splits.csv --output artifacts/run-001
python -m src.evaluate --data data --manifest artifacts/splits.csv --checkpoint artifacts/run-001/best.pt
python -m src.error_analysis --data data --run artifacts/run-001
python -m src.export_onnx --checkpoint artifacts/run-001/best.pt --samples data/normal/sample.png
python -m src.benchmark --checkpoint artifacts/run-001/best.pt --sample data/normal/sample.png
python -m scripts.check_api --checkpoint artifacts/run-001/best.pt --sample data/normal/sample.png
python -m scripts.check_api --checkpoint artifacts/run-001/best.pt --sample data/normal/sample.png --backend onnx
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
docker build -t defect-detector .
docker run --rm -p 8000:8000 defect-detector
```

Replace sample paths with real images. Update README, verification and demo script with
actual results only after these steps succeed. Retain honest limitations and error findings.
