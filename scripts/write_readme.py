"""Render submission documentation directly from generated real evaluation reports."""

import json
from pathlib import Path


def build() -> None:
    root = Path("reports/ksdd2")

    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    dataset, config, history = read("dataset_analysis.json"), read("config.json"), read("history.json")
    result, threshold, parity, latency = (
        read(name) for name in ("evaluation.json", "threshold_selection.json", "best.json", "benchmark.json")
    )
    best_epoch = max(history, key=lambda row: (row["f1"], -row["val_loss"]))["epoch"]
    sizes = [tuple(map(int, size.split("x"))) for size in dataset["dimensions_width_x_height"]]
    pt, ort = latency["backends"]["pytorch"], latency["backends"]["onnx"]
    review = (root / "errors/review.md").read_text(encoding="utf-8")
    contents = f"""# Manufacturing surface-defect classification

The assessment specification stated that a dataset would be provided, but no dataset was included in the materials available to me. I contacted the recruitment team requesting access but did not receive the dataset within the submission window. To demonstrate the complete solution, I used the publicly available Kolektor Surface-Defect Dataset 2 (KSDD2), a real industrial surface-inspection dataset. The training and inference pipeline is dataset-agnostic and can be retrained on the intended assessment dataset without architectural changes.

Classify an inspection image as **normal** (0, no visible defect) or **defective** (1, visible defect). Defective is the positive class everywhere. This submission includes a trained transfer-learning model, audited splits, generated evaluation evidence, FastAPI inference, verified ONNX export, and CPU measurements. Synthetic fixtures are used only for software tests; they contribute no assessment metrics below.

## Dataset and split integrity

[KSDD2 official source](https://www2.vicos.si/resources/kolektorsdd2/) · [official ZIP](https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip).
Kolektor captured and annotated real production surfaces under controlled industrial conditions. Defects include scratches, spots and surface imperfections. This public substitute was **not supplied by Spark AI**.

License: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. Commercial dataset use requires contacting the authors; public availability does not grant commercial rights. This repository's example/contact-sheet figures derive from that dataset and retain the same attribution and license. Citation: Jakob Božič, Domen Tabernik, Danijel Skočaj, “Mixed supervision for surface-defect detection: from weakly to fully supervised learning”, *Computers in Industry*, 2021.

| Partition | Normal | Defective | Total |
|---|---:|---:|---:|
| Official training (before subdivision) | 2,085 | 246 | 2,331 |
| Training | 1,667 | 197 | 1,864 |
| Validation | 418 | 49 | 467 |
| Official test, preserved | 894 | 110 | 1,004 |
| Entire canonical dataset | 2,979 | 356 | 3,335 |

The official archive contains an extra byte-identical `10301 (copy).png` and its mask. Both hashes match their canonical originals; the audit records and excludes the duplicate pair. Canonical counts match the official documentation. All 3,335 RGB PNG images and grayscale masks decode successfully; masks contain 0 and 255, and **any nonzero annotation pixel defines defective**. Actual widths range {min(w for w, h in sizes)}–{max(w for w, h in sizes)} pixels; heights {min(h for w, h in sizes)}–{max(h for w, h in sizes)} pixels (approximately 230 × 630). No further exact decoded image duplicates were found.

Normal:defective imbalance is **{dataset["normal_to_defective_ratio"]:.2f}:1** overall and **{config["class_counts"][0] / config["class_counts"][1]:.2f}:1** in training. An always-normal test classifier already reaches 89.04% accuracy while detecting no defects; defective precision, recall and F1 are therefore essential.

Only the official training directory is subdivided, with scikit-learn stratification, 20% validation and **seed 42**. [Frozen manifest](reports/ksdd2/splits.csv) stores image and mask hashes, labels and membership; loaders reject changed data, duplicate paths and cross-split groups. The official test is excluded from optimization, early stopping, preprocessing/augmentation decisions, model selection and threshold tuning. Training examples alone informed preprocessing. Test evaluation is protected against accidental overwrite. Exact duplicate checks cannot establish independent physical products; product/session IDs and comprehensive near-duplicate verification are unavailable.

![Class distribution](reports/ksdd2/class_distribution.png)
![Official training examples; red boxes locate annotated defects](reports/ksdd2/training_examples.jpg)

Full observed dimensions, annotation details and duplicate evidence: [dataset audit](reports/ksdd2/dataset_analysis.md), [machine-readable audit](reports/ksdd2/dataset_analysis.json).

## Preprocessing, model and training

Use aspect-preserving bilinear resize with padding to **224 pixels wide × 640 high**. Keep the complete frame; no square squash or crop. Padding uses approximately the ImageNet mean RGB, followed by ImageNet normalization. Typical images retain almost their original resolution. Downsampling can still weaken extremely small defects. Validation, test and both inference backends share deterministic preprocessing, enforced by checkpoint metadata.

Training augmentation: **±2° rotation with `expand=True`** followed by padding, and **±10% brightness/contrast jitter**. The expanded canvas preserves rotated edge content. No flips, random crop, blur, erasing or artificial defects. Orientation plausibility was not assumed.

**ImageNet1K_V1-pretrained ResNet18**, with a two-output classification head, provides a lightweight, CPU-deployable baseline for only 197 training positives. No architecture search was performed. Train the head for one epoch; then unfreeze `layer4` and the head while retaining earlier pretrained filters. BatchNorm running statistics stay fixed. This limits adaptation and CPU cost on a small dataset; `--backbone-mode all` supports full fine-tuning for a future validation-controlled experiment.

Weighted cross-entropy uses `N/(2*n_class)`: **normal={config["class_weights"][0]:.6f}, defective={config["class_weights"][1]:.6f}**. Training batches remain shuffled at natural prevalence. Validation and test prevalence is unchanged.

| Setting | Actual run |
|---|---|
| Seed / device / batch size / CPU threads | 42 / {config["device_used"]} / {config["batch_size"]} / {config["threads"]} |
| Optimizer | AdamW, weight decay 0.0001 |
| Head LR / final residual-stage LR | 0.001 / 0.0001 |
| Epoch cap / early-stop patience | {config["epochs"]} / {config["patience"]} |
| Scheduler | ReduceLROnPlateau on validation F1, factor 0.5, patience 2 |
| Checkpoint rule | Highest defective validation F1 at 0.5; ties use lower validation loss |
| Epochs completed / selected epoch | {len(history)} / {best_epoch} |

Logs retain training loss, training metrics, validation loss, accuracy, precision, recall, F1 and learning rates. [Configuration](reports/ksdd2/config.json) · [epoch history](reports/ksdd2/history.json). Weighted loss is aggregated using its effective weight denominator, avoiding incorrect averages of weighted batch means. Random seeds and deterministic PyTorch operations are fixed; numerical identity across platforms is not promised.

## Validation-only operating threshold

After choosing the checkpoint, examine every distinct validation probability boundary plus 0, 0.5 and 1. Select maximum precision subject to **defective recall ≥95%**; ties use F1 then the highest threshold. The selected threshold is **{threshold["selected_threshold"]:.8f}**, with validation precision **{threshold["metrics"]["precision"]:.2%}**, recall **{threshold["metrics"]["recall"]:.2%}** and F1 **{threshold["metrics"]["f1"]:.2%}**.

A false negative may ship a defective product to a customer. A false positive may unnecessarily reject a good item or trigger manual review. The 95% target is a predeclared QC preference, not a production cost optimum. With only 49 validation defects, recall changes in steps of 2.04 percentage points; the 95% target requires at least 47 correct detections (95.92%). The threshold is never tuned on test labels and probability-based confidence is not calibrated certainty. [Complete threshold sweep and validation predictions](reports/ksdd2/threshold_selection.json).

## Final untouched official test results

One final test evaluation after checkpoint and threshold selection:

| Metric (defective positive) | Measured |
|---|---:|
| Accuracy | {result["accuracy"]:.4%} |
| Precision | {result["precision"]:.4%} |
| Recall | {result["recall"]:.4%} |
| F1 | {result["f1"]:.4%} |
| True negatives / false positives | {result["true_negatives"]} / {result["false_positives"]} |
| False negatives / true positives | {result["false_negatives"]} / {result["true_positives"]} |
| Support | 894 normal, 110 defective |

Confusion matrix (rows = ground truth; columns = prediction, order normal/defective):
`{result["confusion_matrix"]}`.

![Final confusion matrix](reports/ksdd2/confusion_matrix.png)

[Generated metrics and hashes](reports/ksdd2/evaluation.json) · [classification report](reports/ksdd2/classification_report.json) · [every image prediction](reports/ksdd2/test_predictions.json). These files are generated from model outputs; quality metrics are not manually edited.

## Actual error review

{review}

Error records contain filename, ground truth, prediction, defective probability and class-probability confidence. Contact sheets preserve full images and show annotation locations where present. Test error review is descriptive and did not trigger retraining or threshold changes. [Error records](reports/ksdd2/errors/errors.json). Optional [Grad-CAM diagnostics](reports/ksdd2/gradcam) are post-evaluation illustrations of coarse final-layer attention, not causal explanations or a segmentation result.

## Architecture and API

[Training and inference Mermaid diagrams](docs/architecture.md) show dataset audit → official separation → stratified training/validation → augmentation → ResNet18 transfer learning → validation threshold → untouched test → checkpoint → ONNX.

FastAPI validates multipart bodies before parsing and limits image uploads to 8 MiB. It checks supported image types, decoding, single frames and pixel limits. One model is loaded at startup. CPU inference is serialized per worker; use one uvicorn worker to avoid loading multiple copies. Structured logs record prediction outputs without retaining uploaded images.

`GET /health`: 200 when ready, 503 when the model is unavailable. `POST /predict`: multipart field `file`; 200 success, 413 oversized upload, 415 unsupported declared media type, 422 malformed image/missing field, 503 unavailable model. Response includes `predicted_class`, `confidence`, `defect_probability`, `decision_threshold`, `inference_time_ms`, `model_version`. Confidence equals `p(defective)` for a defective decision and `1-p(defective)` for a normal decision, even with a threshold below 0.5.

## Setup and complete reproduction

Python 3.12. CPU deployment measurements used {latency["platform"]}, {latency["processor"]} (Intel Core i5-13420H), PyTorch {latency["torch"]}. Training used {config.get("gpu_name") or "CPU"} with PyTorch {config.get("torch_version", latency["torch"])}. The CUDA wheel step follows [PyTorch's official installation index](https://pytorch.org/get-started/locally/); omit it and use `--device cpu` on a CPU-only machine. Run in the repository root (PowerShell):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m pip install --no-deps torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m scripts.download_ksdd2
python -m src.ksdd2
python -m src.train --data data/KolektorSDD2 --manifest reports/ksdd2/splits.csv --output artifacts/reproduced --epochs 10 --freeze-epochs 1 --backbone-mode last-block --batch-size {config["batch_size"]} --lr 0.001 --patience 3 --target-recall 0.95 --seed 42 --threads 4 --device {config["device_used"]}
python -m src.evaluate --data data/KolektorSDD2 --manifest reports/ksdd2/splits.csv --checkpoint artifacts/reproduced/best.pt --threads 4
python -m src.error_analysis --data data/KolektorSDD2 --run artifacts/reproduced
python -m src.export_onnx --checkpoint artifacts/reproduced/best.pt --samples data/KolektorSDD2/train/10000.png data/KolektorSDD2/train/10021.png
python -m src.benchmark --checkpoint artifacts/reproduced/best.pt --sample data/KolektorSDD2/train/10000.png --threads 2 --warmup 10 --repeats 50
python -m scripts.check_api --checkpoint artifacts/reproduced/best.pt --sample data/KolektorSDD2/train/10000.png --additional-samples data/KolektorSDD2/train/10021.png
python -m scripts.check_api --checkpoint artifacts/reproduced/best.pt --sample data/KolektorSDD2/train/10000.png --additional-samples data/KolektorSDD2/train/10021.png --backend onnx
```

Choose a fresh run directory; training rejects existing results and evaluation refuses overwriting the final test report. The locally completed submission checkpoint is `artifacts/run-001/best.pt`. Raw images, ZIPs, caches and large model artifacts are ignored in Git. Reproduction generates the checkpoint; no artifact/release remote is configured. Download/extraction alone requires roughly 2 GiB; CPU training requires additional memory and takes substantially longer than GPU training. For a different supplied dataset, use the generic `src.analyze_data` CSV/folder audit and the same training/inference architecture; KSDD2 uses its dedicated official-split adapter.

Start the local submission API:

```powershell
$env:MODEL_PATH = 'artifacts/run-001/best.pt'
$env:INFERENCE_BACKEND = 'pytorch'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

In another terminal:

```powershell
curl.exe --noproxy "*" http://127.0.0.1:8010/health
curl.exe --noproxy "*" -X POST http://127.0.0.1:8010/predict -F "file=@data/KolektorSDD2/train/10000.png;type=image/png"
curl.exe --noproxy "*" -X POST http://127.0.0.1:8010/predict -F "file=@data/KolektorSDD2/train/10021.png;type=image/png"
```

For ONNX set `MODEL_PATH=artifacts/run-001/best.onnx` and `INFERENCE_BACKEND=onnx` before startup. [Natural 2–3 minute recording script](docs/demo_script.md).

## ONNX parity and CPU latency

Export uses opset 17 and a dynamic batch axis. ONNX checker and Runtime verify random batches and real normal/defective images. Maximum absolute logit difference: **{parity["parity_max_abs_logit_error"]:.8g}**; all compared probability/class decisions agree within `atol=1e-5, rtol=1e-4`. Export metadata stores checkpoint/ONNX hashes, threshold and preprocessing contract. The API rejects a changed ONNX file.

| Backend | Median | p95 |
|---|---:|---:|
| PyTorch CPU | {pt["median_ms"]:.3f} ms | {pt["p95_ms"]:.3f} ms |
| ONNX Runtime CPU | {ort["median_ms"]:.3f} ms | {ort["p95_ms"]:.3f} ms |

Batch size 1, two CPU threads, 10 warmups, 50 timed runs. Scope: RGB preprocessing + model + threshold; excludes decoding, HTTP and startup. Results describe this laptop and its load, not production throughput. [Raw benchmark environment and results](reports/ksdd2/benchmark.json).

## Packaging and software checks

```powershell
docker compose config -q
docker compose up --build -d
curl.exe --noproxy "*" http://127.0.0.1:8011/health
curl.exe --noproxy "*" -X POST http://127.0.0.1:8011/predict -F "file=@data/KolektorSDD2/train/10021.png;type=image/png"
docker compose down
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Docker uses CPU wheels, a non-root user, read-only Compose filesystem and readiness health check. Only required model artifacts enter the build context; data and training caches stay outside the image. [Executed verification and exact Docker status](docs/verification.md). Tests cover preprocessing, mask-derived classes, official test preservation, threshold boundaries, model outputs, API readiness, real HTTP examples, invalid/oversized/corrupt uploads and ONNX metadata/parity. Synthetic unit fixtures test software behavior only.

## Limits and next steps

This is one industrial domain with only 197 training defects and 49 validation defects; transfer to a new factory is unproven. Small spots and weak scratches can remain below the classifier's effective resolution. Binary classification provides no reliable defect localization. Threshold and confidence need calibration to actual inspection costs. License limits dataset-derived commercial use. Before deployment, collect independent production batches and the intended assessment data, validate product/session separation, compare full-backbone fine-tuning and tiled higher-resolution inference on validation data, calibrate probabilities, and monitor illumination/domain drift. Any new model selected after test review requires a fresh holdout.
"""
    Path("README.md").write_text(contents, encoding="utf-8")


if __name__ == "__main__":
    build()
