# Recording guide (2 minutes 45 seconds)

Prepare the dataset and checkpoint with the README reproduction commands first. Open README, the report figures,
`docs/architecture.md`, and two terminals. Do not run training during the recording.

**0:00–0:20 — README disclosure and problem**

“This is a binary surface inspection service: normal means no visible defect; defective means a visible defect.
The assessment mentioned a supplied dataset, but I did not receive it after requesting access. I used the public
KolektorSDD2 industrial dataset to demonstrate the complete pipeline.”

**0:20–0:45 — `reports/ksdd2/class_distribution.png` and `training_examples.jpg`**

“There are 3,335 canonical images. Normal images outnumber defective images by about eight to one.
Labels come from the official masks. I kept the official 1,004-image test set intact and split only the official
training set, using stratification and seed 42. The download includes an extra image and mask copy;
I verified both hashes and excluded that duplicate pair.”

**0:45–1:15 — preprocessing code and architecture diagram**

“I chose ImageNet-pretrained ResNet18 because it is a small, explainable transfer-learning baseline.
I preserve the full image at 224 by 640 instead of compressing these tall images into a small square.
Augmentation uses small rotations and mild brightness and contrast changes. Weighted cross-entropy gives
defective examples more influence. I train the head, then fine-tune the final residual stage at a lower learning rate.”

**1:15–1:35 — README measured results, confusion matrix, and one error sheet**

Read the actual precision, recall, F1, FP and FN values shown in README.

“The threshold was selected on validation data to prioritize defect recall while retaining useful precision.
A missed defect may reach a customer; a false alarm may cause manual review or unnecessary rejection.
These are the results of one final evaluation on the untouched official test set. Here is one visually reviewed error.”

Point to the example's filename, annotation, probability and predicted class. Use the findings in README;
do not claim the image establishes a causal explanation.

**1:35–2:10 — terminal and real HTTP predictions**

Terminal one (PowerShell, repository root):

```powershell
$env:MODEL_PATH = 'artifacts/run-001/best.pt'
$env:INFERENCE_BACKEND = 'pytorch'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Terminal two:

```powershell
curl.exe --noproxy "*" http://127.0.0.1:8010/health
curl.exe --noproxy "*" -X POST http://127.0.0.1:8010/predict -F "file=@data/KolektorSDD2/train/10000.png;type=image/png"
curl.exe --noproxy "*" -X POST http://127.0.0.1:8010/predict -F "file=@data/KolektorSDD2/train/10021.png;type=image/png"
```

“The model loads once during startup. The API validates upload size and image decoding, then returns the class,
its probability-based confidence, defective probability, threshold, timing and model version.”

**2:10–2:30 — `reports/ksdd2/best.json`, `benchmark.json`, and Dockerfile**

“I exported the selected checkpoint to ONNX and checked logits, probabilities and predicted classes against PyTorch.
These CPU medians use warmup and repeated batch-one inference, including preprocessing.
Docker packages the same checkpoint and API as a non-root service.”

Read the measured medians. Only claim completed Docker checks listed in `docs/verification.md`.
If the container is ready, show `curl.exe --noproxy "*" http://127.0.0.1:8011/health`.

**2:30–2:45 — limitations**

“This evaluates one production domain, not the unavailable assessment dataset.
Very small, low-contrast defects remain difficult. Before production use I would collect independent factory data,
calibrate the operating threshold to rejection costs, and monitor drift. The pipeline can be retrained on the intended dataset.”
