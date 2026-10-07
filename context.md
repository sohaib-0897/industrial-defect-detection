# Project handoff

Repository: https://github.com/sohaib-0897/industrial-defect-detection. Run commands from the cloned repository root.

The assessment dataset was not provided. The user authorized the official public KSDD2 substitute. Real data now exists locally under ignored `data/KolektorSDD2/`; the official ZIP is ignored too. Dataset-derived figures are CC BY-NC-SA 4.0 with attribution in README.

The canonical dataset has 3,335 RGB images and masks; an extra byte-identical image/mask copy is excluded with hash evidence. Labels derive from nonzero mask pixels. Official test membership is preserved. Seed-42 stratified official-training subdivision gives train 1,864 (1,667 normal/197 defective), validation 467 (418/49), test 1,004 (894/110). The versioned manifest records image and mask hashes.

The selected ImageNet ResNet18 uses full-frame 224-wide by 640-high letterboxing, mild brightness/contrast augmentation, weighted CE, one head-only epoch and full-backbone fine-tuning. Best epoch is 5; early stopping ends after epoch 8. Validation-only threshold is 0.1374264806509018. The official test was evaluated once: TN 866, FP 28, FN 15, TP 95. Do not rerun evaluation or tune this model against observed test errors. Further selection needs a fresh holdout.

The final checkpoint and verified ONNX files are ignored under `artifacts/run-001/`. Small generated evidence is committed under `reports/ksdd2/`. CPU medians (two threads, 10 warmups, 50 runs, preprocessing included) are PyTorch 492.795 ms and ONNX 310.008 ms. Both live HTTP backends passed normal/defective requests and corrupted-image rejection. Error findings describe visually inspected real images, not causal explanations.

Use README for exact reproduction and API commands, `docs/verification.md` for executed checks and container status, and `docs/demo_script.md` for recording. Historical synthetic measurements are isolated in `docs/smoke_verification.md`; they are software smoke evidence only. GPU training used an ignored isolated CUDA environment; global CPU PyTorch remains available for inference. Origin points to the public GitHub repository; model weights are reproduced locally, with no release artifact configured.
