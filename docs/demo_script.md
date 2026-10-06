# Recording script (2–3 minutes)

**Recording status:** the real-results demonstration is blocked because the assessment dataset could not be located after an extended local search. Bracketed fields below are intentionally unfilled. Do not present the archived smoke measurements as assessment results.

Preparation: run the real-data commands in README, select one actual error from the contact sheets, set MODEL_PATH to the trained checkpoint and start uvicorn. Open dataset_analysis.md, evaluation.json, the confusion matrix, the error sheet and the terminal. If data remains unavailable, use the alternate wording below; never present synthetic figures as manufacturing results.

**0:00–0:20 — Problem**

“This service classifies manufacturing images as normal or defective. The main trade-off is between missing a defect and sending good products for manual review. I built a small training pipeline and an API so the decision is reproducible from the dataset to deployment.”

**0:20–0:45 — Dataset**

“The audit reports [actual count] images, [normal count] normal and [defective count] defective. It also checks image sizes, corrupt files and duplicates. I split roughly 70 percent for training and 15 percent each for validation and test, keeping [actual product/session/duplicate groups] together to avoid leakage.”

If the dataset is still missing: “The real assessment dataset could not be located, so I cannot report manufacturing results. The software has passed its tests using explicitly synthetic fixtures. They are not a substitute for assessment data.”

**0:45–1:10 — Model and training**

“I chose ResNet18 because it has a standard pretrained backbone, is easy to explain and exports cleanly. I replaced the final layer, trained the head first, then fine-tuned the backbone at one tenth of the head learning rate. I preserve the complete frame during resizing so edge defects are visible. Augmentation changes orientation and lighting mildly. The training counts determine whether weighted cross-entropy is needed; the actual weights are [read config.json].”

**1:10–1:40 — Metrics and one error**

“The held-out test result is [precision], [recall] and [F1], at a threshold of [validation-selected threshold]. The threshold was chosen on validation data to target 95 percent defect recall, then maximize precision. Here is one [actual false positive/false negative]. Its true label is [label], prediction [label], and confidence [value]. The visible issue is [verified observation]; I cannot establish its cause from a single example. A missed defect is usually more expensive, but lowering the threshold also increases review load.”

If real results are missing: “These are synthetic integration outputs. I can show an actual software-generated error on a synthetic texture, but I cannot claim a failure pattern for manufacturing images. The real metrics and error review remain blocked until the assessment dataset is supplied.”

**1:40–2:10 — API**

“The model loads once at startup. I’ll check health, then upload an image with curl. The response includes the class, softmax confidence, defect probability, threshold, model version and measured inference time. The API rejects invalid images and oversized requests. Confidence is not calibrated certainty.”

Run: `curl.exe http://localhost:8000/health`, then `curl.exe -X POST -F "file=@sample.jpg" http://localhost:8000/predict`. Set the actual sample path before recording.

**2:10–2:40 — Deployment and trade-off**

“The Dockerfile packages the model and runs as a non-root user. [Show a live container only if Docker has actually been tested.] I also export ONNX and compare its outputs numerically with PyTorch. The repeatable CPU benchmark measured [actual medians] using equal thread counts; these exclude HTTP overhead. The most important trade-off is recall versus manual review, and the next step is validation on independent products under real production lighting.”

If Docker remains unavailable: “Docker configuration is included, but the local engine was stopped, so I have not verified the container build or startup.”
