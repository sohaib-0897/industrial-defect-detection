import json
import threading
import time

import numpy as np
import torch
from PIL import Image

from app.config import Settings
from app.schemas import Prediction
from src.common import CLASS_NAMES, file_hash
from src.data import preprocessing
from src.metrics import classify
from src.model import load_checkpoint


class Predictor:
    def __init__(self, settings: Settings):
        self.backend = settings.backend
        self.transform = preprocessing()
        self.lock = threading.Lock()
        if settings.backend == "pytorch":
            torch.set_num_threads(settings.threads)
            self.model, metadata = load_checkpoint(settings.model_path)
        else:
            import onnxruntime as ort

            metadata = json.loads(settings.model_path.with_suffix(".json").read_text(encoding="utf-8"))
            if metadata["onnx_sha256"] != file_hash(settings.model_path):
                raise ValueError("ONNX model differs from verified export")
            if metadata["class_names"] != list(CLASS_NAMES) or metadata["preprocessing"] != "letterbox224-imagenet-v1":
                raise ValueError("Unsupported ONNX metadata")
            options = ort.SessionOptions()
            options.intra_op_num_threads = settings.threads
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(settings.model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        self.threshold, self.version = float(metadata["threshold"]), metadata["model_version"]
        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid decision threshold")

    def predict(self, image: Image.Image) -> Prediction:
        start = time.perf_counter()
        tensor = self.transform(image).unsqueeze(0)
        with self.lock:
            if self.backend == "pytorch":
                with torch.inference_mode():
                    probability = float(self.model(tensor).softmax(dim=1)[0, 1])
            else:
                logits = self.session.run(["logits"], {"image": tensor.numpy()})[0]
                exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
                probability = float((exponentials / exponentials.sum(axis=1, keepdims=True))[0, 1])
        name, confidence = classify(probability, self.threshold)
        return Prediction(
            predicted_class=name,
            confidence=confidence,
            defect_probability=probability,
            model_version=self.version,
            decision_threshold=self.threshold,
            inference_time_ms=round((time.perf_counter() - start) * 1000, 3),
        )
