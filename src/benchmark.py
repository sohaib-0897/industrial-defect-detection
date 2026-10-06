import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch

from app.config import Settings
from app.inference import Predictor
from src.common import write_json
from src.data import read_image


def benchmark(checkpoint: Path, sample: Path, repeats: int = 50, warmup: int = 10, threads: int = 2) -> dict:
    if repeats < 1 or warmup < 1:
        raise ValueError("Repeats and warmup must be positive")
    image = read_image(sample)
    results = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": str(torch.__version__),
        "threads": threads,
        "batch_size": 1,
        "repeats": repeats,
        "warmup": warmup,
        "scope": "RGB preprocessing + model + threshold; excludes decoding, HTTP and startup",
        "sample": str(sample),
        "backends": {},
    }
    for backend, model_path in [("pytorch", checkpoint), ("onnx", checkpoint.with_suffix(".onnx"))]:
        if not model_path.exists():
            continue
        predictor = Predictor(Settings(model_path=model_path, backend=backend, threads=threads))
        for _ in range(warmup):
            predictor.predict(image)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            predictor.predict(image)
            durations.append((time.perf_counter() - start) * 1000)
        results["backends"][backend] = {
            "mean_ms": float(np.mean(durations)),
            "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
        }
    write_json(checkpoint.parent / "benchmark.json", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/run-001/best.pt"))
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    benchmark(args.checkpoint, args.sample, args.repeats, args.warmup, args.threads)
