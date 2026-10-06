"""Run final test once, diagnostics, export, benchmark and real HTTP checks."""

from pathlib import Path

import torch

from scripts.check_api import check
from scripts.collect_results import collect
from src.benchmark import benchmark
from src.error_analysis import analyze
from src.evaluate import evaluate
from src.export_onnx import export
from src.gradcam import render


def main() -> None:
    data = Path("data/KolektorSDD2")
    run = Path("artifacts/run-001")
    checkpoint = run / "best.pt"
    if not (run / "threshold_selection.json").exists():
        raise ValueError("Training and validation threshold selection must finish first")
    torch.set_num_threads(4)
    print("Final official test evaluation (once)", flush=True)
    evaluate(data, Path("reports/ksdd2/splits.csv"), checkpoint, batch_size=4)
    print("Error contact sheets", flush=True)
    analyze(data, run)
    print("Post-evaluation Grad-CAM", flush=True)
    render(data, checkpoint, Path("reports/ksdd2/gradcam"))
    samples = [data / "train/10000.png", data / "train/10021.png", data / "train/10045.png"]
    print("ONNX export and probability/class parity", flush=True)
    export(checkpoint, samples)
    print("CPU benchmarks", flush=True)
    benchmark(checkpoint, samples[0])
    for backend in ("pytorch", "onnx"):
        print(f"Real HTTP health and both-class predictions: {backend}", flush=True)
        check(checkpoint, samples[0], backend, samples[1:])
    collect(run, Path("reports/ksdd2"))
    print("Generated real reports collected", flush=True)


if __name__ == "__main__":
    main()
