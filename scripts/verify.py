"""Repeatable synthetic end-to-end verification; all quality numbers are fixtures."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.check_api import check
from src.common import write_json


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], check=True)


def verify(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh output directory for verification")
    data, audit, model = output / "data", output / "audit", output / "run"
    sample = data / "normal" / "SYNTHETIC_normal_000.png"
    run("-m", "scripts.make_synthetic", "--output", str(data))
    run("-m", "src.analyze_data", "--data", str(data), "--output", str(audit))
    run(
        "-m",
        "src.train",
        "--data",
        str(data),
        "--manifest",
        str(audit / "splits.csv"),
        "--output",
        str(model),
        "--epochs",
        "2",
        "--freeze-epochs",
        "1",
        "--batch-size",
        "8",
    )
    run(
        "-m",
        "src.evaluate",
        "--data",
        str(data),
        "--manifest",
        str(audit / "splits.csv"),
        "--checkpoint",
        str(model / "best.pt"),
    )
    run("-m", "src.error_analysis", "--data", str(data), "--run", str(model))
    run(
        "-m",
        "src.export_onnx",
        "--checkpoint",
        str(model / "best.pt"),
        "--samples",
        str(sample),
        str(data / "defective" / "SYNTHETIC_defective_000.png"),
    )
    run("-m", "src.benchmark", "--checkpoint", str(model / "best.pt"), "--sample", str(sample))
    run("-m", "pytest", "-q")
    run("-m", "ruff", "check", ".")
    run("-m", "ruff", "format", "--check", ".")
    report = {
        "scope": "synthetic integration only; no manufacturing quality claim",
        "checks": {"pytest": "passed", "ruff_check": "passed", "ruff_format": "passed"},
        "api": [check(model / "best.pt", sample, backend) for backend in ("pytorch", "onnx")],
    }
    for name, path in [
        ("dataset", audit / "dataset_analysis.json"),
        ("evaluation", model / "evaluation.json"),
        ("onnx", model / "best.json"),
        ("latency", model / "benchmark.json"),
    ]:
        report[name] = json.loads(path.read_text(encoding="utf-8"))
    write_json(output / "verification.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke-reproduce"))
    verify(parser.parse_args().output)
