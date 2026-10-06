"""Copy generated evidence to the small, versioned submission report directory."""

import argparse
import json
import shutil
from pathlib import Path


def collect(run: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "history.json",
        "threshold_selection.json",
        "evaluation.json",
        "classification_report.json",
        "test_predictions.json",
        "confusion_matrix.png",
        "best.json",
        "benchmark.json",
        "api-pytorch.json",
        "api-onnx.json",
    ):
        source = run / name
        if source.exists():
            shutil.copy2(source, output / name)
    if (run / "errors").exists():
        shutil.copytree(run / "errors", output / "errors", dirs_exist_ok=True)
    evaluation = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    if evaluation.get("status") != "complete":
        raise ValueError("Only completed real evaluation evidence may be published")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selection = json.loads((run / "threshold_selection.json").read_text(encoding="utf-8"))
    for name in ("precision", "recall", "f1"):
        plt.plot([r["threshold"] for r in selection["tradeoff"]], [r[name] for r in selection["tradeoff"]], label=name)
    plt.axvline(selection["selected_threshold"], color="black", linestyle="--", label="selected on validation")
    plt.axhline(selection["target_recall"], color="gray", linestyle=":", label="recall target")
    plt.xlabel("Defective probability threshold")
    plt.ylabel("Validation metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "threshold_tradeoff.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/run-001"))
    parser.add_argument("--output", type=Path, default=Path("reports/ksdd2"))
    args = parser.parse_args()
    collect(args.run, args.output)
