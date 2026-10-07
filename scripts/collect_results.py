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
    pilot = run.parent / "rotation-pilot" / "history.json"
    if pilot.exists():
        from src.common import write_json

        pilot_history = json.loads(pilot.read_text(encoding="utf-8"))
        comparison = run.parent / "photometric-pilot" / "history.json"
        final_history = json.loads(
            (comparison if comparison.exists() else run / "history.json").read_text(encoding="utf-8")
        )
        matched = [
            {
                "epoch": old["epoch"],
                "rotation_validation_f1_at_0_5": old["f1"],
                "photometric_validation_f1_at_0_5": new["f1"],
            }
            for old, new in zip(pilot_history, final_history)
        ]
        write_json(
            output / "augmentation_review.json",
            {
                "selection_data": "validation only; official test not evaluated in either pilot",
                "same_settings": "seed 42, official training split, ResNet18, batch 16, weighted CE, stage schedule and learning rates",
                "pilot": "expanded +/-2 degree rotation + +/-10% photometric jitter; stopped after 4 completed epochs for ablation",
                "final": "photometric jitter only; no geometric augmentation",
                "matched_epoch_comparison": matched,
                "interpretation": "Rotated training images visibly gain gray borders; improved validation behavior supports removing rotation. This is not a causal proof or a repeated-seed study.",
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/run-001"))
    parser.add_argument("--output", type=Path, default=Path("reports/ksdd2"))
    args = parser.parse_args()
    collect(args.run, args.output)
