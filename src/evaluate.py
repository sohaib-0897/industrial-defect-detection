import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from src.common import CLASS_NAMES, configure_logging, file_hash, write_json
from src.data import load_manifest
from src.metrics import classify, metrics
from src.model import load_checkpoint
from src.train import loader, predict


def evaluate(data: Path, manifest: Path, checkpoint_path: Path, batch_size: int = 16) -> dict:
    model, checkpoint = load_checkpoint(checkpoint_path)
    if file_hash(manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from training manifest")
    rows = [r for r in load_manifest(manifest, data) if r["split"] == "test"]
    output = checkpoint_path.parent
    result_path = output / "evaluation.json"
    # Exclusive creation protects the held-out result from accidental repeated evaluation.
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump({"status": "evaluation_in_progress"}, stream)
    labels, probabilities, _ = predict(model, loader(rows, data, batch_size), "cpu")
    result = metrics(labels, probabilities, checkpoint["threshold"])
    result.update(
        {
            "status": "complete",
            "test_count": len(rows),
            "checkpoint_sha256": file_hash(checkpoint_path),
            "manifest_sha256": file_hash(manifest),
            "model_version": checkpoint["model_version"],
            "small_sample_warning": "Test uncertainty is high; independent production data is required."
            if len(rows) < 100
            else None,
        }
    )
    predictions = []
    for row, label, probability in zip(rows, labels, probabilities, strict=True):
        predicted_class, confidence = classify(float(probability), checkpoint["threshold"])
        predictions.append(
            {
                "path": row["path"],
                "ground_truth": CLASS_NAMES[int(label)],
                "predicted_class": predicted_class,
                "confidence": confidence,
                "defect_probability": float(probability),
                "group": row["group"],
            }
        )
    write_json(output / "test_predictions.json", predictions)
    write_json(output / "classification_report.json", result["per_class"])
    tn, fp = result["confusion_matrix"][0]
    fn, tp = result["confusion_matrix"][1]
    result.update({"true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp})
    display = ConfusionMatrixDisplay(np.asarray(result["confusion_matrix"]), display_labels=CLASS_NAMES)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Held-out test set; rows=true, columns=predicted")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close()
    write_json(result_path, result)
    logging.getLogger(__name__).info("Test evaluation: %s", json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out test once at the validation-selected threshold")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/splits.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/run-001/best.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    configure_logging()
    torch.set_num_threads(args.threads)
    evaluate(args.data, args.manifest, args.checkpoint, args.batch_size)


if __name__ == "__main__":
    main()
