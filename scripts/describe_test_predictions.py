"""Compute ranking summaries from saved final predictions; never load a model."""

import hashlib
import json
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score


def describe(path: Path) -> dict:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    rows = json.loads(content)
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Expected unique saved test predictions")
    if any(row["ground_truth"] not in {"normal", "defective"} for row in rows):
        raise ValueError("Unexpected ground-truth class")
    probabilities = [row["defect_probability"] for row in rows]
    if any(not 0 <= probability <= 1 for probability in probabilities):
        raise ValueError("Invalid saved probability")
    labels = [int(row["ground_truth"] == "defective") for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("Both classes are needed for ROC-AUC")
    return {
        "scope": "Post-evaluation descriptive statistics from stored official-test predictions only",
        "positive_class": "defective",
        "prediction_file": path.as_posix(),
        "prediction_sha256": hashlib.sha256(content).hexdigest(),
        "prediction_hash_encoding": "UTF-8 JSON bytes with CRLF normalized to LF for cross-platform checkouts",
        "sample_count": len(rows),
        "positive_support": sum(labels),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "method": "scikit-learn average_precision_score (stepwise PR summary, not trapezoidal PR-AUC) and roc_auc_score",
        "confidence_intervals": "Not reported; independent product/session sampling is not established",
        "selection_effect": "None: no inference, retraining, threshold selection or primary-result modification",
    }


if __name__ == "__main__":
    folder = Path("reports/ksdd2")
    report = describe(folder / "test_predictions.json")
    (folder / "post_evaluation_statistics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
