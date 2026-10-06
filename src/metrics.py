import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.common import CLASS_NAMES


def classify(probability: float, threshold: float) -> tuple[str, float]:
    if not 0 <= threshold <= 1 or not 0 <= probability <= 1:
        raise ValueError("Probability and threshold must lie in [0, 1]")
    defective = probability >= threshold
    return CLASS_NAMES[int(defective)], probability if defective else 1 - probability


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "per_class": classification_report(
            labels, predicted, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }


def select_threshold(
    labels: np.ndarray, probabilities: np.ndarray, target_recall: float = 0.95
) -> tuple[float, list[dict]]:
    if not 0 < target_recall <= 1:
        raise ValueError("Target recall must be in (0, 1]")
    if set(labels) != {0, 1}:
        raise ValueError("Validation must contain both classes")
    candidates = np.unique(np.r_[0.0, probabilities, 0.5, 1.0])
    # Sorted cumulative counts inspect all boundaries without repeated sklearn reports.
    order = np.argsort(probabilities)
    prefix = np.r_[0, np.cumsum(labels[order])]
    positives = int(labels.sum())
    table = []
    for threshold in candidates:
        below = int(np.searchsorted(probabilities[order], threshold, side="left"))
        tp = positives - int(prefix[below])
        fp = len(labels) - below - tp
        tn = below - int(prefix[below])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        table.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": (tp + tn) / len(labels),
            }
        )
    feasible = [row for row in table if row["recall"] >= target_recall]
    chosen = max(feasible, key=lambda row: (row["precision"], row["f1"], row["threshold"]))
    return chosen["threshold"], table
