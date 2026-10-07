"""Audit committed evidence without downloaded images or trained model artifacts."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.describe_test_predictions import describe
from src.metrics import metrics

REPORTS = Path("reports/ksdd2")


def test_frozen_submission_counts_and_operating_point():
    with (REPORTS / "splits.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3335
    assert Counter(row["label"] for row in rows) == {"normal": 2979, "defective": 356}
    assert Counter((row["split"], row["label"]) for row in rows) == {
        ("train", "normal"): 1667,
        ("train", "defective"): 197,
        ("val", "normal"): 418,
        ("val", "defective"): 49,
        ("test", "normal"): 894,
        ("test", "defective"): 110,
    }
    assert all(row["split"] == "test" for row in rows if row["official_split"] == "test")
    stored = json.loads((REPORTS / "evaluation.json").read_text(encoding="utf-8"))
    expected = {
        "accuracy": 0.9571713147410359,
        "precision": 0.7723577235772358,
        "recall": 0.8636363636363636,
        "f1": 0.8154506437768241,
        "threshold": 0.1374264806509018,
        "confusion_matrix": [[866, 28], [15, 95]],
    }
    assert all(stored[name] == value for name, value in expected.items())
    for file, key in (("best.json", "threshold"), ("threshold_selection.json", "selected_threshold")):
        assert json.loads((REPORTS / file).read_text(encoding="utf-8"))[key] == expected["threshold"]
    predictions = json.loads((REPORTS / "test_predictions.json").read_text(encoding="utf-8"))
    assert {row["path"] for row in predictions} == {row["path"] for row in rows if row["split"] == "test"}
    recomputed = metrics(
        np.array([row["ground_truth"] == "defective" for row in predictions], dtype=int),
        np.array([row["defect_probability"] for row in predictions]),
        expected["threshold"],
    )
    assert all(recomputed[name] == value for name, value in expected.items())
    latency = json.loads((REPORTS / "benchmark.json").read_text(encoding="utf-8"))["backends"]
    assert f"{latency['pytorch']['median_ms']:.2f}" == "492.79"
    assert f"{latency['onnx']['median_ms']:.2f}" == "310.01"


def test_post_evaluation_statistics_provenance_and_reproducibility(tmp_path):
    stored = json.loads((REPORTS / "post_evaluation_statistics.json").read_text(encoding="utf-8"))
    assert stored == describe(REPORTS / "test_predictions.json")
    assert stored["sample_count"] == 1004 and stored["positive_support"] == 110
    content = (REPORTS / "test_predictions.json").read_bytes().replace(b"\r\n", b"\n")
    for variant in (content, content.replace(b"\n", b"\r\n")):
        temporary = tmp_path / "predictions.json"
        temporary.write_bytes(variant)
        portable = describe(temporary)
        for key in ("prediction_sha256", "average_precision", "roc_auc"):
            assert portable[key] == stored[key]
