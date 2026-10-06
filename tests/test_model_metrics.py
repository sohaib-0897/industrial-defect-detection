import numpy as np
import pytest
import torch

from src.metrics import classify, metrics, select_threshold
from src.model import load_checkpoint


def test_model_output_and_checkpoint(checkpoint):
    model, metadata = load_checkpoint(checkpoint)
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 224, 224))
    assert output.shape == (2, 2)
    assert metadata["class_names"] == ["normal", "defective"]
    assert torch.isfinite(output).all()


def test_threshold_boundary_confidence():
    assert classify(0.4, 0.4) == ("defective", 0.4)
    assert classify(0.3, 0.4) == ("normal", 0.7)
    with pytest.raises(ValueError):
        classify(float("nan"), 0.5)


def test_validation_recall_policy():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.45, 0.4, 0.8])
    threshold, table = select_threshold(labels, probabilities, target_recall=1.0)
    assert threshold == 0.4
    result = metrics(labels, probabilities, threshold)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["recall"] == 1.0
    assert any(row["threshold"] == 0.5 for row in table)
