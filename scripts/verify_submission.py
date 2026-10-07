"""Check stored assessment evidence and README links without repeating test inference."""

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from src.common import file_hash


def verify() -> None:
    folder = Path("reports/ksdd2")

    def read(name):
        return json.loads((folder / name).read_text(encoding="utf-8"))

    evaluation, model, selection = (read(name) for name in ("evaluation.json", "best.json", "threshold_selection.json"))
    assert evaluation["status"] == "complete"
    assert evaluation["checkpoint_sha256"] == model["checkpoint_sha256"] == file_hash(Path("artifacts/run-001/best.pt"))
    assert model["onnx_sha256"] == file_hash(Path("artifacts/run-001/best.onnx"))
    assert model["threshold"] == selection["selected_threshold"]
    assert evaluation["manifest_sha256"] == file_hash(folder / "splits.csv")
    with (folder / "splits.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3335
    assert all(row["split"] == "test" for row in rows if row["official_split"] == "test")
    predictions = read("test_predictions.json")
    assert {row["path"] for row in predictions} == {row["path"] for row in rows if row["split"] == "test"}
    counts = Counter((row["ground_truth"], row["predicted_class"]) for row in predictions)
    assert evaluation["confusion_matrix"] == [
        [counts[(actual, predicted)] for predicted in ("normal", "defective")] for actual in ("normal", "defective")
    ]
    assert len(read("errors/errors.json")) == evaluation["false_positives"] + evaluation["false_negatives"]
    for backend in ("pytorch", "onnx"):
        api = read(f"api-{backend}.json")
        assert api["health_status"] == 200 and api["malformed_image_status"] == 422
        assert api["predictions"][0]["predicted_class"] == "normal"
        assert api["predictions"][1]["predicted_class"] == "defective"
    for document in (Path("README.md"), *Path("docs").glob("*.md")):
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if link.startswith(("https://", "http://", "#")):
                continue
            assert (document.parent / link.split("#")[0]).exists(), f"Missing link: {document}: {link}"
    tracked = subprocess.check_output(["git", "ls-files", "-z"], text=True).split("\0")
    assert not any(
        path.startswith(("data/", ".venv/", ".venv-gpu/")) or path.endswith((".pt", ".onnx", ".zip"))
        for path in tracked
    )
    print(
        "Submission evidence, model/manifest hashes, preserved test membership, API reports, documentation links and Git artifact hygiene passed."
    )


if __name__ == "__main__":
    verify()
