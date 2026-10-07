"""Attach descriptive findings from visual review without altering generated predictions."""

import json
from pathlib import Path

from src.common import write_json

NOTES = {
    "test/20719.png": "Isolated pale fleck against textured background; official mask is empty.",
    "test/20651.png": "Dark surface with a faint horizontal light mark and uneven directional texture; mask empty.",
    "test/20008.png": "Pronounced vertical ridged/streaked texture; official mask is empty.",
    "test/20879.png": "Broad mottled brightness and texture variation; official mask is empty.",
    "test/20950.png": "Diagonal/wavy surface striations; official mask is empty.",
    "test/20201.png": "Small pale specks against rough directional texture; official mask is empty.",
    "test/20139.png": "Small, low-contrast vertical mark visible in the annotation crop.",
    "test/20816.png": "Very small, short thin horizontal mark near the left boundary.",
    "test/20969.png": "Larger but subtle low-contrast patch blending into the lower surface texture.",
    "test/20259.png": "Thin short vertical mark with weak contrast against textured background.",
    "test/20142.png": "Narrow vertical scratch along the upper-left image boundary.",
    "test/20801.png": "Short intermittent pale horizontal streak near the upper-left boundary.",
}


def build() -> None:
    folder = Path("artifacts/run-001/errors")
    errors = json.loads((folder / "errors.json").read_text(encoding="utf-8"))
    for row in errors:
        row["review_notes"] = NOTES.get(row["path"], "")
    write_json(folder / "errors.json", errors)
    rows = sorted((row for row in errors if row["path"] in NOTES), key=lambda row: (row["error_type"], row["path"]))
    table = "\n".join(
        f"| {row['path']} | {row['ground_truth']} | {row['predicted_class']} | "
        f"{row['defect_probability']:.6f} | {row['confidence']:.6f} | {row['annotation_pixels']} |"
        for row in rows
    )
    review = f"""Visually inspected six highest-confidence false positives and six highest-confidence false negatives, using full images and enlarged annotation crops. All 43 errors are recorded; descriptive notes apply only to inspected examples.

**False positives:** normal images show pale flecks, strong vertical/diagonal striations, mottled texture and uneven brightness. Examples include the isolated fleck in `test/20719.png` and pronounced vertical texture in `test/20008.png`. Their official masks are empty. These cues resemble defect patterns visually; this does not establish label errors or the classifier's causal mechanism.

**False negatives:** several misses are thin, small or weak-contrast marks, including `test/20816.png` (181 annotated pixels) and `test/20259.png` (323 pixels). The scratch in `test/20142.png` lies along the upper-left boundary. `test/20969.png` has a larger, low-contrast patch (3,390 pixels), so size alone does not explain all misses. The full frame is preserved; no crop removed these regions. Network downsampling/global pooling may weaken localized cues, but that remains a hypothesis.

| Filename | Ground truth | Prediction | p(defective) | Class probability confidence | Annotated pixels |
|---|---|---|---:|---:|---:|
{table}

![Reviewed false positives](false_positive.jpg)
![Reviewed false negatives, with annotated crops](false_negative.jpg)

Findings were recorded after the final test evaluation. No threshold adjustment, retraining or model selection followed this review.
"""
    (folder / "review.md").write_text(review, encoding="utf-8")


if __name__ == "__main__":
    build()
