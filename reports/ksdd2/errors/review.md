Visually inspected six highest-confidence false positives and six highest-confidence false negatives, using full images and enlarged annotation crops. All 43 errors are recorded; descriptive notes apply only to inspected examples.

**False positives:** normal images show pale flecks, strong vertical/diagonal striations, mottled texture and uneven brightness. Examples include the isolated fleck in `test/20719.png` and pronounced vertical texture in `test/20008.png`. Their official masks are empty. These cues resemble defect patterns visually; this does not establish label errors or the classifier's causal mechanism.

**False negatives:** several misses are thin, small or weak-contrast marks, including `test/20816.png` (181 annotated pixels) and `test/20259.png` (323 pixels). The scratch in `test/20142.png` lies along the upper-left boundary. `test/20969.png` has a larger, low-contrast patch (3,390 pixels), so size alone does not explain all misses. The full frame is preserved; no crop removed these regions. Network downsampling/global pooling may weaken localized cues, but that remains a hypothesis.

| Filename | Ground truth | Prediction | p(defective) | Class probability confidence | Annotated pixels |
|---|---|---|---:|---:|---:|
| test/20139.png | defective | normal | 0.028051 | 0.971949 | 559 |
| test/20142.png | defective | normal | 0.034778 | 0.965222 | 1763 |
| test/20259.png | defective | normal | 0.034403 | 0.965597 | 323 |
| test/20801.png | defective | normal | 0.036984 | 0.963016 | 429 |
| test/20816.png | defective | normal | 0.030095 | 0.969905 | 181 |
| test/20969.png | defective | normal | 0.033948 | 0.966052 | 3390 |
| test/20008.png | normal | defective | 0.777744 | 0.777744 | 0 |
| test/20201.png | normal | defective | 0.461324 | 0.461324 | 0 |
| test/20651.png | normal | defective | 0.913001 | 0.913001 | 0 |
| test/20719.png | normal | defective | 0.938966 | 0.938966 | 0 |
| test/20879.png | normal | defective | 0.669904 | 0.669904 | 0 |
| test/20950.png | normal | defective | 0.532558 | 0.532558 | 0 |

![Reviewed false positives](false_positive.jpg)
![Reviewed false negatives, with annotated crops](false_negative.jpg)

Findings were recorded after the final test evaluation. No threshold adjustment, retraining or model selection followed this review.
