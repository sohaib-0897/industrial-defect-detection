import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat

from src.common import write_json
from src.data import read_image


def analyze(data: Path, run: Path, limit: int = 12) -> None:
    predictions = json.loads((run / "test_predictions.json").read_text(encoding="utf-8"))
    output = run / "errors"
    output.mkdir(exist_ok=True)
    errors = []
    for row in predictions:
        if row["ground_truth"] == row["predicted_class"]:
            continue
        kind = "false_negative" if row["ground_truth"] == "defective" else "false_positive"
        image = read_image(data / row["path"])
        gray = image.convert("L")
        statistics = ImageStat.Stat(gray)
        errors.append(
            {
                **row,
                "error_type": kind,
                "mean_brightness": statistics.mean[0],
                "contrast_std": statistics.stddev[0],
                "width": image.width,
                "height": image.height,
                "review_notes": "",
            }
        )
    write_json(output / "errors.json", errors)
    counts = {kind: sum(e["error_type"] == kind for e in errors) for kind in ("false_positive", "false_negative")}
    for kind in counts:
        selected = sorted((e for e in errors if e["error_type"] == kind), key=lambda e: -e["confidence"])[:limit]
        if not selected:
            continue
        sheet = Image.new("RGB", (600, 300 * ((len(selected) + 1) // 2)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(selected):
            x, y = index % 2 * 300, index // 2 * 300
            image = read_image(data / row["path"])
            image.thumbnail((280, 220))
            sheet.paste(image, (x + 10, y + 5))
            text = (
                f"{Path(row['path']).name[:36]}\ntrue={row['ground_truth']} pred={row['predicted_class']}"
                f"\nconfidence={row['confidence']:.6g}\np(defect)={row['defect_probability']:.8f}"
            )
            draw.text((x + 10, y + 225), text, fill="black")
        sheet.save(output / f"{kind}.jpg")
    (output / "review.md").write_text(
        f"# Error review\n\nCounts: {counts}. Highest-confidence examples saved by type.\n\n"
        "Brightness and contrast are descriptive cues, not explanations. Inspect lighting, blur, small/edge defects, "
        "background, orientation, occlusion, low contrast, label noise and product domain. Add verified notes to errors.json. "
        "Do not change the model or threshold after reviewing test errors without obtaining a fresh holdout.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=Path("artifacts/run-001"))
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    analyze(args.data, args.run, args.limit)
