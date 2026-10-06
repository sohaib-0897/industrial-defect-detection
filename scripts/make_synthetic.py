"""Make integration fixtures, never assessment data or quality evidence."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create(root: Path, per_class: int = 40) -> None:
    if root.exists() and any(root.iterdir()):
        raise ValueError("Synthetic output must be an empty directory")
    rng = np.random.default_rng(101)
    for label in ("normal", "defective"):
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            # Independent random coarse textures avoid near-identical fixture groups.
            coarse = rng.integers(60, 190, (16, 16, 3), dtype=np.uint8)
            image = Image.fromarray(coarse).resize((256, 256), Image.Resampling.BILINEAR)
            if label == "defective":
                draw = ImageDraw.Draw(image)
                x, y = rng.integers(40, 180, size=2)
                draw.line((int(x), int(y), int(x + 35), int(y + 30)), fill="black", width=6)
            image.save(folder / f"SYNTHETIC_{label}_{index:03d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke/data"))
    parser.add_argument("--per-class", type=int, default=40)
    args = parser.parse_args()
    create(args.output, args.per_class)
