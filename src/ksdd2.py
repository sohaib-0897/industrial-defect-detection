"""Audit the observed KSDD2 image/mask layout and preserve its official test set."""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

from src.common import file_hash, write_json
from src.data import read_image

SOURCE = "https://www2.vicos.si/resources/kolektorsdd2/"
EXPECTED = {"train": {"normal": 2085, "defective": 246}, "test": {"normal": 894, "defective": 110}}


def mask_label(mask: Path, size: tuple[int, int]) -> tuple[str, int, list[int]]:
    with Image.open(mask) as image:
        image.load()
        if image.size != size or image.mode != "L":
            raise ValueError(f"Mask size/mode mismatch: {mask}")
        values = np.asarray(image)
        unique = np.unique(values).tolist()
        if not set(unique) <= {0, 255}:
            raise ValueError(f"Unexpected mask values: {mask}: {unique}")
        pixels = int(np.count_nonzero(values))
        return "defective" if pixels else "normal", pixels, unique


def official_split(rows: list[dict], seed: int = 42) -> list[dict]:
    official_train = [row for row in rows if row["official_split"] == "train"]
    train, val = train_test_split(
        official_train, test_size=0.2, random_state=seed, stratify=[row["label"] for row in official_train]
    )
    for partition, split in ((train, "train"), (val, "val")):
        for row in partition:
            row["split"] = split
    for row in rows:
        if row["official_split"] == "test":
            row["split"] = "test"
    return rows


def audit(root: Path, output: Path, seed: int = 42, check_counts: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows, corrupt, excluded_copies = [], [], []
    formats, dimensions, modes, mask_values = Counter(), Counter(), Counter(), Counter()
    duplicates = defaultdict(list)
    for split in ("train", "test"):
        folder = root / split
        if not folder.is_dir():
            raise ValueError(f"Missing official directory {folder}")
        files = list(folder.glob("*.png"))
        # The official archive contains an extra byte-identical 10301 image/mask pair.
        # Verify the copies before excluding them; never silently discard unknown files.
        for path in files:
            if " (copy)" in path.stem:
                canonical = path.with_name(path.name.replace(" (copy)", ""))
                if not canonical.exists() or file_hash(path) != file_hash(canonical):
                    raise ValueError(f"Unverified archive copy: {path}")
                excluded_copies.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "canonical": canonical.relative_to(root).as_posix(),
                        "sha256": file_hash(path),
                    }
                )
        files = [p for p in files if " (copy)" not in p.stem]
        images = sorted(p for p in files if not p.stem.endswith("_GT"))
        masks = {p for p in files if p.stem.endswith("_GT")}
        if masks != {path.with_name(path.stem + "_GT.png") for path in images}:
            raise ValueError(f"Missing or orphan annotation in {folder}")
        for path in images:
            relative = path.relative_to(root).as_posix()
            mask = path.with_name(path.stem + "_GT.png")
            try:
                image = read_image(path)
                label, pixels, values = mask_label(mask, image.size)
                with Image.open(path) as raw:
                    formats[raw.format] += 1
                    modes[raw.mode] += 1
                dimensions[f"{image.width}x{image.height}"] += 1
                mask_values[str(values)] += 1
                pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                duplicates[pixel_hash].append(relative)
                rows.append(
                    {
                        "path": relative,
                        "label": label,
                        "group": pixel_hash,
                        "sha256": file_hash(path),
                        "mask": mask.relative_to(root).as_posix(),
                        "mask_sha256": file_hash(mask),
                        "official_split": split,
                        "defect_pixels": pixels,
                        "defect_fraction": pixels / (image.width * image.height),
                    }
                )
            except (OSError, ValueError) as error:
                corrupt.append({"path": relative, "error": str(error)})
    counts = {
        split: dict(Counter(r["label"] for r in rows if r["official_split"] == split)) for split in ("train", "test")
    }
    if corrupt or (check_counts and counts != EXPECTED):
        write_json(output / "audit_failure.json", {"corrupt": corrupt, "observed": counts, "expected": EXPECTED})
        raise ValueError("Dataset audit failed; inspect audit_failure.json")
    exact = [paths for paths in duplicates.values() if len(paths) > 1]
    if exact:
        raise ValueError(f"Exact decoded duplicates require explicit leakage review: {exact[:5]}")
    official_split(rows, seed)
    with (output / "splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split_counts = {
        split: dict(Counter(r["label"] for r in rows if r["split"] == split)) for split in ("train", "val", "test")
    }
    report = {
        "source": SOURCE,
        "license": "CC BY-NC-SA 4.0",
        "seed": seed,
        "image_count": len(rows),
        "annotation_count": len(rows),
        "official_counts": counts,
        "split_counts": split_counts,
        "class_counts": dict(Counter(r["label"] for r in rows)),
        "formats": dict(formats),
        "dimensions_width_x_height": dict(dimensions),
        "image_modes": dict(modes),
        "mask_values": dict(mask_values),
        "corrupt": corrupt,
        "excluded_byte_identical_archive_copies": excluded_copies,
        "raw_png_file_count": sum(1 for _ in root.rglob("*.png")),
        "exact_decoded_duplicates": exact,
        "duplicate_scope": "All image SHA256 and decoded RGB pixel hashes; no perceptual or product identity guarantee",
        "normal_to_defective_ratio": sum(r["label"] == "normal" for r in rows)
        / sum(r["label"] == "defective" for r in rows),
        "split_policy": "Stratified 80/20 official training only, seed 42; official test preserved",
        "other_files": [
            p.relative_to(root).as_posix() for p in sorted(root.rglob("*")) if p.is_file() and p.suffix != ".png"
        ],
        "manifest_sha256": file_hash(output / "splits.csv"),
    }
    write_json(output / "dataset_analysis.json", report)
    names = ["train", "val", "test"]
    x = np.arange(3)
    for offset, label in ((-0.18, "normal"), (0.18, "defective")):
        plt.bar(x + offset, [split_counts[s][label] for s in names], width=0.36, label=label)
    plt.xticks(x, names)
    plt.ylabel("Images (original class prevalence)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "class_distribution.png", dpi=150)
    plt.close()
    # Only official training examples inform preprocessing/augmentation choices.
    selected = [
        r for label in ("normal", "defective") for r in rows if r["official_split"] == "train" and r["label"] == label
    ]
    selected = [r for label in ("normal", "defective") for r in selected if r["label"] == label][:4] + [
        r for r in selected if r["label"] == "defective"
    ][:4]
    sheet = Image.new("RGB", (1000, 1240), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        image = read_image(root / row["path"])
        with Image.open(root / row["mask"]) as mask:
            box = mask.getbbox()
        if box:
            ImageDraw.Draw(image).rectangle(box, outline="red", width=2)
        image.thumbnail((230, 570))
        x0, y0 = (index % 4) * 250, (index // 4) * 620
        sheet.paste(image, (x0 + 10, y0 + 10))
        draw.text((x0 + 10, y0 + 585), f"{row['path']}\n{row['label']} ({row['defect_pixels']} mask px)", fill="black")
    sheet.save(output / "training_examples.jpg", quality=90)
    (output / "dataset_analysis.md").write_text(
        "# KSDD2 audit\n\n"
        + f"Source: {SOURCE}. License: CC BY-NC-SA 4.0.\n\n"
        + f"Observed {len(rows)} images and paired 8-bit grayscale masks; nonzero mask = defective.\n\n"
        + f"Official counts: `{counts}`.\n\nSeed {seed}, stratified 80/20 official training: `{split_counts}`.\n\n"
        + f"Normal:defective imbalance = {report['normal_to_defective_ratio']:.3f}:1. Accuracy alone hides missed defects.\n\n"
        + f"Formats: {dict(formats)}; dimensions (width x height): {dict(dimensions)}.\n\n"
        + f"Excluded verified byte-identical archive copies: `{excluded_copies}`.\n\n"
        + "All canonical images and masks decoded successfully. No exact decoded image duplicates after removing the archive copy pair. Near duplicates and "
        "same-product capture groups are not established by this audit. No product IDs are supplied. "
        "The official test set is never repartitioned. Masks provide binary labels and review overlays, not model inputs.\n\n"
        "![Class distribution](class_distribution.png)\n\n![Training examples; red = annotation box](training_examples.jpg)\n\n"
        "Jakob Božič, Domen Tabernik, Danijel Skočaj. Mixed supervision for surface-defect detection: "
        "from weakly to fully supervised learning. Computers in Industry, 2021.\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/KolektorSDD2"))
    parser.add_argument("--output", type=Path, default=Path("reports/ksdd2"))
    args = parser.parse_args()
    print(audit(args.data, args.output))
